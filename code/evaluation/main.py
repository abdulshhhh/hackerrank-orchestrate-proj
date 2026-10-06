#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT if (CODE_ROOT / "dataset" / "requests.csv").is_file() else CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from buywait import AffordabilityAgent, DataRepository  # noqa: E402
from buywait.forecast import RESERVE_POLICIES  # noqa: E402


ZERO = Decimal("0")
ROUNDING_QUANTA = tuple(Decimal(value) for value in ("0.01", "0.1", "1", "10", "50", "100", "1000", "10000", "100000", "1000000"))


@dataclass
class Residual:
    request_id: str
    requested: Decimal
    currency: str
    expected: Decimal
    predicted: Decimal
    delta: Decimal
    delta_pct: Decimal
    predicted_status: str
    expected_status: str
    binding_constraint: str
    bucket: str
    bucket_detail: str
    pending_overlap_ids: tuple[str, ...]
    pending_counterfactual: Decimal
    rounding_quantum: Decimal | None
    shorter_horizon_days: int
    shorter_horizon_capacity: Decimal
    foreign_event_ids: tuple[str, ...]
    fx_alternative_ids: tuple[str, ...]
    fx_counterfactual: Decimal
    fx_audit_ok: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Buy or Wait? against the public samples")
    parser.add_argument(
        "--residuals",
        action="store_true",
        help="print a residual table and hypothesis classification sorted by absolute amount error",
    )
    parser.add_argument(
        "--compare-reserve-policies",
        action="store_true",
        help="compare all four reserve policies against the public samples",
    )
    parser.add_argument(
        "--compare-variable-reserves",
        action="store_true",
        help="grid-search essential-variable burn estimators and reserve horizons",
    )
    return parser.parse_args()


def capped_capacity(agent, request, profile, baseline) -> Decimal:
    request_date = date.fromisoformat(request["request_date"])
    opening = Decimal(profile["current_available_balance"])
    reserve = Decimal(profile["minimum_balance_to_keep"])
    requested = Decimal(request["requested_amount"])
    return min(requested, agent.forecast.capacity_today(request_date, opening, reserve, baseline, profile))


def binding_constraint(agent, request, profile, baseline, predicted: Decimal) -> str:
    requested = Decimal(request["requested_amount"])
    if predicted == requested:
        return "requested_amount cap"

    request_date = date.fromisoformat(request["request_date"])
    opening = Decimal(profile["current_available_balance"])
    reserve = Decimal(profile["minimum_balance_to_keep"])
    if opening <= reserve:
        return "minimum_balance_to_keep"

    pending = tuple(flow for flow in baseline if flow.amount < ZERO and flow.source == "pending")
    projected = tuple(flow for flow in baseline if flow.amount < ZERO and flow.source != "pending")
    full_capacity = agent.forecast.capacity_today(request_date, opening, reserve, baseline, profile)
    without_pending = agent.forecast.capacity_today(
        request_date, opening, reserve, tuple(flow for flow in baseline if flow not in pending), profile
    )
    without_projected = agent.forecast.capacity_today(
        request_date, opening, reserve, tuple(flow for flow in baseline if flow not in projected), profile
    )
    pending_effect = without_pending - full_capacity
    projected_effect = without_projected - full_capacity
    if pending_effect > ZERO and pending_effect >= projected_effect:
        return "pending debit reserve"
    if projected_effect > ZERO:
        return "projected essential outflow"
    return "minimum_balance_to_keep"


def duplicate_pending_counterfactual(agent, request, profile, baseline) -> tuple[Decimal, list[str]]:
    recurrences = [flow for flow in baseline if flow.amount < ZERO and flow.source == "recurrence"]
    duplicate_ids: list[str] = []
    duplicate_flows = []
    for pending in (flow for flow in baseline if flow.amount < ZERO and flow.source == "pending"):
        overlaps = any(
            recurrence.category == pending.category
            and abs((recurrence.date - pending.date).days) <= 3
            for recurrence in recurrences
        )
        if overlaps:
            duplicate_flows.append(pending)
            duplicate_ids.append(pending.event_id)
    adjusted = tuple(flow for flow in baseline if flow not in duplicate_flows)
    return capped_capacity(agent, request, profile, adjusted), duplicate_ids


def rounding_match(predicted: Decimal, expected: Decimal) -> Decimal | None:
    if predicted == expected:
        return None
    for quantum in ROUNDING_QUANTA:
        rounded = (predicted / quantum).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * quantum
        if rounded == expected:
            return quantum
    return None


def best_shorter_horizon(agent, request, profile, baseline, expected: Decimal) -> tuple[int, Decimal]:
    request_date = date.fromisoformat(request["request_date"])
    best_days = agent.forecast.horizon_days
    best_capacity = capped_capacity(agent, request, profile, baseline)
    best_error = abs(best_capacity - expected)
    for days in range(agent.forecast.horizon_days):
        cutoff = request_date + timedelta(days=days)
        shortened = tuple(flow for flow in baseline if flow.date <= cutoff)
        capacity = capped_capacity(agent, request, profile, shortened)
        error = abs(capacity - expected)
        if error < best_error:
            best_days, best_capacity, best_error = days, capacity, error
    return best_days, best_capacity


def fx_counterfactual(repository, agent, request, profile, baseline) -> tuple[Decimal, list[str], list[str], bool]:
    request_date = date.fromisoformat(request["request_date"])
    home = profile["home_currency"]
    event_by_id = {event["event_id"]: event for event in repository.events_by_user.get(request["user_id"], [])}
    changed_ids: list[str] = []
    foreign_ids: list[str] = []
    audited_ok = True
    adjusted = []
    explicit_sources = {"pending", "scheduled", "scheduled_credit", "failed_retry"}
    for flow in baseline:
        event = event_by_id.get(flow.event_id)
        if not event or flow.source not in explicit_sources or event["currency"] == home or not event["settlement_date"]:
            adjusted.append(flow)
            continue
        foreign_ids.append(flow.event_id)
        settlement = date.fromisoformat(event["settlement_date"])
        signed_settlement_value = repository.convert(repository.event_amount(event), event["currency"], home, settlement)
        if flow.amount < ZERO:
            signed_settlement_value = -signed_settlement_value
        audited_ok = audited_ok and flow.amount == signed_settlement_value
        request_rate = repository.rate_by_key.get((request_date.isoformat(), event["currency"], home))
        if request_rate is None:
            adjusted.append(flow)
            continue
        request_date_value = repository.event_amount(event) * request_rate
        if flow.amount < ZERO:
            request_date_value = -request_date_value
        adjusted.append(replace(flow, amount=request_date_value))
        if request_date_value != flow.amount:
            changed_ids.append(flow.event_id)
    return (
        capped_capacity(agent, request, profile, tuple(adjusted)),
        sorted(set(foreign_ids)),
        sorted(set(changed_ids)),
        audited_ok,
    )


def classify_residual(repository, agent, request, profile, baseline, expected, predicted) -> tuple[str, str]:
    delta = predicted - expected
    if abs(delta) <= Decimal("0.01"):
        return "exact", "within 0.01"

    pending_capacity, duplicate_ids = duplicate_pending_counterfactual(agent, request, profile, baseline)
    if duplicate_ids and delta < ZERO and abs(pending_capacity - expected) < abs(delta):
        return "(a) pending double-count", ",".join(duplicate_ids)

    quantum = rounding_match(predicted, expected)
    if quantum is not None:
        return "(b) rounding", f"round to {quantum}"

    horizon_days, horizon_capacity = best_shorter_horizon(agent, request, profile, baseline, expected)
    if delta < ZERO and horizon_days < agent.forecast.horizon_days and abs(horizon_capacity - expected) < abs(delta):
        return "(c) horizon", f"best cutoff {horizon_days}d -> {format_decimal(horizon_capacity)}"

    fx_capacity, _, fx_ids, audited_ok = fx_counterfactual(repository, agent, request, profile, baseline)
    if fx_ids and abs(fx_capacity - expected) < abs(delta):
        return "(d) FX date", f"request-date counterfactual improves: {','.join(fx_ids)}"
    if not audited_ok:
        return "(d) FX date", "settlement-date audit failed"

    return "other: forecast sizing/timing", "none of (a)-(d) improves the residual"


def format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def print_residuals(residuals: list[Residual]) -> None:
    ordered = sorted(residuals, key=lambda row: (-abs(row.delta), row.request_id))
    print("Residual analysis (signed delta = predicted - expected)")
    print()
    print("| request_id | requested_amount | currency | expected_safe | predicted_safe | signed_delta | delta_%_requested | our_status | expected_status | binding_constraint |")
    print("|---|---:|:---:|---:|---:|---:|---:|---|---|---|")
    for row in ordered:
        print(
            f"| {row.request_id} | {format_decimal(row.requested)} | {row.currency} | "
            f"{format_decimal(row.expected)} | {format_decimal(row.predicted)} | "
            f"{row.delta:+f} | {row.delta_pct:+.4f}% | {row.predicted_status} | "
            f"{row.expected_status} | {row.binding_constraint} |"
        )

    print()
    print("Hypothesis buckets (all 25 rows)")
    buckets: dict[str, list[str]] = {}
    for row in residuals:
        buckets.setdefault(row.bucket, []).append(row.request_id)
    bucket_order = (
        "(a) pending double-count", "(b) rounding", "(c) horizon", "(d) FX date",
        "other: forecast sizing/timing", "exact",
    )
    for bucket in bucket_order:
        request_ids = sorted(buckets.get(bucket, []))
        print(f"- {bucket}: {len(request_ids)}" + (f" ({', '.join(request_ids)})" if request_ids else ""))

    print()
    print("Hypothesis audit details")
    pending_candidates = [row for row in residuals if row.pending_overlap_ids]
    if pending_candidates:
        for row in sorted(pending_candidates, key=lambda item: item.request_id):
            print(
                f"- (a) {row.request_id}: overlap {','.join(row.pending_overlap_ids)}; "
                f"without it {format_decimal(row.pending_counterfactual)} vs expected {format_decimal(row.expected)} "
                f"(original {format_decimal(row.predicted)})"
            )
    else:
        print("- (a) no pending/recurrence overlaps found")
    rounded = [row for row in residuals if row.rounding_quantum is not None]
    print(
        "- (b) exact clean-unit rounding matches: "
        + (", ".join(f"{row.request_id}@{row.rounding_quantum}" for row in rounded) if rounded else "none")
    )
    shorter = [row for row in residuals if row.shorter_horizon_days < 90]
    print(
        "- (c) rows where a 0-89 day cutoff reduces absolute error: "
        + (", ".join(f"{row.request_id}@{row.shorter_horizon_days}d" for row in shorter) if shorter else "none")
    )
    foreign = [row for row in residuals if row.foreign_event_ids]
    if foreign:
        for row in sorted(foreign, key=lambda item: item.request_id):
            alternatives = ",".join(row.fx_alternative_ids) or "none available"
            print(
                f"- (d) {row.request_id}: foreign explicit events {','.join(row.foreign_event_ids)}; "
                f"settlement-date audit={'pass' if row.fx_audit_ok else 'FAIL'}; request-date alternatives={alternatives}"
            )
    else:
        print("- (d) no foreign explicit cash flows found")


def evaluate_reserve_policy(repository, samples, policy: str):
    agent = AffordabilityAgent(repository, reserve_policy=policy)
    metrics = Counter(rows=len(samples))
    residuals = []
    for sample in samples:
        decision, _ = agent.decide(sample)
        predicted = decision.as_row()
        profile = repository.profile_by_user[sample["user_id"]]
        expected = Decimal(sample["amount_safe_to_pay"])
        actual = Decimal(predicted["amount_safe_to_pay"])
        checks = {
            "amount": abs(actual - expected) <= Decimal("0.01"),
            "status": predicted["affordability_status"] == sample["affordability_status"],
            "method": predicted["recommended_payment_method"] == sample["recommended_payment_method"],
            "plan": predicted["payment_plan"] == sample["payment_plan"],
            "earliest": predicted["earliest_date_for_full_payment"] == sample["earliest_date_for_full_payment"],
            "changes": predicted["spending_changes_needed"] == sample["spending_changes_needed"],
        }
        for field, passed in checks.items():
            metrics[field] += int(passed)
        residuals.append((sample, predicted, profile, expected, actual))
    metrics["categorical"] = sum(metrics[field] for field in ("status", "method", "plan", "earliest", "changes"))
    total_absolute = sum(abs(actual - expected) for _, _, _, expected, actual in residuals)
    return agent, metrics, total_absolute, residuals


def compare_reserve_policies(repository, samples) -> int:
    results = {
        policy: evaluate_reserve_policy(repository, samples, policy)
        for policy in RESERVE_POLICIES
    }
    baseline_metrics = results["current_baseline"][1]
    ranked = sorted(
        RESERVE_POLICIES,
        key=lambda policy: (
            -results[policy][1]["amount"],
            -results[policy][1]["categorical"],
            results[policy][2],
            RESERVE_POLICIES.index(policy),
        ),
    )
    winner = ranked[0]
    categorical_loss = baseline_metrics["categorical"] - results[winner][1]["categorical"]
    if categorical_loss > 1:
        winner = "current_baseline"
    print("Reserve policy comparison")
    print()
    print("| policy | total_abs_residual | amount | status | method | plan | earliest | changes | categorical_total |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy in RESERVE_POLICIES:
        metrics = results[policy][1]
        print(
            f"| {policy} | {format_decimal(results[policy][2])} | {metrics['amount']}/25 | "
            f"{metrics['status']}/25 | {metrics['method']}/25 | {metrics['plan']}/25 | "
            f"{metrics['earliest']}/25 | {metrics['changes']}/25 | {metrics['categorical']}/125 |"
        )
    print()
    print(f"Selected policy: {winner}")
    if ranked[0] != winner:
        print(
            f"Rejected {ranked[0]} because it lost {categorical_loss} categorical matches versus current_baseline."
        )
    print()
    print(f"Full residual table for {winner} (signed delta = predicted - expected)")
    print()
    print("| request_id | requested_amount | currency | expected_safe | predicted_safe | signed_delta | delta_%_requested | our_status | expected_status |")
    print("|---|---:|:---:|---:|---:|---:|---:|---|---|")
    ordered = sorted(results[winner][3], key=lambda item: (-abs(item[4] - item[3]), item[0]["request_id"]))
    for sample, predicted, profile, expected, actual in ordered:
        requested = Decimal(sample["requested_amount"])
        delta = actual - expected
        delta_pct = delta / requested * Decimal("100")
        print(
            f"| {sample['request_id']} | {format_decimal(requested)} | {profile['home_currency']} | "
            f"{format_decimal(expected)} | {format_decimal(actual)} | {delta:+f} | {delta_pct:+.4f}% | "
            f"{predicted['affordability_status']} | {sample['affordability_status']} |"
        )
    return 0


def evaluate_variable_reserve(repository, samples, estimator: str, days: str):
    agent = AffordabilityAgent(
        repository,
        reserve_policy="current_baseline",
        variable_estimator=estimator,
        variable_days=days,
    )
    metrics = Counter(rows=len(samples))
    total_absolute = ZERO
    for sample in samples:
        decision, _ = agent.decide(sample)
        predicted = decision.as_row()
        expected = Decimal(sample["amount_safe_to_pay"])
        actual = Decimal(predicted["amount_safe_to_pay"])
        checks = {
            "amount": abs(actual - expected) <= Decimal("0.01"),
            "status": predicted["affordability_status"] == sample["affordability_status"],
            "method": predicted["recommended_payment_method"] == sample["recommended_payment_method"],
            "plan": predicted["payment_plan"] == sample["payment_plan"],
            "earliest": predicted["earliest_date_for_full_payment"] == sample["earliest_date_for_full_payment"],
            "changes": predicted["spending_changes_needed"] == sample["spending_changes_needed"],
        }
        for field, passed in checks.items():
            metrics[field] += int(passed)
        total_absolute += abs(actual - expected)
    metrics["categorical"] = sum(metrics[field] for field in ("status", "method", "plan", "earliest", "changes"))
    return metrics, total_absolute


def compare_variable_reserves(repository, samples) -> int:
    combinations = [("none", "none")] + [
        (estimator, days)
        for estimator in ("median", "p75")
        for days in ("next_income", "7", "14", "30")
    ]
    results = {
        combination: evaluate_variable_reserve(repository, samples, *combination)
        for combination in combinations
    }
    baseline = results[("none", "none")][0]
    ranked = sorted(
        combinations,
        key=lambda combination: (
            -results[combination][0]["amount"],
            -results[combination][0]["categorical"],
            results[combination][1],
            combinations.index(combination),
        ),
    )
    winner = ranked[0]
    loss = baseline["categorical"] - results[winner][0]["categorical"]
    if loss > 1:
        winner = ("none", "none")
    print("Essential-variable reserve grid")
    print(f"Derived variable event threshold: {AffordabilityAgent(repository).forecast.variable_event_threshold} events")
    print()
    print("| estimator | horizon | total_abs_residual | amount | status | method | plan | earliest | changes | categorical_total |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for estimator, days in combinations:
        metrics, total_absolute = results[(estimator, days)]
        print(
            f"| {estimator} | {days} | {format_decimal(total_absolute)} | {metrics['amount']}/25 | "
            f"{metrics['status']}/25 | {metrics['method']}/25 | {metrics['plan']}/25 | "
            f"{metrics['earliest']}/25 | {metrics['changes']}/25 | {metrics['categorical']}/125 |"
        )
    print()
    print(f"Selected combination: estimator={winner[0]}, horizon={winner[1]}")
    if ranked[0] != winner:
        print(
            f"Rejected estimator={ranked[0][0]}, horizon={ranked[0][1]} because it lost {loss} categorical matches versus baseline."
        )
    return 0


def main() -> int:
    args = parse_args()
    repository = DataRepository(REPO_ROOT / "dataset")
    agent = AffordabilityAgent(repository)
    with (REPO_ROOT / "dataset" / "sample_requests.csv").open(newline="", encoding="utf-8-sig") as handle:
        samples = list(csv.DictReader(handle))
    if args.compare_reserve_policies:
        return compare_reserve_policies(repository, samples)
    if args.compare_variable_reserves:
        return compare_variable_reserves(repository, samples)
    metrics = Counter()
    mismatches = []
    residuals: list[Residual] = []
    for sample in samples:
        profile = repository.profile_by_user[sample["user_id"]]
        request_date = date.fromisoformat(sample["request_date"])
        facts = agent.evidence.interpret(
            repository.messages_by_user.get(sample["user_id"], []),
            repository.events_by_user.get(sample["user_id"], []),
            request_date,
        )
        facts = agent.semantic.augment(
            facts,
            repository.messages_by_user.get(sample["user_id"], []),
            repository.events_by_user.get(sample["user_id"], []),
            request_date,
        )
        baseline = agent.forecast.build_baseline(sample, profile, facts)
        decision = agent.planner.decide(sample, profile, baseline)
        agent.validator.validate(decision, sample, profile)
        predicted = decision.as_row()
        expected_amount = Decimal(sample["amount_safe_to_pay"])
        predicted_amount = Decimal(predicted["amount_safe_to_pay"])
        requested_amount = Decimal(sample["requested_amount"])
        bucket, detail = classify_residual(
            repository, agent, sample, profile, baseline, expected_amount, predicted_amount
        )
        pending_counterfactual, pending_ids = duplicate_pending_counterfactual(agent, sample, profile, baseline)
        quantum = rounding_match(predicted_amount, expected_amount)
        shorter_days, shorter_capacity = best_shorter_horizon(
            agent, sample, profile, baseline, expected_amount
        )
        fx_capacity, foreign_ids, fx_alternative_ids, fx_audit_ok = fx_counterfactual(
            repository, agent, sample, profile, baseline
        )
        residuals.append(
            Residual(
                request_id=sample["request_id"],
                requested=requested_amount,
                currency=profile["home_currency"],
                expected=expected_amount,
                predicted=predicted_amount,
                delta=predicted_amount - expected_amount,
                delta_pct=(predicted_amount - expected_amount) / requested_amount * Decimal("100"),
                predicted_status=predicted["affordability_status"],
                expected_status=sample["affordability_status"],
                binding_constraint=binding_constraint(agent, sample, profile, baseline, predicted_amount),
                bucket=bucket,
                bucket_detail=detail,
                pending_overlap_ids=tuple(pending_ids),
                pending_counterfactual=pending_counterfactual,
                rounding_quantum=quantum,
                shorter_horizon_days=shorter_days,
                shorter_horizon_capacity=shorter_capacity,
                foreign_event_ids=tuple(foreign_ids),
                fx_alternative_ids=tuple(fx_alternative_ids),
                fx_counterfactual=fx_capacity,
                fx_audit_ok=fx_audit_ok,
            )
        )
        metrics["rows"] += 1
        checks = {
            "amount": abs(Decimal(predicted["amount_safe_to_pay"]) - Decimal(sample["amount_safe_to_pay"])) <= Decimal("0.01"),
            "status": predicted["affordability_status"] == sample["affordability_status"],
            "method": predicted["recommended_payment_method"] == sample["recommended_payment_method"],
            "plan": predicted["payment_plan"] == sample["payment_plan"],
            "earliest": predicted["earliest_date_for_full_payment"] == sample["earliest_date_for_full_payment"],
            "changes": predicted["spending_changes_needed"] == sample["spending_changes_needed"],
        }
        for name, passed in checks.items():
            metrics[name] += int(passed)
        if not all(checks.values()):
            mismatches.append((sample["request_id"], {name: passed for name, passed in checks.items() if not passed}, predicted))
    if args.residuals:
        print_residuals(residuals)
    else:
        print("Sample evaluation")
        for name in ("amount", "status", "method", "plan", "earliest", "changes"):
            print(f"  {name:8}: {metrics[name]}/{metrics['rows']}")
        print("Mismatches")
        for request_id, failures, predicted in mismatches:
            print(request_id, ",".join(failures), predicted)
    report = [
        "# Public Sample Evaluation",
        "",
        "Generated by `python3 code/evaluation/main.py` against the 25 participant-facing solved examples.",
        "",
        "| Field | Exact matches |",
        "|---|---:|",
    ]
    for name in ("amount", "status", "method", "plan", "earliest", "changes"):
        report.append(f"| {name} | {metrics[name]}/{metrics['rows']} |")
    report.extend(
        [
            "",
            "The sample file is used only as a regression suite. No sample request IDs or output labels are imported by the prediction pipeline.",
            "",
        ]
    )
    (CODE_ROOT / "evaluation" / "sample_report.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
