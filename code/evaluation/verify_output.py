#!/usr/bin/env python3
"""Verify output.csv against the submission contract in section 6.2.

This re-derives every rule from requests.csv and the profiles rather than trusting the
pipeline that produced the file, so it catches a regression the planner's own validator
would not: the planner checks a Decision object in memory, this checks the bytes actually
written to disk.

    python3 code/evaluation/verify_output.py
    python3 code/evaluation/verify_output.py --output output.csv --dataset dataset

Exits non-zero and prints every violation when the file does not satisfy the contract.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT if (CODE_ROOT / "dataset" / "requests.csv").is_file() else CODE_ROOT.parent

COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
MAX_EXPLANATION = 300
MAX_SPENDING_CHANGES = 3


def parse_plan(text: str) -> list[tuple[date, Decimal]]:
    return [
        (date.fromisoformat(entry.split(":")[0]), Decimal(entry.split(":")[1]))
        for entry in text.split("|")
    ]


def verify(output_path: Path, dataset_dir: Path) -> list[str]:
    problems: list[str] = []
    with (dataset_dir / "requests.csv").open(newline="", encoding="utf-8-sig") as handle:
        requests = list(csv.DictReader(handle))
    with output_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    if header != COLUMNS:
        problems.append(f"column order is {header}, expected {COLUMNS}")
        return problems

    expected_ids = [row["request_id"] for row in requests]
    actual_ids = [row["request_id"] for row in rows]
    if len(rows) != len(requests):
        problems.append(f"row count {len(rows)}, expected {len(requests)}")
    if len(set(actual_ids)) != len(actual_ids):
        problems.append("duplicate request_id values")
    missing = set(expected_ids) - set(actual_ids)
    if missing:
        problems.append(f"missing request_ids: {sorted(missing)[:5]}")
    if actual_ids != expected_ids:
        problems.append("row order does not match requests.csv")

    by_id = {row["request_id"]: row for row in requests}
    for row in rows:
        rid = row["request_id"]
        source = by_id.get(rid)
        if source is None:
            problems.append(f"{rid}: not present in requests.csv")
            continue
        requested = Decimal(source["requested_amount"])
        request_date = source["request_date"]
        deadline = date.fromisoformat(source["desired_completion_date"])

        try:
            safe = Decimal(row["amount_safe_to_pay"])
        except InvalidOperation:
            problems.append(f"{rid}: amount_safe_to_pay is not a number")
            continue
        if not Decimal("0") <= safe <= requested:
            problems.append(f"{rid}: amount_safe_to_pay {safe} outside [0, {requested}]")

        status, method = row["affordability_status"], row["recommended_payment_method"]
        if status not in STATUSES:
            problems.append(f"{rid}: unknown affordability_status {status!r}")
        if method not in METHODS:
            problems.append(f"{rid}: unknown recommended_payment_method {method!r}")

        earliest = row["earliest_date_for_full_payment"]
        if status == "affordable_now" and earliest != request_date:
            problems.append(f"{rid}: affordable_now earliest {earliest!r} != request_date {request_date}")
        if earliest:
            try:
                date.fromisoformat(earliest)
            except ValueError:
                problems.append(f"{rid}: earliest_date_for_full_payment {earliest!r} is not ISO")

        plan = row["payment_plan"]
        if plan != "none":
            try:
                payments = parse_plan(plan)
            except (ValueError, IndexError, InvalidOperation):
                problems.append(f"{rid}: payment_plan is unparseable: {plan!r}")
                payments = []
            if payments:
                if payments != sorted(payments, key=lambda item: item[0]):
                    problems.append(f"{rid}: payment_plan is not chronological")
                if any(amount <= Decimal("0") for _, amount in payments):
                    problems.append(f"{rid}: payment_plan contains a non-positive amount")
                if method == "partial_payment":
                    if len(payments) != 2:
                        problems.append(f"{rid}: partial_payment needs exactly two payments, found {len(payments)}")
                    else:
                        total = payments[0][1] + payments[1][1]
                        if total != requested:
                            problems.append(f"{rid}: partial payments sum to {total}, expected {requested}")
                        if payments[0][0] >= payments[1][0]:
                            problems.append(f"{rid}: partial payments are not on distinct ascending dates")
                        if payments[0][1] != safe:
                            problems.append(f"{rid}: first partial payment {payments[0][1]} != amount_safe_to_pay {safe}")
                        if payments[1][0] > deadline:
                            problems.append(f"{rid}: second partial payment after desired_completion_date")
        elif method in {"full_payment", "partial_payment", "installments"}:
            problems.append(f"{rid}: method {method} requires a payment_plan")

        changes = row["spending_changes_needed"]
        if changes != "none":
            actions = changes.split("|")
            if len(actions) > MAX_SPENDING_CHANGES:
                problems.append(f"{rid}: {len(actions)} spending changes, maximum is {MAX_SPENDING_CHANGES}")
            for action in actions:
                if not (action.startswith("stop:") or action.startswith("reduce_to:")):
                    problems.append(f"{rid}: unrecognised spending change {action!r}")

        explanation = row["decision_explanation"]
        if not explanation.strip():
            problems.append(f"{rid}: decision_explanation is empty")
        if len(explanation) >= MAX_EXPLANATION:
            problems.append(f"{rid}: decision_explanation is {len(explanation)} characters")
        if any(character in explanation for character in ("\n", "\r", '"', "|")):
            problems.append(f"{rid}: decision_explanation contains a CSV-unsafe character")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify output.csv against the submission contract")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output.csv")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    args = parser.parse_args()
    problems = verify(args.output.resolve(), args.dataset.resolve())
    if problems:
        print(f"FAIL: {len(problems)} contract violation(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"PASS: {args.output.resolve()} satisfies the section 6.2 output contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
