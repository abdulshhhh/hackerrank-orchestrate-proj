from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from itertools import product

from .data import DataRepository
from .forecast import ForecastEngine
from .models import CandidatePlan, Decision, SpendingChange, ZERO, format_display_money, format_plan


class PlanGenerator:
    """Enumerate allowed plans, prove their safety, then apply rubric ordering."""

    def __init__(self, repository: DataRepository, forecast: ForecastEngine) -> None:
        self.repository = repository
        self.forecast = forecast
        self.event_by_id = {event["event_id"]: event for event in repository.events}

    def decide(self, request, profile, baseline) -> Decision:
        request_date = date.fromisoformat(request["request_date"])
        desired_date = date.fromisoformat(request["desired_completion_date"])
        requested = Decimal(request["requested_amount"])
        opening = Decimal(profile["current_available_balance"])
        reserve = Decimal(profile["minimum_balance_to_keep"])
        currency = profile["home_currency"]
        accepted = set(filter(None, profile["payment_methods_user_will_consider"].split("|")))
        capacity = min(requested, self.forecast.capacity_today(request_date, opening, reserve, baseline, profile))
        earliest = self._earliest_full_date(request_date, requested, opening, reserve, baseline)
        change_sets = self._change_sets(self.forecast.eligible_changes(baseline))
        candidates: list[CandidatePlan] = []

        if "full_payment" in accepted:
            for changes in change_sets:
                plan = CandidatePlan("full_payment", ((request_date, requested),), requested, changes)
                if self._safe(plan, opening, reserve, baseline):
                    candidates.append(plan)

        if (
            request["allows_partial_payment"].casefold() == "true"
            and "partial_payment" in accepted
            and ZERO < capacity < requested
            and earliest is not None
            and request_date < earliest
            and earliest <= desired_date
        ):
            for changes in change_sets:
                plan = CandidatePlan(
                    "partial_payment",
                    ((request_date, capacity), (earliest, requested - capacity)),
                    requested,
                    changes,
                )
                if self._safe(plan, opening, reserve, baseline):
                    candidates.append(plan)

        if "installments" in accepted:
            maximum = int(profile["max_installment_months"]) if profile["max_installment_months"] else 0
            for option in self.repository.options_by_request.get(request["request_id"], []):
                if option["payment_method"] != "installments":
                    continue
                count = int(option["number_of_payments"])
                if not maximum or count > maximum:
                    continue
                first = date.fromisoformat(option["first_payment_date"])
                frequency = int(option["payment_frequency_days"])
                payments = tuple((first + timedelta(days=frequency * index), Decimal(option["payment_amount"])) for index in range(count))
                if not payments or payments[-1][0] > desired_date:
                    continue
                for changes in change_sets:
                    plan = CandidatePlan("installments", payments, Decimal(option["total_payable_amount"]), changes, option["payment_option_id"])
                    if self._safe(plan, opening, reserve, baseline):
                        candidates.append(plan)

        if "full_payment" in accepted and earliest is not None and request_date < earliest <= desired_date:
            plan = CandidatePlan("wait", ((earliest, requested),), requested)
            if self._safe(plan, opening, reserve, baseline):
                candidates.append(plan)

        if not candidates:
            return self._not_affordable(request, currency, requested, capacity, earliest)

        winner = min(candidates, key=self._rank)
        status = "affordable_now" if winner.method == "full_payment" and not winner.changes and capacity >= requested else "affordable_with_plan"
        if winner.method == "wait":
            status = "affordable_later"
        safety = self.forecast.safety(opening, reserve, baseline, winner.payments, winner.changes, request_date)
        return Decision(
            request_id=request["request_id"],
            amount_safe_to_pay=capacity,
            affordability_status=status,
            recommended_payment_method=winner.method,
            payment_plan=format_plan(winner.payments),
            earliest_date_for_full_payment=earliest.isoformat() if earliest else "",
            spending_changes_needed="|".join(change.render() for change in winner.changes) or "none",
            decision_explanation=self._explain(winner, currency, requested, reserve, capacity, safety.minimum_projected_balance),
        )

    def _earliest_full_date(self, request_date, requested, opening, reserve, baseline):
        for offset in range(self.forecast.horizon_days + 1):
            candidate = request_date + timedelta(days=offset)
            if self.forecast.safety(opening, reserve, baseline, ((candidate, requested),), start_date=request_date).safe:
                return candidate
        return None

    def _safe(self, plan, opening, reserve, baseline):
        return self.forecast.safety(opening, reserve, baseline, plan.payments, plan.changes).safe

    @staticmethod
    def _change_sets(changes: tuple[SpendingChange, ...]) -> tuple[tuple[SpendingChange, ...], ...]:
        by_recurrence: dict[str, list[SpendingChange]] = defaultdict(list)
        for change in changes:
            by_recurrence[change.recurrence_id].append(change)
        choices = [[None, *items] for _, items in sorted(by_recurrence.items())]
        combinations = [
            tuple(item for item in selection if item is not None)
            for selection in product(*choices)
            if sum(item is not None for item in selection) <= 3
        ] if choices else [()]
        return tuple(sorted(set(combinations), key=lambda items: (len(items), tuple(item.render() for item in items))))

    @staticmethod
    def _rank(plan: CandidatePlan):
        return (
            bool(plan.changes),
            plan.total_payable,
            plan.payments[0][0],
            len(plan.payments),
            len(plan.changes),
            plan.option_id,
        )

    def _explain(self, plan, currency, requested, reserve, capacity, projected_floor):
        reserve_text = f"{currency} {format_display_money(reserve)}"
        if plan.changes:
            phrases = []
            for change in plan.changes:
                description = self.event_by_id.get(change.event_id, {}).get("description", change.event_id).lower()
                if change.action == "stop":
                    phrases.append(f"stop the {description}")
                else:
                    phrases.append(f"reduce the {description} to {currency} {format_display_money(change.new_amount)}")
            if plan.method == "full_payment":
                action = f"pay {currency} {format_display_money(requested)} today"
            elif plan.method == "partial_payment":
                first, second = plan.payments
                action = (
                    f"pay {currency} {format_display_money(first[1])} today and the remaining {currency} "
                    f"{format_display_money(second[1])} on {second[0].strftime('%-d %B %Y')}"
                )
            elif plan.method == "installments":
                action = (
                    f"use {len(plan.payments)} installments of {currency} {format_display_money(plan.payments[0][1])}, "
                    f"starting {plan.payments[0][0].strftime('%-d %B %Y')}"
                )
            else:
                action = f"wait and pay {currency} {format_display_money(requested)} on {plan.payments[0][0].strftime('%-d %B %Y')}"
            prefix = ", then ".join([" and ".join(phrases), action])
            return f"{prefix[:1].upper() + prefix[1:]}. This keeps the {reserve_text} minimum protected."
        if plan.method == "full_payment":
            return f"Pay {currency} {format_display_money(requested)} today. This leaves at least {reserve_text} available over the next 90 days."
        if plan.method == "partial_payment":
            first, second = plan.payments
            return (
                f"Pay {currency} {format_display_money(first[1])} today and the remaining {currency} {format_display_money(second[1])} "
                f"on {second[0].strftime('%-d %B %Y')}. This completes the request and protects the {reserve_text} minimum."
            )
        if plan.method == "installments":
            return (
                f"Use {len(plan.payments)} installments of {currency} {format_display_money(plan.payments[0][1])}, starting "
                f"{plan.payments[0][0].strftime('%-d %B %Y')}. This leaves at least {reserve_text} available."
            )
        payment_date = plan.payments[0][0]
        return (
            f"Pay {currency} {format_display_money(requested)} in full on {payment_date.strftime('%-d %B %Y')}. "
            f"Paying earlier would put the {reserve_text} minimum at risk."
        )

    def conservative_fallback(self, request, profile, decision, reason_code: str):
        """Replace a decision whose flaw an independent re-check confirmed.

        Reached only when the adjudicator flagged a problem AND a deterministic re-check
        agreed, so the recorded plan is known to violate a rule. The safe answer is to
        recommend nothing rather than to guess a repair: the capacity figure is preserved
        because it was computed before the flawed plan was assembled.
        """
        currency = profile["home_currency"]
        return replace(
            decision,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=(
                f"Do not make this payment by {request['desired_completion_date']}. "
                f"The reviewed plan was withdrawn after an automated check confirmed "
                f"{reason_code.replace('_', ' ')}, and no remaining option keeps the "
                f"{currency} {profile['minimum_balance_to_keep']} minimum protected."
            ),
        )

    @staticmethod
    def _not_affordable(request, currency, requested, capacity, earliest):
        deadline = date.fromisoformat(request["desired_completion_date"])
        if capacity > ZERO:
            explanation = (
                f"Do not make this payment by {deadline.strftime('%-d %B %Y')}. Only {currency} "
                f"{format_display_money(capacity)} is safe today, so no available option completes the request in time."
            )
        else:
            explanation = f"Do not make this payment by {deadline.strftime('%-d %B %Y')}. None of the available options keeps the minimum balance protected."
        return Decision(
            request_id=request["request_id"],
            amount_safe_to_pay=capacity,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=earliest.isoformat() if earliest else "",
            spending_changes_needed="none",
            decision_explanation=explanation,
        )
