from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from .data import DataRepository, DataError
from .models import Decision


OUTPUT_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


class DecisionValidator:
    """Independent output-schema and cross-field guardrail."""

    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository

    def validate(self, decision: Decision, request: dict[str, str], profile: dict[str, str]) -> None:
        requested = Decimal(request["requested_amount"])
        request_date = date.fromisoformat(request["request_date"])
        deadline = date.fromisoformat(request["desired_completion_date"])
        accepted = set(filter(None, profile["payment_methods_user_will_consider"].split("|")))
        if not Decimal("0") <= decision.amount_safe_to_pay <= requested:
            raise DataError(f"amount_safe_to_pay out of bounds for {request['request_id']}")
        if decision.affordability_status not in STATUSES:
            raise DataError(f"invalid status for {request['request_id']}")
        if decision.recommended_payment_method not in METHODS:
            raise DataError(f"invalid method for {request['request_id']}")
        if not decision.decision_explanation.strip():
            raise DataError(f"empty explanation for {request['request_id']}")
        payments = self._parse_plan(decision.payment_plan)
        if payments != sorted(payments):
            raise DataError(f"non-chronological plan for {request['request_id']}")
        if payments and payments[0][0] < request_date:
            raise DataError(f"plan starts before request date for {request['request_id']}")
        if payments and payments[-1][0] > deadline:
            raise DataError(f"plan misses deadline for {request['request_id']}")
        method = decision.recommended_payment_method
        expected_statuses = {
            "full_payment": {"affordable_now", "affordable_with_plan"},
            "partial_payment": {"affordable_with_plan"},
            "installments": {"affordable_with_plan"},
            "wait": {"affordable_later"},
            "not_recommended": {"not_affordable"},
        }
        if decision.affordability_status not in expected_statuses[method]:
            raise DataError(f"status and method disagree for {request['request_id']}")
        required_preference = "full_payment" if method == "wait" else method
        if method != "not_recommended" and required_preference not in accepted:
            raise DataError(f"method conflicts with user preference for {request['request_id']}")
        if method in {"full_payment", "wait"}:
            if len(payments) != 1 or payments[0][1] != requested:
                raise DataError(f"invalid one-payment plan for {request['request_id']}")
        if method == "partial_payment":
            if request["allows_partial_payment"].casefold() != "true":
                raise DataError(f"partial payment not allowed for {request['request_id']}")
            if (
                len(payments) != 2
                or payments[0][0] != request_date
                or payments[0][1] != decision.amount_safe_to_pay
                or sum(amount for _, amount in payments) != requested
            ):
                raise DataError(f"invalid partial plan for {request['request_id']}")
        if method == "installments" and not self._matches_option(request["request_id"], payments):
            raise DataError(f"installment plan does not match an offer for {request['request_id']}")
        if method == "installments" and profile["max_installment_months"]:
            if len(payments) > int(profile["max_installment_months"]):
                raise DataError(f"installment plan exceeds user limit for {request['request_id']}")
        if method == "not_recommended" and decision.payment_plan != "none":
            raise DataError(f"fallback includes a payment for {request['request_id']}")
        if decision.affordability_status == "affordable_now" and decision.earliest_date_for_full_payment != request["request_date"]:
            raise DataError(f"affordable_now has wrong earliest date for {request['request_id']}")
        self._validate_changes(decision.spending_changes_needed, request["user_id"], profile)

    @staticmethod
    def _parse_plan(text: str) -> list[tuple[date, Decimal]]:
        if text == "none":
            return []
        result = []
        for entry in text.split("|"):
            day, amount = entry.split(":", 1)
            result.append((date.fromisoformat(day), Decimal(amount)))
        return result

    def _matches_option(self, request_id: str, payments: list[tuple[date, Decimal]]) -> bool:
        for option in self.repository.options_by_request.get(request_id, []):
            if option["payment_method"] != "installments" or int(option["number_of_payments"]) != len(payments):
                continue
            first = date.fromisoformat(option["first_payment_date"])
            frequency = int(option["payment_frequency_days"])
            amount = Decimal(option["payment_amount"])
            expected = [(first + timedelta(days=frequency * index), amount) for index in range(len(payments))]
            if payments == expected:
                return True
        return False

    def _validate_changes(self, text: str, user_id: str, profile: dict[str, str]) -> None:
        if text == "none":
            return
        entries = text.split("|")
        if len(entries) > 3:
            raise DataError(f"more than three spending changes for {user_id}")
        event_by_id = {event["event_id"]: event for event in self.repository.events_by_user.get(user_id, [])}
        reducible = set(filter(None, profile["expense_categories_user_is_willing_to_reduce"].split("|")))
        stoppable = set(filter(None, profile["expense_categories_user_is_willing_to_stop"].split("|")))
        used: set[str] = set()
        for entry in entries:
            parts = entry.split(":")
            if len(parts) not in {2, 3} or parts[0] not in {"stop", "reduce_to"}:
                raise DataError(f"bad spending change: {entry}")
            event = event_by_id.get(parts[1])
            if not event or event["flexibility"] == "fixed" or event["event_id"] in used:
                raise DataError(f"ineligible spending change: {entry}")
            if parts[0] == "stop" and (
                event["category"] not in stoppable or event["flexibility"] not in {"stoppable", "reducible_or_stoppable"}
            ):
                raise DataError(f"stop action conflicts with preferences: {entry}")
            if parts[0] == "reduce_to":
                if (
                    len(parts) != 3
                    or event["category"] not in reducible
                    or event["flexibility"] not in {"reducible", "reducible_or_stoppable"}
                    or not event["minimum_allowed_amount"]
                    or Decimal(parts[2]) != Decimal(event["minimum_allowed_amount"])
                ):
                    raise DataError(f"reduction action conflicts with preferences: {entry}")
            used.add(event["event_id"])
