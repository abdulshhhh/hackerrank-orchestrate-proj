from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal


ZERO = Decimal("0")


@dataclass(frozen=True)
class CashFlow:
    date: date
    amount: Decimal
    source: str
    category: str
    event_id: str = ""
    recurrence_id: str = ""
    flexibility: str = "fixed"
    minimum_amount: Decimal | None = None


@dataclass(frozen=True)
class SpendingChange:
    action: Literal["stop", "reduce_to"]
    recurrence_id: str
    event_id: str
    new_amount: Decimal | None = None

    def render(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        assert self.new_amount is not None
        value = self.new_amount.quantize(Decimal("0.01"))
        rendered = format(value.quantize(Decimal("1")), "f") if value == value.to_integral_value() else format(value, ".2f")
        return f"reduce_to:{self.event_id}:{rendered}"


@dataclass(frozen=True)
class CandidatePlan:
    method: Literal["full_payment", "partial_payment", "installments", "wait"]
    payments: tuple[tuple[date, Decimal], ...]
    total_payable: Decimal
    changes: tuple[SpendingChange, ...] = ()
    option_id: str = ""


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    minimum_projected_balance: Decimal
    minimum_date: date


@dataclass(frozen=True)
class Decision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str

    def as_row(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": format_money(self.amount_safe_to_pay),
            "affordability_status": self.affordability_status,
            "recommended_payment_method": self.recommended_payment_method,
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": self.earliest_date_for_full_payment,
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }


@dataclass
class EvidenceFacts:
    suppress_recurring_income: bool = False
    recurring_salary_amount: Decimal | None = None
    recurring_salary_currency: str | None = None
    recurring_salary_start: date | None = None
    recurring_salary_day: int | None = None
    next_salary_amount: Decimal | None = None
    next_salary_currency: str | None = None
    next_salary_date: date | None = None
    one_time_credits: list[tuple[date, Decimal, str, str]] = field(default_factory=list)
    rent_multiplier: Decimal = Decimal("1")
    include_failed_event_ids: set[str] = field(default_factory=set)
    add_childcare_recurrence: bool = False
    ignored_evidence: list[str] = field(default_factory=list)


def format_money(value: Decimal) -> str:
    value = value.quantize(Decimal("0.01"))
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_display_money(value: Decimal) -> str:
    value = value.quantize(Decimal("0.01"))
    decimals = 0 if value == value.to_integral_value() else 2
    return f"{value:,.{decimals}f}"


def format_plan(payments: tuple[tuple[date, Decimal], ...]) -> str:
    if not payments:
        return "none"
    def plan_amount(amount: Decimal) -> str:
        value = amount.quantize(Decimal("0.01"))
        return format(value.quantize(Decimal("1")), "f") if value == value.to_integral_value() else format(value, ".2f")

    return "|".join(f"{day.isoformat()}:{plan_amount(amount)}" for day, amount in payments)
