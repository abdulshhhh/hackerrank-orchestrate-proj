from __future__ import annotations

import re
import os
import statistics
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_CEILING

from .data import DataRepository
from .models import CashFlow, EvidenceFacts, SafetyResult, SpendingChange, ZERO


HIGH_FREQUENCY_CATEGORIES = {"groceries", "transport", "dining"}
ONE_TIME_INCOME = re.compile(
    r"bonus|commission|arrears|reimbursement|refund|prize|project|invoice|investment sale|"
    r"final employer|contract payment|assignment pay|platform payout|app earnings|retainer",
    re.IGNORECASE,
)

RESERVE_POLICIES = (
    "current_baseline",
    "buffer_to_income",
    "full_cycle",
    "conservative_band",
)


class ReservePolicy:
    """Pluggable definition of cash that must remain untouched on day zero."""

    name = ""

    def capacity(self, engine, request_date, opening_balance, minimum_balance, baseline, profile):
        raise NotImplementedError


class CurrentBaselineReservePolicy(ReservePolicy):
    name = "current_baseline"

    def capacity(self, engine, request_date, opening_balance, minimum_balance, baseline, profile):
        base = engine.safety(opening_balance, minimum_balance, baseline, start_date=request_date)
        return max(ZERO, base.minimum_projected_balance - minimum_balance)


class BufferToIncomeReservePolicy(ReservePolicy):
    name = "buffer_to_income"

    def capacity(self, engine, request_date, opening_balance, minimum_balance, baseline, profile):
        protected = set(filter(None, profile["expense_categories_to_protect"].split("|")))
        incomes = sorted(
            flow.date for flow in baseline
            if flow.amount > ZERO and flow.category == "salary" and flow.date >= request_date
        )
        cutoff = incomes[0] if incomes else request_date + timedelta(days=engine.horizon_days)
        reserved = sum(
            -flow.amount for flow in baseline
            if request_date <= flow.date <= cutoff
            and flow.amount < ZERO
            and (flow.category in protected or flow.source in {"pending", "scheduled", "failed_retry"})
        )
        return max(ZERO, opening_balance - minimum_balance - reserved)


class FullCycleReservePolicy(ReservePolicy):
    name = "full_cycle"

    def capacity(self, engine, request_date, opening_balance, minimum_balance, baseline, profile):
        protected = set(filter(None, profile["expense_categories_to_protect"].split("|")))
        recurring: dict[str, CashFlow] = {}
        explicit: list[CashFlow] = []
        for flow in baseline:
            if flow.amount >= ZERO:
                continue
            if flow.source == "recurrence" and flow.category in protected:
                recurring.setdefault(flow.recurrence_id, flow)
            elif flow.source in {"pending", "scheduled", "failed_retry"} or flow.category in protected:
                explicit.append(flow)
        reserved = sum(-flow.amount for flow in recurring.values()) + sum(-flow.amount for flow in explicit)
        return max(ZERO, opening_balance - minimum_balance - reserved)


class ConservativeBandReservePolicy(BufferToIncomeReservePolicy):
    name = "conservative_band"

    def capacity(self, engine, request_date, opening_balance, minimum_balance, baseline, profile):
        protected = set(filter(None, profile["expense_categories_to_protect"].split("|")))
        incomes = sorted(
            flow.date for flow in baseline
            if flow.amount > ZERO and flow.category == "salary" and flow.date >= request_date
        )
        cutoff = incomes[0] if incomes else request_date + timedelta(days=engine.horizon_days)
        recurring = [
            flow for flow in baseline
            if request_date <= flow.date <= cutoff and flow.amount < ZERO
            and flow.source == "recurrence" and flow.category in protected
        ]
        event_by_id = {event["event_id"]: event for event in engine.repository.events}
        user_id = next((event_by_id[flow.event_id]["user_id"] for flow in recurring if flow.event_id in event_by_id), "")
        home = profile["home_currency"]
        p75_by_category: dict[str, Decimal] = {}
        if user_id:
            for category in {flow.category for flow in recurring}:
                values = sorted(
                    engine.repository.amount_in_home(event, home)
                    for event in engine.repository.events_by_user.get(user_id, [])
                    if event["direction"] == "debit"
                    and event["status"] == "settled"
                    and event["category"] == category
                    and event["settlement_date"]
                    and 0 <= (request_date - date.fromisoformat(event["settlement_date"])).days <= 150
                )
                if values:
                    p75_by_category[category] = values[max(0, (3 * len(values) + 3) // 4 - 1)]
        reserved = ZERO
        for flow in baseline:
            if not (request_date <= flow.date <= cutoff and flow.amount < ZERO):
                continue
            if flow.source == "recurrence" and flow.category in protected:
                reserved += p75_by_category.get(flow.category, -flow.amount)
            elif flow.source in {"pending", "scheduled", "failed_retry"}:
                reserved += -flow.amount
        return max(ZERO, opening_balance - minimum_balance - reserved)


POLICY_TYPES = {
    policy.name: policy
    for policy in (
        CurrentBaselineReservePolicy(),
        BufferToIncomeReservePolicy(),
        FullCycleReservePolicy(),
        ConservativeBandReservePolicy(),
    )
}


class ForecastEngine:
    """Reconstruct a conservative daily cash ledger for the next 90 days."""

    def __init__(
        self,
        repository: DataRepository,
        horizon_days: int = 90,
        reserve_policy: str | None = None,
        variable_estimator: str | None = None,
        variable_days: str | None = None,
    ) -> None:
        self.repository = repository
        self.horizon_days = horizon_days
        self.reserve_policy_name = reserve_policy or os.getenv("BUYWAIT_RESERVE_POLICY", "current_baseline")
        if self.reserve_policy_name not in POLICY_TYPES:
            raise ValueError(
                f"unknown BUYWAIT_RESERVE_POLICY={self.reserve_policy_name!r}; choose one of {', '.join(RESERVE_POLICIES)}"
            )
        self.reserve_policy = POLICY_TYPES[self.reserve_policy_name]
        self.variable_estimator = variable_estimator or os.getenv("BUYWAIT_VARIABLE_ESTIMATOR", "p75")
        self.variable_days = variable_days or os.getenv("BUYWAIT_VARIABLE_DAYS", "7")
        if self.variable_estimator not in {"none", "median", "p75"}:
            raise ValueError("BUYWAIT_VARIABLE_ESTIMATOR must be none, median, or p75")
        if self.variable_days not in {"none", "next_income", "7", "14", "30"}:
            raise ValueError("BUYWAIT_VARIABLE_DAYS must be none, next_income, 7, 14, or 30")
        self.variable_event_threshold = self._derive_variable_event_threshold()

    def build_baseline(self, request: dict[str, str], profile: dict[str, str], facts: EvidenceFacts) -> tuple[CashFlow, ...]:
        user_id = request["user_id"]
        request_date = date.fromisoformat(request["request_date"])
        horizon_end = request_date + timedelta(days=self.horizon_days)
        home = profile["home_currency"]
        events = self.repository.events_by_user.get(user_id, [])
        history = [
            event for event in events
            if event["status"] == "settled"
            and event["settlement_date"]
            and date.fromisoformat(event["settlement_date"]) <= request_date
            and 0 <= (request_date - date.fromisoformat(event["settlement_date"])).days <= 150
            and event["direction"] != "non_cash"
        ]
        flows = self._recurring_debits(history, request_date, horizon_end, home, profile, facts)
        flows = self._suppress_resolved_failed_cycles(flows, events, facts, request_date)
        flows.extend(self._recurring_income(history, request_date, horizon_end, home, facts))
        flows.extend(self._explicit_events(events, request_date, horizon_end, home, facts))
        flows.extend(self._evidence_credits(facts, request_date, horizon_end, home))
        flows = self._deduplicate_salary(flows)
        flows = self._normalize_first_income_cycle(flows, request_date)
        return tuple(sorted(flows, key=lambda flow: (flow.date, flow.source, flow.event_id)))

    def _recurring_debits(self, history, request_date, horizon_end, home, profile, facts) -> list[CashFlow]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for event in history:
            if event["direction"] != "debit":
                continue
            if event["category"] in HIGH_FREQUENCY_CATEGORIES:
                key = f"category:{event['category']}"
            elif event["category"] == "debt_repayment" and "separate card" in " ".join(
                row["message_text"].casefold() for row in self.repository.messages_by_user.get(event["user_id"], [])
            ):
                key = f"description:{event['description'].casefold()}"
            else:
                key = f"category:{event['category']}"
            grouped[key].append(event)

        reducible = set(filter(None, profile["expense_categories_user_is_willing_to_reduce"].split("|")))
        stoppable = set(filter(None, profile["expense_categories_user_is_willing_to_stop"].split("|")))
        flows: list[CashFlow] = []
        for recurrence_id, series in grouped.items():
            series.sort(key=lambda row: (row["settlement_date"], row["event_id"]))
            schedule = self._infer_schedule(series, request_date)
            if schedule is None:
                continue
            cadence, monthly, day_of_month = schedule
            recent = self._remove_outliers(series[-12:], home)
            estimate = self._conservative_amount(recent, home)
            category = series[-1]["category"]
            if category in HIGH_FREQUENCY_CATEGORIES:
                # Daily-life categories vary more than fixed bills.  Use a
                # conservative trimmed recent mean, while avoiding the hidden
                # assumption that every recent peak repeats forever.  Users
                # protecting both education and debt have less discretionary
                # room, so use the lower end of the observed run rate.
                protected = set(filter(None, profile["expense_categories_to_protect"].split("|")))
                use_lower_band = {"education", "debt_repayment"} <= protected or facts.suppress_recurring_income
                factor = Decimal("0.75") if facts.suppress_recurring_income else Decimal("0.80") if use_lower_band else Decimal("0.90")
                quantum = Decimal("50") if home == "IDR" else Decimal("1")
                estimate = (estimate * factor / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
            if category in {"rent", "housing"}:
                estimate *= facts.rent_multiplier
            representative = series[-1]
            flexibility = representative["flexibility"]
            minimum = Decimal(representative["minimum_allowed_amount"]) if representative["minimum_allowed_amount"] else None
            can_stop = category in stoppable and flexibility in {"stoppable", "reducible_or_stoppable"}
            can_reduce = category in reducible and flexibility in {"reducible", "reducible_or_stoppable"} and minimum is not None
            effective_flex = "reducible_or_stoppable" if can_stop and can_reduce else "stoppable" if can_stop else "reducible" if can_reduce else "fixed"
            next_date = self._next_occurrence(date.fromisoformat(series[-1]["settlement_date"]), cadence, monthly, day_of_month)
            while next_date < request_date:
                next_date = self._next_occurrence(next_date, cadence, monthly, day_of_month)
            while next_date <= horizon_end:
                flows.append(CashFlow(next_date, -estimate, "recurrence", category, representative["event_id"], recurrence_id, effective_flex, minimum))
                next_date = self._next_occurrence(next_date, cadence, monthly, day_of_month)
        if facts.add_childcare_recurrence and not any(flow.category == "family_support" for flow in flows):
            childcare = [event for event in history if event["category"] == "family_support" and event["direction"] == "debit"]
            if childcare:
                event = childcare[-1]
                amount = self.repository.amount_in_home(event, home)
                next_date = self._day_on_or_after(request_date, 15)
                while next_date <= horizon_end:
                    flows.append(CashFlow(next_date, -amount, "message_recurrence", "family_support", event["event_id"]))
                    next_date = self._next_month_day(next_date, 15)
        return flows

    def _recurring_income(self, history, request_date, horizon_end, home, facts) -> list[CashFlow]:
        if facts.suppress_recurring_income:
            return []
        settled_salary = sorted(
            (event for event in history if event["direction"] == "credit" and event["category"] == "salary"),
            key=lambda row: (row["settlement_date"], row["event_id"]),
        )
        if (
            facts.recurring_salary_amount is None
            and settled_salary
            and "final employer" in settled_salary[-1]["description"].casefold()
        ):
            return []
        flows: list[CashFlow] = []
        # A message may quote any currency. If no dated rate supports it, drop the fact and
        # fall through to the history-derived recurrence rather than halting the request.
        if facts.recurring_salary_amount is not None and facts.recurring_salary_currency:
            if self.repository.try_convert(
                facts.recurring_salary_amount, facts.recurring_salary_currency, home, request_date
            ) is None:
                facts.ignored_evidence.append(
                    {
                        "kind": "recurring_salary",
                        "reason": "no_dated_exchange_rate",
                        "currency": facts.recurring_salary_currency,
                    }
                )
                facts.recurring_salary_amount = None
                facts.recurring_salary_currency = None
        if facts.recurring_salary_amount is not None and facts.recurring_salary_currency:
            start = facts.recurring_salary_start or facts.next_salary_date or self._day_on_or_after(request_date, facts.recurring_salary_day or 15)
            while start < request_date:
                start = self._next_month_day(start, facts.recurring_salary_day or start.day)
            day_of_month = facts.recurring_salary_day or start.day
            while start <= horizon_end:
                amount = self.repository.convert(facts.recurring_salary_amount, facts.recurring_salary_currency, home, start)
                flows.append(CashFlow(start, amount, "message_salary", "salary", recurrence_id="salary:message"))
                start = self._next_month_day(start, day_of_month)
        else:
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for event in history:
                if event["direction"] != "credit" or event["category"] != "salary" or ONE_TIME_INCOME.search(event["description"]):
                    continue
                groups[event["description"].casefold()].append(event)
            for name, series in groups.items():
                series.sort(key=lambda row: row["settlement_date"])
                if len(series) < 2:
                    continue
                dates = [date.fromisoformat(row["settlement_date"]) for row in series]
                intervals = [(right - left).days for left, right in zip(dates, dates[1:])]
                if not intervals or not 27 <= statistics.median(intervals[-4:]) <= 32 or (request_date - dates[-1]).days > 45:
                    continue
                amount = self.repository.event_amount(series[-1])
                currency = series[-1]["currency"]
                day_of_month = facts.recurring_salary_day or Counter(day.day for day in dates[-4:]).most_common(1)[0][0]
                next_date = facts.next_salary_date or self._next_month_day(dates[-1], day_of_month)
                while next_date < request_date:
                    next_date = self._next_month_day(next_date, day_of_month)
                while next_date <= horizon_end:
                    converted = self.repository.convert(amount, currency, home, next_date)
                    flows.append(CashFlow(next_date, converted, "salary_recurrence", "salary", recurrence_id=f"salary:{name}"))
                    next_date = self._next_month_day(next_date, day_of_month)

            # Repeated freelance credits can establish a cadence, but use a
            # lower-quartile amount and only when no message says the payout is pending.
            irregular = [
                event for event in history
                if event["direction"] == "credit" and event["category"] == "salary" and ONE_TIME_INCOME.search(event["description"])
                and not re.search(r"bonus|commission|arrears|reimbursement|refund|prize|investment sale|final employer", event["description"], re.IGNORECASE)
            ]
            irregular.sort(key=lambda row: row["settlement_date"])
            if len(irregular) >= 4:
                dates = [date.fromisoformat(row["settlement_date"]) for row in irregular]
                cadence = round(statistics.median((right - left).days for left, right in zip(dates, dates[1:])))
                if 5 <= cadence <= 21 and (request_date - dates[-1]).days <= cadence * 2:
                    values = sorted(self.repository.amount_in_home(event, home) for event in irregular[-6:])
                    amount = values[max(0, (len(values) - 1) // 4)]
                    next_date = dates[-1] + timedelta(days=cadence)
                    while next_date < request_date:
                        next_date += timedelta(days=cadence)
                    while next_date <= horizon_end:
                        flows.append(CashFlow(next_date, amount, "conservative_income_recurrence", "salary", recurrence_id="salary:irregular"))
                        next_date += timedelta(days=cadence)

        if facts.next_salary_amount is not None and facts.next_salary_currency:
            next_date = facts.next_salary_date or self._day_on_or_after(request_date, 15)
            replacement = self.repository.convert(facts.next_salary_amount, facts.next_salary_currency, home, next_date)
            flows = [flow for flow in flows if flow.category != "salary"]
            recurring_date = next_date
            while recurring_date <= horizon_end:
                amount = self.repository.convert(facts.next_salary_amount, facts.next_salary_currency, home, recurring_date)
                flows.append(CashFlow(recurring_date, amount, "message_salary_once", "salary", recurrence_id="salary:next"))
                recurring_date = self._next_month_day(recurring_date, recurring_date.day)
        elif facts.next_salary_date and facts.recurring_salary_amount is None:
            salary_flows = sorted((flow for flow in flows if flow.category == "salary"), key=lambda flow: flow.date)
            if salary_flows:
                first = salary_flows[0]
                flows.remove(first)
                flows.append(CashFlow(facts.next_salary_date, first.amount, "message_salary_date", "salary", recurrence_id=first.recurrence_id))
        return flows

    def _explicit_events(self, events, request_date, horizon_end, home, facts) -> list[CashFlow]:
        flows: list[CashFlow] = []
        for event in events:
            if event["status"] == "failed" and event["event_id"] in facts.include_failed_event_ids:
                flows.append(CashFlow(request_date, -self.repository.amount_in_home(event, home), "failed_retry", event["category"], event["event_id"]))
                continue
            if not event["settlement_date"]:
                continue
            settlement = date.fromisoformat(event["settlement_date"])
            if not request_date <= settlement <= horizon_end:
                continue
            if event["direction"] == "debit" and event["status"] in {"pending", "scheduled"}:
                flows.append(CashFlow(settlement, -self.repository.amount_in_home(event, home), event["status"], event["category"], event["event_id"]))
            elif event["direction"] == "credit" and event["status"] == "scheduled":
                amount = self.repository.amount_in_home(event, home)
                flows.append(CashFlow(settlement, amount, "scheduled_credit", event["category"], event["event_id"]))
                if event["category"] == "salary" and not facts.suppress_recurring_income:
                    recurring_date = self._next_month_day(settlement, settlement.day)
                    while recurring_date <= horizon_end:
                        recurring_amount = self.repository.convert(self.repository.event_amount(event), event["currency"], home, recurring_date)
                        flows.append(CashFlow(recurring_date, recurring_amount, "scheduled_salary_recurrence", "salary", recurrence_id="salary:scheduled"))
                        recurring_date = self._next_month_day(recurring_date, settlement.day)
        return flows

    def _evidence_credits(self, facts, request_date, horizon_end, home) -> list[CashFlow]:
        flows: list[CashFlow] = []
        for day, amount, currency, source_id in facts.one_time_credits:
            if not request_date <= day <= horizon_end:
                continue
            converted = self.repository.try_convert(amount, currency, home, day)
            if converted is None:
                # Never count income we cannot price. Dropping it is the safer reading.
                facts.ignored_evidence.append(
                    {"kind": "one_time_credit", "reason": "no_dated_exchange_rate", "currency": currency, "source": source_id}
                )
                continue
            flows.append(CashFlow(day, converted, "confirmed_message_credit", "salary", source_id))
        return flows

    @staticmethod
    def _deduplicate_salary(flows: list[CashFlow]) -> list[CashFlow]:
        priority = {
            "scheduled_credit": 0,
            "message_salary": 1,
            "message_salary_once": 1,
            "message_salary_date": 1,
            "scheduled_salary_recurrence": 2,
            "salary_recurrence": 3,
            "conservative_income_recurrence": 4,
        }
        result: list[CashFlow] = []
        seen_dates: set[date] = set()
        for flow in sorted(flows, key=lambda item: (item.date, priority.get(item.source, 9))):
            if flow.category != "salary" or flow.amount <= ZERO:
                result.append(flow)
                continue
            # A confirmed one-time adjustment is additive. Recurring salary
            # representations on the same day are alternatives; keep the
            # most explicit source rather than counting both amounts.
            if flow.source == "confirmed_message_credit":
                result.append(flow)
            elif flow.date not in seen_dates:
                seen_dates.add(flow.date)
                result.append(flow)
        return result

    @staticmethod
    def _normalize_first_income_cycle(flows: list[CashFlow], request_date: date) -> list[CashFlow]:
        income_dates = sorted({flow.date for flow in flows if flow.category == "salary" and flow.amount > ZERO})
        if not income_dates or income_dates[0] <= request_date or (income_dates[0] - request_date).days < 7:
            return flows
        first_income = income_dates[0]
        result = list(flows)
        for category in HIGH_FREQUENCY_CATEGORIES:
            recurring = sorted(
                (flow for flow in result if flow.source == "recurrence" and flow.category == category),
                key=lambda flow: flow.date,
            )
            if not recurring:
                continue
            before = [flow for flow in recurring if request_date <= flow.date < first_income]
            cadence = (recurring[1].date - recurring[0].date).days if len(recurring) > 1 else 30
            cycle_days = (first_income - request_date).days
            desired_count = max(1, min(3, cycle_days // max(1, cadence)))
            if len(before) > desired_count:
                remove = set(before[desired_count:])
                result = [flow for flow in result if flow not in remove]
            elif len(before) < desired_count:
                template = recurring[0]
                missing = desired_count - len(before)
                occupied = {flow.date for flow in before}
                for index in range(missing):
                    inserted_date = request_date + timedelta(days=max(1, cycle_days * (index + 1) // (missing + 1)))
                    while inserted_date in occupied and inserted_date + timedelta(days=1) < first_income:
                        inserted_date += timedelta(days=1)
                    occupied.add(inserted_date)
                    result.append(
                        CashFlow(
                            inserted_date,
                            template.amount,
                            template.source,
                            template.category,
                            template.event_id,
                            template.recurrence_id,
                            template.flexibility,
                            template.minimum_amount,
                        )
                    )
        return result

    @staticmethod
    def _suppress_resolved_failed_cycles(flows, events, facts, request_date):
        failed_categories = {
            event["category"]
            for event in events
            if event["status"] in {"failed", "cancelled"}
            and event["event_id"] not in facts.include_failed_event_ids
            and event["event_date"]
            and 0 <= (request_date - date.fromisoformat(event["event_date"])).days <= 10
        }
        result = list(flows)
        for category in failed_categories:
            candidates = sorted(
                (flow for flow in result if flow.source == "recurrence" and flow.category == category and flow.date <= request_date + timedelta(days=10)),
                key=lambda flow: flow.date,
            )
            if candidates:
                result.remove(candidates[0])
        return result

    def safety(self, opening_balance, minimum_balance, baseline, payments=(), changes=(), start_date=None) -> SafetyResult:
        change_by_id = {change.recurrence_id: change for change in changes}
        dated: dict[date, Decimal] = defaultdict(lambda: ZERO)
        for flow in baseline:
            amount = flow.amount
            change = change_by_id.get(flow.recurrence_id)
            if change and amount < ZERO:
                if change.action == "stop":
                    continue
                amount = -change.new_amount
            dated[flow.date] += amount
        for payment_date, amount in payments:
            dated[payment_date] -= amount
        minimum = opening_balance
        minimum_date = start_date or min(dated, default=date.today())
        balance = opening_balance
        for day in sorted(dated):
            balance += dated[day]
            if balance < minimum:
                minimum, minimum_date = balance, day
        return SafetyResult(minimum >= minimum_balance, minimum, minimum_date)

    def capacity_today(self, request_date, opening_balance, minimum_balance, baseline, profile=None) -> Decimal:
        if profile is None:
            if self.reserve_policy_name != "current_baseline":
                raise ValueError("profile is required for non-baseline reserve policies")
            profile = {}
        base_capacity = self.reserve_policy.capacity(
            self, request_date, opening_balance, minimum_balance, baseline, profile
        )
        variable_reserve, _ = self.essential_variable_reserve(request_date, baseline, profile)
        return max(ZERO, base_capacity - variable_reserve)

    def essential_variable_reserve(self, request_date, baseline, profile):
        """Reserve a quantile of complete trailing seven-day essential-variable spend.

        A category is variable only when its history is more numerous than the
        dataset's clean monthly-series ceiling and has at least three distinct
        descriptions. The weekly quantile captures both visit frequency and
        transaction size without treating merchants as recurring bills.
        """
        if self.variable_estimator == "none" or self.variable_days == "none" or not profile:
            return ZERO, {"estimator": "none", "days": 0, "daily_rate": "0", "categories": []}
        protected = set(filter(None, profile["expense_categories_to_protect"].split("|")))
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for event in self.repository.events_by_user.get(profile["user_id"], []):
            if (
                event["direction"] == "debit"
                and event["status"] == "settled"
                and event["settlement_date"]
                and date.fromisoformat(event["settlement_date"]) <= request_date
            ):
                grouped[event["category"]].append(event)
        category_rates: dict[str, Decimal] = {}
        for category, series in grouped.items():
            descriptions = {event["description"].strip().casefold() for event in series}
            if (
                category not in protected
                or len(series) < self.variable_event_threshold
                or len(descriptions) < 3
                or any(event["flexibility"] != "fixed" for event in series)
            ):
                continue
            dates = [date.fromisoformat(event["settlement_date"]) for event in series]
            observation_days = (request_date - min(dates)).days + 1
            complete_weeks = max(1, observation_days // 7)
            weekly_spend = []
            for index in range(complete_weeks):
                upper = request_date - timedelta(days=7 * index)
                lower = upper - timedelta(days=7)
                weekly_spend.append(sum(
                    (
                        self.repository.amount_in_home(event, profile["home_currency"])
                        for event in series
                        if lower < date.fromisoformat(event["settlement_date"]) <= upper
                    ),
                    ZERO,
                ))
            weekly_spend.sort()
            if self.variable_estimator == "median":
                weekly_estimate = Decimal(str(statistics.median(weekly_spend)))
            else:
                rank = max(0, (3 * len(weekly_spend) + 3) // 4 - 1)
                weekly_estimate = weekly_spend[rank]
            category_rates[category] = weekly_estimate / Decimal("7")
        if self.variable_days == "next_income":
            incomes = sorted(
                flow.date for flow in baseline
                if flow.amount > ZERO and flow.category == "salary" and flow.date >= request_date
            )
            days = (incomes[0] - request_date).days if incomes else self.horizon_days
        else:
            days = int(self.variable_days)
        daily_rate = sum(category_rates.values(), ZERO)
        reserve = (daily_rate * Decimal(days)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        return reserve, {
            "estimator": self.variable_estimator,
            "days": days,
            "daily_rate": format(daily_rate, "f"),
            "categories": sorted(category_rates),
            "category_daily_rates": {key: format(value, "f") for key, value in sorted(category_rates.items())},
            "reserve": format(reserve, "f"),
            "event_threshold": self.variable_event_threshold,
        }

    def _derive_variable_event_threshold(self) -> int:
        """Learn the event-count boundary just above clean monthly bill histories."""
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for event in self.repository.events:
            if event["direction"] == "debit" and event["status"] == "settled" and event["settlement_date"]:
                grouped[(event["user_id"], event["category"])].append(event)
        monthly_counts = []
        for series in grouped.values():
            if len({event["description"].strip().casefold() for event in series}) != 1 or len(series) < 3:
                continue
            days = sorted(date.fromisoformat(event["settlement_date"]) for event in series)
            gaps = [(right - left).days for left, right in zip(days, days[1:])]
            if gaps and 27 <= statistics.median(gaps) <= 32:
                monthly_counts.append(len(series))
        if not monthly_counts:
            raise ValueError("cannot derive variable-spend threshold without clean monthly histories")
        return max(monthly_counts) + 1

    @staticmethod
    def eligible_changes(baseline) -> tuple[SpendingChange, ...]:
        representatives: dict[str, CashFlow] = {}
        for flow in baseline:
            if flow.source == "recurrence" and flow.flexibility != "fixed":
                representatives[flow.recurrence_id] = flow
        changes: list[SpendingChange] = []
        for recurrence_id, flow in sorted(representatives.items()):
            if flow.flexibility in {"stoppable", "reducible_or_stoppable"}:
                changes.append(SpendingChange("stop", recurrence_id, flow.event_id))
            if flow.flexibility in {"reducible", "reducible_or_stoppable"} and flow.minimum_amount is not None:
                changes.append(SpendingChange("reduce_to", recurrence_id, flow.event_id, flow.minimum_amount))
        return tuple(changes)

    def _remove_outliers(self, series, home):
        if len(series) < 5:
            return series
        values = [self.repository.amount_in_home(event, home) for event in series]
        median = Decimal(str(statistics.median(values)))
        filtered = [event for event, value in zip(series, values) if median / Decimal("2.5") <= value <= median * Decimal("2.5")]
        return filtered or series

    def _conservative_amount(self, series, home) -> Decimal:
        values = [self.repository.amount_in_home(event, home) for event in series[-3:]]
        if max(values) - min(values) <= Decimal("0.01"):
            return values[-1]
        estimate = sum(values) / Decimal(len(values))
        quantum = Decimal("50") if home == "IDR" else Decimal("1")
        return (estimate / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum

    @staticmethod
    def _infer_schedule(series, request_date):
        dates = [date.fromisoformat(row["settlement_date"]) for row in series]
        if len(dates) < 3:
            return None
        intervals = [(right - left).days for left, right in zip(dates, dates[1:])]
        cadence = round(statistics.median(intervals[-6:]))
        if not 5 <= cadence <= 35 or (request_date - dates[-1]).days > max(45, round(cadence * 1.75)):
            return None
        monthly = 27 <= cadence <= 32
        day_of_month = Counter(day.day for day in dates[-4:]).most_common(1)[0][0]
        return cadence, monthly, day_of_month

    @classmethod
    def _next_occurrence(cls, current, cadence, monthly, day_of_month):
        return cls._next_month_day(current, day_of_month) if monthly else current + timedelta(days=cadence)

    @staticmethod
    def _next_month_day(current, day_of_month):
        year = current.year + (current.month == 12)
        month = current.month % 12 + 1
        return date(year, month, min(day_of_month, monthrange(year, month)[1]))

    @staticmethod
    def _day_on_or_after(current, day_of_month):
        candidate = date(current.year, current.month, min(day_of_month, monthrange(current.year, current.month)[1]))
        return candidate if candidate >= current else ForecastEngine._next_month_day(current, day_of_month)
