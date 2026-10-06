from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
from decimal import Decimal
import os
import time

from .adjudicator import DecisionAdjudicator
from .data import DataRepository
from .evidence import EvidenceInterpreter
from .explanation import ExplanationWriter
from .forecast import ForecastEngine
from .loop import EvidenceToolLoop
from .models import Decision
from .planner import PlanGenerator
from .semantic import SemanticEvidenceAugmenter
from .validator import DecisionValidator


class AffordabilityAgent:
    """Bounded single-agent orchestrator with deterministic financial tools.

    The orchestration stages are intentionally narrow: retrieve evidence,
    normalize it, build a ledger, enumerate plans, and independently validate.
    Arithmetic and safety gates are never delegated to free-form text.
    """

    MAX_STAGES = 5

    def __init__(
        self,
        repository: DataRepository,
        reserve_policy: str | None = None,
        variable_estimator: str | None = None,
        variable_days: str | None = None,
    ) -> None:
        self.repository = repository
        self.evidence = EvidenceInterpreter()
        self.forecast = ForecastEngine(
            repository,
            reserve_policy=reserve_policy,
            variable_estimator=variable_estimator,
            variable_days=variable_days,
        )
        self.planner = PlanGenerator(repository, self.forecast)
        self.validator = DecisionValidator(repository)
        currencies = {
            value
            for row in repository.rates
            for value in (row["from_currency"], row["to_currency"])
        }
        self.semantic = SemanticEvidenceAugmenter(currencies)
        self.tool_loop = EvidenceToolLoop(self.semantic)
        self.adjudicator = DecisionAdjudicator(repository)
        self.explanation = ExplanationWriter()
        self.use_tool_loop = os.getenv("BUYWAIT_TOOL_LOOP", "1") == "1"
        self.use_adjudicator = os.getenv("BUYWAIT_ADJUDICATOR", "1") == "1"

    def decide(self, request: dict[str, str]) -> tuple[Decision, dict]:
        deadline = time.monotonic() + self.semantic.runtime.timeout_seconds
        trace: dict = {"request_id": request["request_id"], "stages": []}
        profile = self.repository.profile_by_user[request["user_id"]]
        trace["stages"].append("retrieve_user_context")

        facts = self.evidence.interpret(
            self.repository.messages_by_user.get(request["user_id"], []),
            self.repository.events_by_user.get(request["user_id"], []),
            date.fromisoformat(request["request_date"]),
        )
        if self.use_tool_loop:
            facts, tool_sequence = self.tool_loop.gather(
                facts,
                self.repository.messages_by_user.get(request["user_id"], []),
                self.repository.events_by_user.get(request["user_id"], []),
                self.repository.images_by_user.get(request["user_id"], []),
                date.fromisoformat(request["request_date"]),
                self.repository,
                deadline,
            )
        else:
            facts = self.semantic.augment(
                facts,
                self.repository.messages_by_user.get(request["user_id"], []),
                self.repository.events_by_user.get(request["user_id"], []),
                date.fromisoformat(request["request_date"]),
                deadline,
            )
            tool_sequence = ["single_call_extractor"]
        trace["stages"].append("interpret_untrusted_evidence")

        baseline = self.forecast.build_baseline(request, profile, facts)
        _, variable_reserve = self.forecast.essential_variable_reserve(
            date.fromisoformat(request["request_date"]), baseline, profile
        )
        trace["stages"].append("build_90_day_ledger")

        decision = self.planner.decide(request, profile, baseline)
        trace["stages"].append("enumerate_and_rank_plans")

        self.validator.validate(decision, request, profile)
        explanation_trace = self._explanation_trace(request, profile, facts, baseline, variable_reserve, decision)
        explanation, used_fallback = self.explanation.write(explanation_trace, decision.decision_explanation)
        decision = replace(decision, decision_explanation=explanation)
        self.validator.validate(decision, request, profile)
        audit = {"reason_code": "approved", "confirmed": False, "checked": False}
        if self.use_adjudicator:
            audit = self.adjudicator.review(decision, request, profile, explanation_trace, deadline)
            if audit["confirmed"]:
                decision = self.planner.conservative_fallback(request, profile, decision, audit["reason_code"])
                self.validator.validate(decision, request, profile)
        trace["stages"].append("validate_output")
        if len(trace["stages"]) > self.MAX_STAGES:
            raise RuntimeError("agent exceeded stage cap")
        trace.update(
            {
                "evidence": self._json_safe(asdict(facts)),
                "baseline_flow_count": len(baseline),
                "essential_variable_reserve": variable_reserve,
                "explanation_trace": explanation_trace,
                "explanation_fallback": used_fallback,
                "image_evidence_issues": self.repository.image_evidence_issues,
                "evidence_tool_sequence": tool_sequence,
                "adjudication": audit,
                "decision": decision.as_row(),
            }
        )
        return decision, trace

    def _explanation_trace(self, request, profile, facts, baseline, variable_reserve, decision):
        request_date = date.fromisoformat(request["request_date"])
        opening = Decimal(profile["current_available_balance"])
        minimum = Decimal(profile["minimum_balance_to_keep"])
        pending = sum((-flow.amount for flow in baseline if flow.amount < 0 and flow.source == "pending"), Decimal("0"))
        scheduled = sum(
            (-flow.amount for flow in baseline if flow.amount < 0 and flow.source in {"scheduled", "failed_retry"}),
            Decimal("0"),
        )
        projected_outflow = sum((-flow.amount for flow in baseline if flow.amount < 0), Decimal("0"))
        baseline_safety = self.forecast.safety(opening, minimum, baseline, start_date=request_date)
        next_income = next(
            (
                {"date": flow.date, "amount": flow.amount, "source": flow.source}
                for flow in sorted(baseline, key=lambda item: item.date)
                if flow.date >= request_date and flow.amount > 0 and flow.category == "salary"
            ),
            None,
        )
        requested = Decimal(request["requested_amount"])
        if decision.amount_safe_to_pay >= requested:
            binding = "requested_amount_cap"
        elif Decimal(variable_reserve.get("reserve", "0")) > 0:
            binding = "essential_variable_spending_reserve"
        elif pending > 0:
            binding = "pending_debit_reserve"
        else:
            binding = "minimum_balance_and_projected_outflows"
        eligible_message_ids = [
            row["message_id"]
            for row in self.repository.messages_by_user.get(request["user_id"], [])
            if date.fromisoformat(row["sent_at"][:10]) <= request_date
        ]
        decision_fields = decision.as_row()
        decision_fields.pop("decision_explanation")
        decision_fields["payment_count"] = 0 if decision.payment_plan == "none" else len(decision.payment_plan.split("|"))
        return self._json_safe(
            {
                "request_id": request["request_id"],
                "request_date": request["request_date"],
                "desired_completion_date": request["desired_completion_date"],
                "currency": profile["home_currency"],
                "requested_amount": requested,
                "opening_balance": opening,
                "minimum_balance_to_keep": minimum,
                "forecast_horizon_days": self.forecast.horizon_days,
                "baseline_minimum_projected_balance": baseline_safety.minimum_projected_balance,
                "baseline_minimum_date": baseline_safety.minimum_date,
                "projected_debit_outflow": projected_outflow,
                "reserve_breakdown": {
                    "essential_variable": variable_reserve,
                    "pending_debits": pending,
                    "scheduled_or_retry_debits": scheduled,
                },
                "binding_constraint": binding,
                "next_confirmed_income": next_income,
                "evidence_facts": asdict(facts),
                "evidence_source_message_ids": eligible_message_ids,
                "spending_change_details": self._spending_change_details(decision.spending_changes_needed),
                "decision": decision_fields,
            }
        )

    def _spending_change_details(self, changes: str) -> list[dict[str, str]]:
        """Resolve each spending change to the human-readable commitment it affects.

        The output column carries opaque event ids, but a useful explanation names the
        commitment the user recognises ("the family streaming plan"), so the writer needs
        the description and category alongside the id.
        """
        if not changes or changes == "none":
            return []
        details: list[dict[str, str]] = []
        for token in changes.split("|"):
            parts = token.split(":")
            event = self.repository.event_by_id.get(parts[1]) if len(parts) > 1 else None
            if event is None:
                continue
            details.append(
                {
                    "action": parts[0],
                    "event_id": parts[1],
                    "description": event.get("description", ""),
                    "category": event.get("category", ""),
                    "current_amount": event.get("amount", ""),
                    "new_amount": parts[2] if len(parts) > 2 else "",
                }
            )
        return details

    @staticmethod
    def _json_safe(value):
        if isinstance(value, (date, Decimal)):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {key: AffordabilityAgent._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [AffordabilityAgent._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [AffordabilityAgent._json_safe(item) for item in value]
        return value
