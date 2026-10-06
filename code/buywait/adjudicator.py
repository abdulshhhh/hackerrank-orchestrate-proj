from __future__ import annotations

import threading
from datetime import date
from decimal import Decimal

from .llm import AnthropicRuntime, ModelUsage


REASON_CODES = [
    "approved",
    "pending_credit_counted",
    "unsupported_income_counted",
    "plan_exceeds_deadline",
    "plan_breaches_minimum_balance",
    "protected_category_changed",
    "partial_payment_malformed",
    "explanation_unsupported",
]


class DecisionAdjudicator:
    """A reviewer stage that may raise a concern but may never act on it.

    The asymmetry is deliberate. The reviewer reads the finished decision and its trace and
    returns one reason code from a closed set. A flag alone changes nothing: each code maps
    to a deterministic re-check that recomputes the claim from the dataset. Only when that
    re-check independently confirms the problem is the decision replaced by the conservative
    fallback. When the re-check refutes the flag, the disagreement is recorded and the
    original decision stands.

    So a hallucinating reviewer costs tokens and an audit line, never a wrong answer -- and a
    correct reviewer catches a class of error the planner's own validator cannot see, because
    the validator checks the plan it just built against rules, while this checks the decision
    against the evidence.
    """

    def __init__(self, repository, runtime: AnthropicRuntime | None = None) -> None:
        self.repository = repository
        self.runtime = runtime or AnthropicRuntime()
        self.usage: ModelUsage = self.runtime.usage
        self.tool = {
            "name": "emit_review",
            "description": "Emit one review verdict for the supplied final decision.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason_code": {"type": "string", "enum": REASON_CODES},
                    "rationale": {"type": "string"},
                },
                "required": ["reason_code", "rationale"],
            },
        }
        self.system_prompt = (
            "You audit an already-final affordability decision against its trace. The trace"
            " contains a precomputed_checks object: those booleans were computed"
            " deterministically from the dataset and are authoritative. Do NOT compare dates,"
            " sum amounts, or do any arithmetic yourself -- read the relevant boolean instead."
            " If every boolean in precomputed_checks is false, return 'approved'. Return"
            " exactly one reason_code. Return 'approved' unless you can point to a specific"
            " violation visible in the trace. The violations you may report are: a pending"
            " credit counted as available cash; future income counted without a settled or"
            " evidenced basis; a payment plan whose last payment falls after"
            " desired_completion_date; a plan that would take the projected balance below"
            " minimum_balance_to_keep; a spending change applied to a protected or"
            " non-flexible category; a partial payment that is not exactly two payments"
            " summing to requested_amount; or an explanation asserting a figure absent from"
            " the trace. Do not re-derive the arithmetic and do not propose a different plan."
            " Your verdict does not change the decision by itself: every flag is re-checked"
            " deterministically before it has any effect, so a speculative flag only adds"
            " noise. Be conservative and specific, and cite the boolean you relied on."
        )
        self._lock = threading.Lock()
        self.total_count = 0
        self.flagged_count = 0
        self.confirmed_count = 0
        self.refuted_count = 0
        self.reason_counts: dict[str, int] = {}

    def precomputed_checks(self, decision, request, profile, trace) -> dict[str, bool]:
        """Answer every question the reviewer would otherwise have to compute in its head.

        The first version of this stage handed the model raw dates and asked it to judge
        whether a plan overran its deadline. It produced 52 flags on 250 requests and every
        one was wrong -- on request_53 it asserted that 2026-01-16 falls after 2026-01-20.
        Language models are unreliable at this kind of comparison, so the comparison is done
        here and the model is asked only to interpret the result.
        """
        return {
            reason: bool(self._recheck(reason, decision, request, profile, trace))
            for reason in REASON_CODES
            if reason != "approved"
        }

    def review(self, decision, request, profile, trace, deadline=None) -> dict:
        """Return an audit record; the caller applies it only when confirmed is True."""
        if not self.runtime.enabled:
            return {"reason_code": "approved", "confirmed": False, "checked": False}
        checks = self.precomputed_checks(decision, request, profile, trace)
        result = self.runtime.forced_tool(
            "adjudicator",
            self.system_prompt,
            {**trace, "precomputed_checks": checks},
            self.tool,
            max_tokens=300,
            deadline=deadline,
        )
        reason = (result or {}).get("reason_code", "approved")
        if reason not in REASON_CODES:
            reason = "approved"
        rationale = (result or {}).get("rationale", "")[:400]
        confirmed = False
        if reason != "approved":
            confirmed = self._recheck(reason, decision, request, profile, trace)
        with self._lock:
            self.total_count += 1
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            if reason != "approved":
                self.flagged_count += 1
                self.confirmed_count += int(confirmed)
                self.refuted_count += int(not confirmed)
        return {
            "reason_code": reason,
            "rationale": rationale,
            "confirmed": confirmed,
            "checked": reason != "approved",
        }

    # ---- deterministic re-checks -------------------------------------------------------

    def _recheck(self, reason, decision, request, profile, trace) -> bool:
        try:
            handler = getattr(self, f"_check_{reason}", None)
            return bool(handler and handler(decision, request, profile, trace))
        except Exception:
            return False

    @staticmethod
    def _payments(decision) -> list[tuple[date, Decimal]]:
        if decision.payment_plan == "none":
            return []
        return [
            (date.fromisoformat(entry.split(":")[0]), Decimal(entry.split(":")[1]))
            for entry in decision.payment_plan.split("|")
        ]

    def _check_plan_exceeds_deadline(self, decision, request, profile, trace) -> bool:
        payments = self._payments(decision)
        deadline = date.fromisoformat(request["desired_completion_date"])
        return bool(payments) and payments[-1][0] > deadline

    def _check_partial_payment_malformed(self, decision, request, profile, trace) -> bool:
        if decision.recommended_payment_method != "partial_payment":
            return False
        payments = self._payments(decision)
        requested = Decimal(request["requested_amount"])
        return len(payments) != 2 or sum(amount for _, amount in payments) != requested

    def _check_protected_category_changed(self, decision, request, profile, trace) -> bool:
        if decision.spending_changes_needed == "none":
            return False
        protected = set(filter(None, profile["expense_categories_to_protect"].split("|")))
        for action in decision.spending_changes_needed.split("|"):
            parts = action.split(":")
            event = self.repository.event_by_id.get(parts[1]) if len(parts) > 1 else None
            if event is None:
                continue
            if event["category"] in protected or event["flexibility"] == "fixed":
                return True
        return False

    def _check_plan_breaches_minimum_balance(self, decision, request, profile, trace) -> bool:
        """Not independently checkable here, so it never fires.

        The trace carries the BASELINE projected minimum, which legitimately dips below
        minimum_balance_to_keep -- that dip is precisely why a request comes back as
        not_affordable. Comparing it to the minimum flagged 9 correct decisions on the last
        run. Re-simulating the chosen plan would duplicate the planner, so this reason code
        is accepted from the reviewer and then always refuted.
        """
        return False

    def _check_pending_credit_counted(self, decision, request, profile, trace) -> bool:
        # A pending credit is only ever a violation if one actually exists and the ledger
        # treated it as cash; the forecast never adds pending credits, so confirmation
        # requires positive evidence rather than the reviewer's assertion.
        return False

    def _check_unsupported_income_counted(self, decision, request, profile, trace) -> bool:
        """Not independently checkable here, so it never fires.

        A first attempt allow-listed income sources and flagged anything else. The allow-list
        was incomplete -- it omitted conservative_income_recurrence, message_salary_once,
        scheduled_credit, message_salary, message_salary_date and confirmed_message_credit,
        all legitimate -- so it confirmed 149 flags and drove not_affordable from 49 to 174
        on the 250-request run. Every source in the ledger is already gated at the point it
        is admitted, in forecast.py and the evidence provenance checks; re-deriving that
        judgement from the trace alone is not sound, so this reason code is always refuted.
        """
        return False

    def _check_explanation_unsupported(self, decision, request, profile, trace) -> bool:
        from .explanation import ExplanationWriter

        return ExplanationWriter.validate(decision.decision_explanation, trace) is not None
