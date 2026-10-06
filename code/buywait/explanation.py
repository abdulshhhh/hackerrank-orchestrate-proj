from __future__ import annotations

import json
import re
import threading
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .llm import AnthropicRuntime, ModelUsage


NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
MAX_CHARACTERS = 300

# Each recommended method must be named in the prose using its own vocabulary. The
# patterns match word stems and common synonyms rather than one fixed phrase, because
# a semantically correct explanation ("no payment plan is recommended") must not be
# rejected for failing to contain a specific literal substring.
METHOD_PATTERNS = {
    "full_payment": r"\bpay(?:ing|s|ment)?\b|\bsettle\b|\bin full\b",
    "partial_payment": r"\bpartial\w*\b|\bremaining\b|\brest\b|\bpart of\b|\bbalance of\b|\bsecond payment\b|\btwo payments\b",
    "installments": r"\binstal?lments?\b|\binstal?ment\b|\b(?:two|three|four|\d+)\s+payments\b|\bmonthly payments\b",
    "wait": r"\bwait\w*\b|\blater\b|\buntil\b|\bdelay\w*\b|\bpostpon\w*\b|\bhold off\b|\bdefer\w*\b|\bfrom \d{4}-\d{2}-\d{2}\b|\bby \d{4}-\d{2}-\d{2}\b|\bon \d{4}-\d{2}-\d{2}\b",
    "not_recommended": (
        r"\bnot\s+(?:afford\w*|recommend\w*|advis\w*|safe|possible|feasible|proceed\w*)\b"
        r"|\bdo(?:es)? not\b|\bdon't\b|\bcannot\b|\bcan't\b|\bunaffordable\b|\binsufficient\b"
        r"|\bno\s+(?:payment\s+)?plan\b|\bavoid\b|\bnot\s+enough\b|\bshould not\b"
    ),
}

# Only genuine contradictions are listed. A capacity statement such as "only 608.6 is
# safe to pay now" is correct on an affordable_later row, so the forbidden patterns
# target claims about the FULL amount rather than the bare words "pay now".
STATUS_CONTRADICTIONS = {
    "affordable_now": (
        r"\bcannot afford\w*\b",
        r"\bnot affordable\b",
        r"\bdo not proceed\b",
        r"\bnot recommended\b",
        r"\bwait until\b",
    ),
    "affordable_with_plan": (
        r"\bdo not proceed\b",
        r"\bnot recommended\b",
        r"\bcannot be (?:paid|afforded|completed)\b",
    ),
    "affordable_later": (
        r"\baffordable now\b",
        r"\bpay (?:the )?full[^.]{0,40}\b(?:now|today)\b",
        r"\bfull (?:payment|amount)[^.]{0,30}\bis safe\b",
    ),
    "not_affordable": (
        r"\baffordable now\b",
        r"\bpay (?:the )?full[^.]{0,40}\b(?:now|today)\b",
        r"\bsafe in full\b",
        r"\bfull (?:payment|amount)[^.]{0,30}\bis safe\b",
    ),
}

SPENDING_CHANGE_PATTERN = r"\bstop\w*\b|\breduc\w*\b|\bcut\w*\b|\blower\w*\b|\btrim\w*\b|\bpaus\w*\b|\bcancel\w*\b|\bdrop\w*\b|\bscale back\b"

REPAIR_HINTS = {
    "missing": "Your previous draft was empty. Emit one grounded explanation.",
    "length": f"Your previous draft exceeded {MAX_CHARACTERS} characters. Rewrite it in at most 240 characters.",
    "csv_safety": "Your previous draft contained a quote, pipe, or line break. Emit one single line of plain prose.",
    "sentence_count": "Your previous draft did not use 1 to 3 sentences. Rewrite it as 1 to 3 complete sentences.",
    "ungrounded_number": "Your previous draft used a number that is not in the trace. Never compute, total, or round; quote figures exactly as they appear.",
    "method_contradiction": "Your previous draft did not clearly name the recommended_payment_method. State that action explicitly in plain words.",
    "status_contradiction": "Your previous draft contradicted the affordability_status. Describe only the status recorded in the trace.",
    "missing_spending_change": "Your previous draft omitted the required spending changes. Name the commitment in plain words and the action taken on it.",
    "missing_minimum_balance": "Your previous draft did not state minimum_balance_to_keep. Name that exact figure as the balance the plan protects.",
}


class ExplanationWriter:
    """Write grounded prose from a bounded decision trace, verify it, repair once, then fail closed.

    The model never sees the dataset and never influences the decision: it receives the
    already-final trace and returns prose only. Every draft is checked deterministically
    for number grounding, status and method consistency, and CSV safety. A rejected draft
    earns exactly one repair attempt carrying the rejection reason; a second failure falls
    back to the pre-validated deterministic template.
    """

    def __init__(self, runtime: AnthropicRuntime | None = None) -> None:
        self.runtime = runtime or AnthropicRuntime()
        self.usage: ModelUsage = self.runtime.usage
        self.tool = {
            "name": "emit_decision_explanation",
            "description": "Emit one grounded explanation of the supplied final decision.",
            "input_schema": {
                "type": "object",
                "properties": {"explanation": {"type": "string"}},
                "required": ["explanation"],
            },
        }
        self.system_prompt = (
            Path(__file__).resolve().parents[1] / "prompts" / "explanation_system.txt"
        ).read_text(encoding="utf-8")
        self._lock = threading.Lock()
        self.total_count = 0
        self.fallback_count = 0
        self.repaired_count = 0
        self.rejection_counts: dict[str, int] = {}
        self.first_pass_rejections: dict[str, int] = {}

    def _draft(self, trace: dict, repair_reason: str | None) -> tuple[str, str | None]:
        payload = trace if repair_reason is None else {**trace, "revision_note": REPAIR_HINTS[repair_reason]}
        result = self.runtime.forced_tool(
            "decision_explanation",
            self.system_prompt,
            payload,
            self.tool,
            max_tokens=180,
        )
        candidate = result.get("explanation", "").strip() if result else ""
        return candidate, self.validate(candidate, trace)

    def write(self, trace: dict, fallback: str) -> tuple[str, bool]:
        candidate, rejection = self._draft(trace, None)
        first_pass = rejection
        repaired = False
        if rejection:
            retry, retry_rejection = self._draft(trace, rejection)
            if not retry_rejection:
                candidate, rejection, repaired = retry, None, True
            else:
                rejection = retry_rejection
        with self._lock:
            self.total_count += 1
            if first_pass:
                self.first_pass_rejections[first_pass] = self.first_pass_rejections.get(first_pass, 0) + 1
            if repaired:
                self.repaired_count += 1
            if rejection:
                self.fallback_count += 1
                self.rejection_counts[rejection] = self.rejection_counts.get(rejection, 0) + 1
        return (fallback if rejection else candidate), bool(rejection)

    @classmethod
    def validate(cls, text: str, trace: dict) -> str | None:
        if not text:
            return "missing"
        if len(text) >= MAX_CHARACTERS:
            return "length"
        if any(character in text for character in ("\n", "\r", '"', "|")):
            return "csv_safety"
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
        if not 1 <= sentence_count <= 3:
            return "sentence_count"
        allowed_numbers = cls._numbers(json.dumps(trace, sort_keys=True, ensure_ascii=False))
        if not cls._numbers(text) <= allowed_numbers:
            return "ungrounded_number"
        lowered = text.casefold()
        status = trace["decision"]["affordability_status"]
        method = trace["decision"]["recommended_payment_method"]
        if any(re.search(pattern, lowered) for pattern in STATUS_CONTRADICTIONS[status]):
            return "status_contradiction"
        if not re.search(METHOD_PATTERNS[method], lowered):
            return "method_contradiction"
        changes = trace["decision"]["spending_changes_needed"]
        if changes != "none" and not re.search(SPENDING_CHANGE_PATTERN, lowered):
            return "missing_spending_change"
        # The golden house style always names the balance the user asked to keep as the
        # figure being protected, so an explanation that cites some other reserve reads
        # as a different -- and less checkable -- justification.
        try:
            minimum = Decimal(str(trace["minimum_balance_to_keep"]))
        except (KeyError, InvalidOperation):
            return None
        if minimum not in cls._numbers(text):
            return "missing_minimum_balance"
        return None

    @staticmethod
    def _numbers(text: str) -> set[Decimal]:
        values: set[Decimal] = set()
        for token in NUMBER_RE.findall(text):
            try:
                values.add(Decimal(token.replace(",", "")))
            except InvalidOperation:
                continue
        return values
