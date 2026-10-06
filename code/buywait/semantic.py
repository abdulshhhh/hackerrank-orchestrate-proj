from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .evidence import EvidenceInterpreter, MONEY_RE
from .llm import AnthropicRuntime, ModelUsage
from .models import EvidenceFacts


ZERO = Decimal("0")


class SemanticEvidenceAugmenter:
    """Anthropic structured evidence extractor with literal provenance guardrails."""

    def __init__(self, currencies=(), runtime: AnthropicRuntime | None = None) -> None:
        self.runtime = runtime or AnthropicRuntime()
        self.usage: ModelUsage = self.runtime.usage
        currency_values = sorted(set(filter(None, currencies)))
        currency_schema = {"type": "string"}
        if currency_values:
            currency_schema["enum"] = currency_values
        nullable_currency = {"anyOf": [currency_schema, {"type": "null"}]}
        nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        self.tool = {
            "name": "emit_evidence_facts",
            "description": "Emit only financial facts directly supported by the supplied messages.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "proposals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": [
                                    "recurring_salary", "next_salary", "one_time_credit",
                                    "suppress_income", "rent_multiplier", "retry_failed_debit",
                                ]},
                                "message_id": {"type": "string"},
                                "amount": nullable_string,
                                "currency": nullable_currency,
                                "date": nullable_string,
                                "related_event_id": nullable_string,
                            },
                            "required": ["kind", "message_id", "amount", "currency", "date", "related_event_id"],
                        },
                    }
                },
                "required": ["proposals"],
            },
        }
        self.system_prompt = (Path(__file__).resolve().parents[1] / "prompts" / "evidence_system.txt").read_text(encoding="utf-8")

    def augment(self, base, messages, events, request_date, deadline: float | None = None) -> EvidenceFacts:
        eligible = [row for row in messages if date.fromisoformat(row["sent_at"][:10]) <= request_date]
        if not eligible:
            return base
        payload = {
            "request_date": request_date.isoformat(),
            "messages": [
                {
                    "message_id": row["message_id"],
                    "source_type": row["source_type"],
                    "related_event_id": row.get("related_event_id", ""),
                    "sent_at": row["sent_at"],
                    "message_text": row["message_text"],
                }
                for row in eligible
            ],
        }
        result = self.runtime.forced_tool(
            "evidence_interpreter", self.system_prompt, payload, self.tool, deadline=deadline
        )
        if not result:
            return base
        return self._merge_verified(base, result.get("proposals", []), eligible, events)

    @staticmethod
    def _merge_verified(base, proposals, messages, events) -> EvidenceFacts:
        message_by_id = {row["message_id"]: row for row in messages}
        event_by_id = {row["event_id"]: row for row in events}
        for proposal in proposals:
            message = message_by_id.get(proposal.get("message_id", ""))
            if not message:
                continue
            source = message["source_type"]
            text = message["message_text"]
            literal_money = {(currency.upper(), Decimal(amount)) for currency, amount in MONEY_RE.findall(text)}
            literal_dates = set(EvidenceInterpreter._extract_dates(text))
            try:
                amount = Decimal(proposal["amount"]) if proposal.get("amount") is not None else None
                fact_date = date.fromisoformat(proposal["date"]) if proposal.get("date") else None
            except (InvalidOperation, ValueError):
                continue
            currency = proposal.get("currency")
            kind = proposal.get("kind")
            if kind != "rent_multiplier" and amount is not None and (currency, amount) not in literal_money:
                continue
            if fact_date is not None and fact_date not in literal_dates:
                continue
            if kind in {"recurring_salary", "next_salary"} and source == "employer" and amount is not None:
                if kind == "recurring_salary" and base.recurring_salary_amount is None:
                    base.recurring_salary_amount, base.recurring_salary_currency = amount, currency
                    base.recurring_salary_start = fact_date
                    base.recurring_salary_day = fact_date.day if fact_date else None
                elif kind == "next_salary" and base.next_salary_amount is None:
                    base.next_salary_amount, base.next_salary_currency, base.next_salary_date = amount, currency, fact_date
            elif kind == "one_time_credit" and source in {"employer", "service_provider", "financial_service"} and amount is not None and fact_date:
                item = (fact_date, amount, currency, message["message_id"])
                if item not in base.one_time_credits:
                    base.one_time_credits.append(item)
            elif kind == "suppress_income" and source in {"employer", "service_provider"}:
                base.suppress_recurring_income = True
            elif kind == "rent_multiplier" and source == "service_provider" and amount is not None:
                literal_numbers = {Decimal(value) for value in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", text)}
                if amount in literal_numbers and ZERO < amount < Decimal("100"):
                    base.rent_multiplier = Decimal("1") + amount / Decimal("100")
            elif kind == "retry_failed_debit" and source == "bank":
                event_id = proposal.get("related_event_id") or message.get("related_event_id", "")
                if event_id in event_by_id and event_by_id[event_id]["status"] == "failed":
                    base.include_failed_event_ids.add(event_id)
        return base
