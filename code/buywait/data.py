from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from .vision import ImageEvidenceTool


INPUT_SCHEMAS = {
    "requests.csv": {"request_id", "user_id", "request_date", "requested_amount", "desired_completion_date"},
    "financial_profiles.csv": {"user_id", "home_currency", "current_available_balance", "minimum_balance_to_keep"},
    "financial_events.csv": {"event_id", "user_id", "direction", "amount", "currency", "settlement_date", "status"},
    "request_payment_options.csv": {"payment_option_id", "request_id", "payment_method", "payment_amount"},
    "messages.csv": {"message_id", "user_id", "sent_at", "source_type", "message_text"},
    "images.csv": {"image_id", "user_id", "related_event_id"},
    "exchange_rates.csv": {"rate_date", "from_currency", "to_currency", "rate"},
}


class DataError(ValueError):
    pass


class DataRepository:
    """Validated, indexed access to the participant-facing dataset only."""

    def __init__(self, dataset_dir: Path, evidence_cache: Path | None = None) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.requests = self._read("requests.csv")
        self.profiles = self._read("financial_profiles.csv")
        self.events = self._read("financial_events.csv")
        self.options = self._read("request_payment_options.csv")
        self.messages = self._read("messages.csv")
        self.images = self._read("images.csv")
        self.rates = self._read("exchange_rates.csv")
        self.profile_by_user = self._unique(self.profiles, "user_id")
        self.request_by_id = self._unique(self.requests, "request_id")
        self.events_by_user = self._group(self.events, "user_id")
        self.options_by_request = self._group(self.options, "request_id")
        self.messages_by_user = self._group(self.messages, "user_id")
        self.images_by_event = self._unique(self.images, "related_event_id")
        self.images_by_user = self._group(self.images, "user_id")
        self.rate_by_key = {
            (row["rate_date"], row["from_currency"], row["to_currency"]): Decimal(row["rate"])
            for row in self.rates
        }
        cache_path = evidence_cache or Path(__file__).with_name("image_facts.json")
        cached_facts = self._load_image_facts(cache_path)
        self.image_tool = ImageEvidenceTool(cache_path)
        self.event_by_id = event_by_id = self._unique(self.events, "event_id")
        self.image_facts = self.image_tool.resolve_all(self.images, self.dataset_dir, cached_facts, event_by_id)
        self.vision_usage = self.image_tool.runtime.usage
        self.image_evidence_issues = [*self.image_tool.issues, *self.image_tool.disagreements]
        self._validate_relations()

    def _read(self, name: str) -> list[dict[str, str]]:
        path = self.dataset_dir / name
        if not path.is_file():
            raise DataError(f"missing required input: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = INPUT_SCHEMAS[name] - fields
            if missing:
                raise DataError(f"{name} is missing columns: {sorted(missing)}")
            return list(reader)

    @staticmethod
    def _unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            value = row[key]
            if value in result:
                raise DataError(f"duplicate {key}: {value}")
            result[value] = row
        return result

    @staticmethod
    def _group(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            result[row[key]].append(row)
        return dict(result)

    @staticmethod
    def _load_image_facts(path: Path) -> dict[str, dict[str, str]]:
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["image_id"]: item for item in payload.get("images", [])}

    def _validate_relations(self) -> None:
        for request in self.requests:
            if request["user_id"] not in self.profile_by_user:
                raise DataError(f"missing profile for {request['request_id']}")
        for event in self.events:
            if not event["amount"] and event["event_id"] not in self.images_by_event:
                raise DataError(f"blank amount lacks image evidence: {event['event_id']}")

    def event_amount(self, event: dict[str, str]) -> Decimal:
        if event["amount"]:
            return Decimal(event["amount"])
        image_row = self.images_by_event[event["event_id"]]
        image_id = image_row["image_id"]
        fact = self.image_facts.get(image_id)
        if not fact:
            raise DataError(f"no extracted fact for {image_id}")
        image_path = self.dataset_dir / "media" / "images" / f"{image_id}.png"
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != fact["sha256"]:
            raise DataError(f"cached image fact hash mismatch for {image_id}")
        return Decimal(fact["amount"])

    def convert(self, amount: Decimal, currency: str, home_currency: str, on_date: date) -> Decimal:
        if currency == home_currency:
            return amount
        key = (on_date.isoformat(), currency, home_currency)
        try:
            return amount * self.rate_by_key[key]
        except KeyError as exc:
            raise DataError(f"missing dated exchange rate {key}") from exc

    def try_convert(self, amount: Decimal, currency: str, home_currency: str, on_date: date) -> Decimal | None:
        """Convert, or return None when no dated rate supports it.

        Dataset events must still fail loudly through convert(): a missing rate there is
        malformed input. Facts derived from untrusted messages are different -- a message
        may name any currency at all -- so an unconvertible one is dropped and recorded
        rather than allowed to halt the run.
        """
        try:
            return self.convert(amount, currency, home_currency, on_date)
        except DataError:
            return None

    def amount_in_home(self, event: dict[str, str], home_currency: str) -> Decimal:
        if not event["settlement_date"]:
            raise DataError(f"cash event has no settlement date: {event['event_id']}")
        return self.convert(
            self.event_amount(event), event["currency"], home_currency, date.fromisoformat(event["settlement_date"])
        )
