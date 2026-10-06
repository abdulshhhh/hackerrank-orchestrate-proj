from __future__ import annotations

import re
from calendar import monthrange
from datetime import date
from decimal import Decimal

from .models import EvidenceFacts


DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
NATURAL_DATE_RE = re.compile(
    r"\b([0-3]?\d)\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)"
    r"\s+(20\d{2})\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"\b(INR|IDR|EUR|USD|ZAR)\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

MONTHS = {
    name.casefold(): index
    for index, names in enumerate(
        (
            (),
            ("January", "Januari"), ("February", "Februari"), ("March", "Maret"),
            ("April",), ("May", "Mei"), ("June", "Juni"), ("July", "Juli"),
            ("August", "Agustus"), ("September",), ("October", "Oktober"),
            ("November",), ("December", "Desember"),
        )
    )
    for name in names
}

END_INCOME = (
    "contract has ended",
    "kontrak musiman saat ini telah berakhir",
    "employment has ended",
    "pekerjaan anda telah berakhir",
    "hubungan kerja anda telah berakhir",
    "tidak ada pembayaran gaji rutin yang dijadwalkan",
    "no off-season income",
    "belum ada pendapatan di luar musim",
)
UNAVAILABLE_INCOME = (
    "payout is still pending",
    "pembayaran berikutnya dari",
    "is still in payment processing",
    "masih dalam proses pembayaran",
    "has not been credited",
    "belum masuk ke rekening",
    "still pending approval",
    "masih menunggu persetujuan",
)
SALARY_HINTS = ("salary", "payroll", "monthly pay", "gaji", "penggajian")
NEXT_ONLY_HINTS = (
    "next salary is reduced",
    "temporary monthly pay",
    "gaji bulanan sementara",
    "jumlah yang lebih rendah",
    "approved unpaid leave",
)
RECURRING_HINTS = (
    "increased to",
    "naik menjadi",
    "confirmed base salary",
    "gaji pokok yang dikonfirmasi",
    "remaining confirmed monthly salary",
    "sisa gaji bulanan yang dikonfirmasi",
    "first salary",
    "gaji pertama",
    "regular salary",
    "gaji rutin",
    "resumes on",
)
INVOICE_CONFIRMED = (
    "client approved an invoice payment",
    "klien menyetujui pembayaran faktur",
    "invoice payment of",
)
FAILED_DEBIT = (
    "previous debit attempt failed",
    "debit sebelumnya gagal",
    "another debit will be attempted",
    "debit akan dicoba lagi",
)


class EvidenceInterpreter:
    """Convert untrusted prose into an allow-listed set of financial facts.

    Text is never executed and cannot alter the decision policy. Only facts
    matching a source-aware financial template are accepted.
    """

    def interpret(
        self,
        messages: list[dict[str, str]],
        events: list[dict[str, str]],
        request_date: date,
    ) -> EvidenceFacts:
        facts = EvidenceFacts()
        event_by_id = {event["event_id"]: event for event in events}
        ordered = sorted(messages, key=lambda row: (row["sent_at"], row["message_id"]))
        for message in ordered:
            if date.fromisoformat(message["sent_at"][:10]) > request_date:
                continue
            text = message["message_text"]
            lower = text.casefold()
            source = message["source_type"]
            dates = self._extract_dates(text)
            monies = [(currency.upper(), Decimal(amount)) for currency, amount in MONEY_RE.findall(text)]

            if source in {"employer", "service_provider"} and any(term in lower for term in END_INCOME):
                facts.suppress_recurring_income = True
                facts.ignored_evidence.append(f"{message['message_id']}: ended income")
                continue

            salary_source = source == "employer" or (source == "financial_service" and "employer has confirmed" in lower)
            if salary_source and any(term in lower for term in SALARY_HINTS):
                if "bonus" in lower and ("not been approved" in lower or "belum disetujui" in lower):
                    facts.ignored_evidence.append(f"{message['message_id']}: unconfirmed bonus")
                    continue
                if "commission" in lower and ("pending" in lower or "belum disetujui" in lower):
                    facts.ignored_evidence.append(f"{message['message_id']}: unconfirmed commission")
                if monies:
                    currency, amount = monies[0]
                    if source == "financial_service" and dates:
                        facts.one_time_credits.append((dates[0], amount, currency, message["message_id"]))
                    elif any(term in lower for term in NEXT_ONLY_HINTS):
                        facts.next_salary_amount = amount
                        facts.next_salary_currency = currency
                        facts.next_salary_date = dates[0] if dates else None
                    elif any(term in lower for term in RECURRING_HINTS) or dates:
                        facts.recurring_salary_amount = amount
                        facts.recurring_salary_currency = currency
                        facts.recurring_salary_start = dates[0] if dates else None
                        if dates:
                            facts.recurring_salary_day = dates[0].day
                    if len(monies) > 1 and ("arrears" in lower or "tunggakan" in lower):
                        credit_date = dates[0] if dates else self._next_month_day(request_date, 15)
                        arrears_currency, arrears = monies[1]
                        facts.one_time_credits.append((credit_date, arrears, arrears_currency, message["message_id"]))
                elif dates and ("expected on" in lower or "diperkirakan masuk" in lower):
                    facts.next_salary_date = dates[0]
                    facts.recurring_salary_day = dates[0].day
                if "childcare payment begins" in lower or "pembayaran penitipan anak" in lower:
                    facts.add_childcare_recurrence = True
                continue

            if source == "service_provider" and any(term in lower for term in INVOICE_CONFIRMED) and monies and dates:
                currency, amount = monies[0]
                facts.one_time_credits.append((dates[0], amount, currency, message["message_id"]))
                continue

            if source == "service_provider" and ("rent" in lower or "sewa" in lower) and "12%" in lower:
                facts.rent_multiplier = Decimal("1.12")
                continue

            if source == "bank" and any(term in lower for term in FAILED_DEBIT):
                related = message.get("related_event_id", "")
                if related and related in event_by_id:
                    facts.include_failed_event_ids.add(related)
                continue

            if any(term in lower for term in UNAVAILABLE_INCOME):
                facts.ignored_evidence.append(f"{message['message_id']}: unavailable credit")
                if source == "service_provider" and ("payout" in lower or "pembayaran berikutnya" in lower):
                    facts.suppress_recurring_income = True

        return facts

    @staticmethod
    def _next_month_day(day: date, target_day: int) -> date:
        year = day.year + (day.month == 12)
        month = day.month % 12 + 1
        return date(year, month, min(target_day, monthrange(year, month)[1]))

    @staticmethod
    def _extract_dates(text: str) -> list[date]:
        found = [date.fromisoformat(value) for value in DATE_RE.findall(text)]
        for day, month, year in NATURAL_DATE_RE.findall(text):
            value = date(int(year), MONTHS[month.casefold()], int(day))
            if value not in found:
                found.append(value)
        return found
