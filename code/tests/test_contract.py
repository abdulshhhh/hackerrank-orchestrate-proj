"""Output-contract regressions.

Two halves, and the second matters as much as the first: the committed output.csv must
satisfy section 6.2, AND the verifier must actually reject a file that does not. A checker
that passes everything is worse than no checker, because it reads like assurance.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT if (CODE_ROOT / "dataset" / "requests.csv").is_file() else CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "evaluation"))

import verify_output  # noqa: E402


DATASET = REPO_ROOT / "dataset"
OUTPUT = REPO_ROOT / "output.csv"


@unittest.skipUnless(OUTPUT.is_file(), "output.csv has not been generated yet")
class CommittedOutputTests(unittest.TestCase):
    def test_committed_output_satisfies_the_contract(self):
        problems = verify_output.verify(OUTPUT, DATASET)
        self.assertEqual([], problems, f"{len(problems)} contract violation(s)")


@unittest.skipUnless(OUTPUT.is_file(), "output.csv has not been generated yet")
class VerifierDetectsCorruptionTests(unittest.TestCase):
    """Mutate one field at a time and require the verifier to notice."""

    @classmethod
    def setUpClass(cls):
        with OUTPUT.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            cls.columns = list(reader.fieldnames or [])
            cls.rows = list(reader)

    def check(self, mutate) -> list[str]:
        rows = [dict(row) for row in self.rows]
        columns = list(self.columns)
        mutate(rows, columns)
        directory = Path(tempfile.mkdtemp())
        try:
            path = directory / "output.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            return verify_output.verify(path, DATASET)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_unmutated_copy_still_passes(self):
        self.assertEqual([], self.check(lambda rows, columns: None))

    def test_reordered_columns_are_caught(self):
        def mutate(rows, columns):
            columns[1], columns[2] = columns[2], columns[1]
        self.assertTrue(self.check(mutate))

    def test_dropped_row_is_caught(self):
        self.assertTrue(self.check(lambda rows, columns: rows.pop()))

    def test_reordered_rows_are_caught(self):
        self.assertTrue(self.check(lambda rows, columns: rows.reverse()))

    def test_amount_above_requested_is_caught(self):
        def mutate(rows, columns):
            rows[0]["amount_safe_to_pay"] = "999999999999"
        self.assertTrue(self.check(mutate))

    def test_negative_amount_is_caught(self):
        def mutate(rows, columns):
            rows[0]["amount_safe_to_pay"] = "-1"
        self.assertTrue(self.check(mutate))

    def test_unknown_status_is_caught(self):
        def mutate(rows, columns):
            rows[0]["affordability_status"] = "probably_fine"
        self.assertTrue(self.check(mutate))

    def test_empty_explanation_is_caught(self):
        def mutate(rows, columns):
            rows[0]["decision_explanation"] = "   "
        self.assertTrue(self.check(mutate))

    def test_csv_unsafe_explanation_is_caught(self):
        def mutate(rows, columns):
            rows[0]["decision_explanation"] = "Pay now | then wait"
        self.assertTrue(self.check(mutate))

    def test_non_chronological_plan_is_caught(self):
        def mutate(rows, columns):
            for row in rows:
                if row["payment_plan"] != "none" and "|" in row["payment_plan"]:
                    row["payment_plan"] = "|".join(reversed(row["payment_plan"].split("|")))
                    return
            self.skipTest("no multi-payment plan in output")
        self.assertTrue(self.check(mutate))

    def test_affordable_now_with_wrong_earliest_date_is_caught(self):
        def mutate(rows, columns):
            for row in rows:
                if row["affordability_status"] == "affordable_now":
                    row["earliest_date_for_full_payment"] = "1999-01-01"
                    return
            self.skipTest("no affordable_now row in output")
        self.assertTrue(self.check(mutate))

    def test_too_many_spending_changes_is_caught(self):
        def mutate(rows, columns):
            rows[0]["spending_changes_needed"] = "stop:a|stop:b|stop:c|stop:d"
        self.assertTrue(self.check(mutate))


if __name__ == "__main__":
    unittest.main()
