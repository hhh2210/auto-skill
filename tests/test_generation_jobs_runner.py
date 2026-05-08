from __future__ import annotations

import unittest

from scripts.data.run_generation_jobs import count_statuses, has_failed_generation_rows


class GenerationJobsRunnerTests(unittest.TestCase):
    def test_status_counts_and_failed_rows(self) -> None:
        rows = [
            {"status": "success"},
            {"status": "error"},
            {"status": "rejected_incomplete_generation"},
        ]

        self.assertEqual(
            count_statuses(rows),
            {"error": 1, "rejected_incomplete_generation": 1, "success": 1},
        )
        self.assertTrue(has_failed_generation_rows(rows))

    def test_success_rows_are_not_failures(self) -> None:
        self.assertFalse(has_failed_generation_rows([{"status": "success"}]))


if __name__ == "__main__":
    unittest.main()
