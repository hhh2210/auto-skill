from __future__ import annotations

import unittest

from scripts.eval.audit_train_examples import (
    desired_output_text,
    summarize_rows,
    train_private_index,
)


class TrainExampleAuditTests(unittest.TestCase):
    def test_train_private_index_uses_train_private_rows(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "train_private": [{"task_ref": "train-1", "supervision": {"items": []}}],
                "heldout_private": [{"task_ref": "heldout-1"}],
            }
        ]

        index = train_private_index(rows)

        self.assertIn(("pack-1", "train-1"), index)
        self.assertNotIn(("pack-1", "heldout-1"), index)

    def test_desired_output_text_requires_generated_status(self) -> None:
        text, status = desired_output_text({"desired_output": {"status": "needs_generation"}})

        self.assertIsNone(text)
        self.assertEqual(status, "desired_output_not_generated")

    def test_desired_output_text_returns_generated_text(self) -> None:
        text, status = desired_output_text(
            {"desired_output": {"status": "generated", "text": "final output"}}
        )

        self.assertEqual(text, "final output")
        self.assertIsNone(status)

    def test_summary_counts_status_by_source(self) -> None:
        summary = summarize_rows(
            [
                {
                    "pack_id": "pack",
                    "task_id": "train-1",
                    "source": "WritingBench",
                    "mode": "desired_output",
                    "status": "success",
                    "overall_score": 8,
                },
                {
                    "pack_id": "pack",
                    "task_id": "train-2",
                    "source": "PresentBench",
                    "mode": "desired_output",
                    "status": "desired_output_not_generated",
                    "overall_score": None,
                },
            ]
        )

        self.assertEqual(summary["modes"]["desired_output"]["count"], 1)
        self.assertEqual(
            summary["status_counts_by_source"],
            {
                "PresentBench": {"desired_output_not_generated": 1},
                "WritingBench": {"success": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
