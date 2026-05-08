from __future__ import annotations

import unittest

from auto_skill.eval_summary import expected_score_cells, summarize_score_rows


class EvalSummaryTests(unittest.TestCase):
    def test_expected_score_cells_respects_heldout_limit(self) -> None:
        packs = [
            {
                "pack_id": "pack-1",
                "heldout_tasks": [{"task_id": "task-1"}, {"task_id": "task-2"}],
            }
        ]

        self.assertEqual(
            expected_score_cells(packs, ["prompt_only", "auto_skill"], limit_heldout=1),
            [
                ("pack-1", "task-1", "prompt_only"),
                ("pack-1", "task-1", "auto_skill"),
            ],
        )

    def test_summary_reports_missing_cells_and_pairs(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 5,
            },
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "auto_skill",
                "status": "missing_ours_full_skill",
                "overall_score": None,
            },
        ]

        summary = summarize_score_rows(
            rows,
            evaluator_kind="test",
            score_summary_key="mean_score",
            expected_cells=[
                ("pack-1", "task-1", "prompt_only"),
                ("pack-1", "task-1", "auto_skill"),
                ("pack-1", "task-2", "prompt_only"),
                ("pack-1", "task-2", "auto_skill"),
            ],
        )

        self.assertFalse(summary["coverage"]["complete"])
        self.assertEqual(len(summary["coverage"]["missing_cells"]), 2)
        self.assertEqual(len(summary["coverage"]["non_success_cells"]), 1)
        self.assertEqual(summary["paired_deltas"]["auto_skill"]["expected_pairs"], 2)
        self.assertEqual(len(summary["paired_deltas"]["auto_skill"]["missing_pairs"]), 2)

    def test_summary_does_not_treat_boolean_as_score(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": True,
            }
        ]

        summary = summarize_score_rows(
            rows,
            evaluator_kind="test",
            score_summary_key="mean_score",
            expected_cells=[("pack-1", "task-1", "prompt_only")],
        )

        self.assertFalse(summary["coverage"]["complete"])
        self.assertEqual(summary["coverage"]["successful_score_cells"], 0)
        self.assertNotIn("prompt_only", summary["modes"])


if __name__ == "__main__":
    unittest.main()
