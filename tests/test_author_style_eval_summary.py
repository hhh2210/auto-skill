from __future__ import annotations

import unittest

from auto_skill.author_style_eval_summary import summarize_author_style_eval_rows


class AuthorStyleEvalSummaryTests(unittest.TestCase):
    def test_summarizes_modes_and_cross_file_paired_deltas(self) -> None:
        rows = [
            {
                "pack_id": "p1",
                "task_id": "t1",
                "mode": "few_shot_examples_only",
                "status": "success",
                "style_likeness_1_to_10": 7,
                "candidate_beats_negatives": 3,
                "hard_negative_count": 4,
            },
            {
                "pack_id": "p1",
                "task_id": "t1",
                "mode": "prompt_only",
                "status": "success",
                "style_likeness_1_to_10": 3,
                "candidate_beats_negatives": 0,
                "hard_negative_count": 4,
            },
            {
                "pack_id": "p1",
                "task_id": "t1",
                "mode": "one_shot_skill_from_examples",
                "status": "success",
                "judge_report": {
                    "style_likeness_1_to_10": 8,
                    "candidate_beats_negatives": 4,
                },
                "hard_negative_count": 4,
            },
        ]

        summary = summarize_author_style_eval_rows(rows)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["modes"]["few_shot_examples_only"]["mean_style_likeness"], 7)
        self.assertEqual(summary["modes"]["prompt_only"]["hard_negative_win_rate"], 0.0)
        self.assertEqual(
            summary["paired_deltas"]["prompt_only"]["mean_style_likeness_delta"],
            -4,
        )
        self.assertEqual(
            summary["paired_deltas"]["one_shot_skill_from_examples"][
                "mean_style_likeness_delta"
            ],
            1,
        )
        self.assertEqual(
            summary["pack_macro_paired_deltas"]["one_shot_skill_from_examples"][
                "mean_pack_style_likeness_delta"
            ],
            1,
        )

    def test_pack_macro_deltas_average_tasks_before_packs(self) -> None:
        rows = [
            {
                "pack_id": "p1",
                "task_id": "t1",
                "mode": "few_shot_examples_only",
                "status": "success",
                "style_likeness_1_to_10": 8,
                "candidate_beats_negatives": 4,
                "hard_negative_count": 4,
            },
            {
                "pack_id": "p1",
                "task_id": "t1",
                "mode": "candidate",
                "status": "success",
                "style_likeness_1_to_10": 6,
                "candidate_beats_negatives": 2,
                "hard_negative_count": 4,
            },
            {
                "pack_id": "p1",
                "task_id": "t2",
                "mode": "few_shot_examples_only",
                "status": "success",
                "style_likeness_1_to_10": 8,
                "candidate_beats_negatives": 4,
                "hard_negative_count": 4,
            },
            {
                "pack_id": "p1",
                "task_id": "t2",
                "mode": "candidate",
                "status": "success",
                "style_likeness_1_to_10": 8,
                "candidate_beats_negatives": 4,
                "hard_negative_count": 4,
            },
            {
                "pack_id": "p2",
                "task_id": "t1",
                "mode": "few_shot_examples_only",
                "status": "success",
                "style_likeness_1_to_10": 5,
                "candidate_beats_negatives": 2,
                "hard_negative_count": 4,
            },
            {
                "pack_id": "p2",
                "task_id": "t1",
                "mode": "candidate",
                "status": "success",
                "style_likeness_1_to_10": 6,
                "candidate_beats_negatives": 3,
                "hard_negative_count": 4,
            },
        ]

        summary = summarize_author_style_eval_rows(rows)

        macro = summary["pack_macro_paired_deltas"]["candidate"]
        self.assertEqual(macro["paired_packs"], 2)
        self.assertEqual(macro["mean_pack_style_likeness_delta"], 0)
        self.assertEqual(macro["mean_pack_hard_negative_win_rate_delta"], 0)
        self.assertEqual(macro["pack_style_delta_sign_counts"], {"negative": 1, "positive": 1})


if __name__ == "__main__":
    unittest.main()
