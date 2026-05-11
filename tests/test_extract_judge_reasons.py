from __future__ import annotations

import unittest

from auto_skill.judge_reason_extraction import extract_losing_task_judge_reasons


def _row(
    *,
    pack_id: str = "pack",
    task_id: str = "task",
    mode: str,
    scores: dict[str, int],
    status: str = "success",
) -> dict[str, object]:
    return {
        "pack_id": pack_id,
        "task_id": task_id,
        "mode": mode,
        "status": status,
        "source_task_id": "source-1",
        "judge_model": "mimo-v2.5-pro",
        "solver_model": "qwen3.5-plus",
        "scores": {
            name: [{"score": score, "reason": f"{mode} reason for {name}"}]
            for name, score in scores.items()
        },
    }


class JudgeReasonExtractionTests(unittest.TestCase):
    def test_extracts_losing_task_with_paired_criteria_and_deltas(self) -> None:
        rows = [
            _row(mode="few_shot_examples_only", scores={"A": 8, "B": 7}),
            _row(mode="ours_no_validation", scores={"A": 6, "B": 7}),
        ]
        result = extract_losing_task_judge_reasons(rows)

        self.assertEqual(result.loss_count, 1)
        self.assertEqual(result.win_count, 0)
        self.assertEqual(len(result.rows), 1)
        row = result.rows[0]
        self.assertEqual(row["schema_version"], "judge-reason-pairs/v1")
        self.assertEqual(row["pack_id"], "pack")
        self.assertEqual(row["mode"], "ours_no_validation")
        self.assertAlmostEqual(row["overall_delta"], -1.0)
        self.assertEqual(row["worst_criterion"]["criterion_name"], "A")
        criteria = row["criteria"]
        self.assertEqual([item["criterion_name"] for item in criteria], ["A", "B"])
        self.assertEqual(criteria[0]["baseline"]["score"], 8.0)
        self.assertEqual(criteria[0]["candidate"]["score"], 6.0)
        self.assertEqual(criteria[0]["delta"], -2.0)
        self.assertIn("few_shot_examples_only reason", criteria[0]["baseline"]["reason"])
        self.assertIn("ours_no_validation reason", criteria[0]["candidate"]["reason"])

    def test_non_losing_pairs_are_counted_but_not_output_by_default(self) -> None:
        rows = [
            _row(mode="few_shot_examples_only", scores={"A": 7, "B": 7}),
            _row(mode="ours_no_validation", scores={"A": 8, "B": 8}),
        ]
        result = extract_losing_task_judge_reasons(rows)

        self.assertEqual(result.loss_count, 0)
        self.assertEqual(result.win_count, 1)
        self.assertEqual(result.rows, ())

    def test_reports_missing_expected_mode_without_fabricating_row(self) -> None:
        rows = [_row(mode="few_shot_examples_only", scores={"A": 8, "B": 7})]
        result = extract_losing_task_judge_reasons(
            rows,
            expected_task_keys=[("pack", "task")],
        )

        self.assertEqual(result.rows, ())
        self.assertEqual(len(result.missing_cells), 1)
        missing = result.missing_cells[0]
        self.assertEqual(missing.pack_id, "pack")
        self.assertEqual(missing.task_id, "task")
        self.assertEqual(missing.missing_modes, ("ours_no_validation",))

    def test_reports_mismatched_criterion_sets(self) -> None:
        rows = [
            _row(mode="few_shot_examples_only", scores={"A": 8, "B": 7}),
            _row(mode="ours_no_validation", scores={"A": 6, "C": 5}),
        ]
        result = extract_losing_task_judge_reasons(rows)

        self.assertEqual(result.rows, ())
        self.assertEqual(len(result.mismatched_criteria), 1)
        mismatch = result.mismatched_criteria[0]
        self.assertEqual(mismatch.baseline_only, ("B",))
        self.assertEqual(mismatch.mode_only, ("C",))

    def test_duplicate_success_cells_are_reported_and_excluded(self) -> None:
        rows = [
            _row(mode="few_shot_examples_only", scores={"A": 8, "B": 7}),
            _row(mode="few_shot_examples_only", scores={"A": 8, "B": 7}),
            _row(mode="ours_no_validation", scores={"A": 6, "B": 6}),
        ]
        result = extract_losing_task_judge_reasons(rows)

        self.assertEqual(result.rows, ())
        self.assertEqual(len(result.duplicate_cells), 1)
        duplicate = result.duplicate_cells[0]
        self.assertEqual(duplicate.mode, "few_shot_examples_only")
        self.assertEqual(duplicate.count, 2)
        self.assertEqual(result.missing_cells[0].missing_modes, ("few_shot_examples_only",))


if __name__ == "__main__":
    unittest.main()
