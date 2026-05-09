from __future__ import annotations

import unittest
from pathlib import Path

from scripts.metrics.summarize_mvp_metrics import (
    expected_cells_for_eval,
    metrics_model_inventories,
    row_judge_model,
    summarize_cross_eval_groups,
)


class SummarizeMvpMetricsTests(unittest.TestCase):
    def test_row_judge_model_falls_back_to_nested_judge_model(self) -> None:
        row = {"judge": {"model": "qwen3.5-plus"}}

        self.assertEqual(row_judge_model(row), "qwen3.5-plus")

    def test_expected_cells_use_observed_modes_for_split_eval_files(self) -> None:
        packs = [
            {
                "pack_id": "writingbench_pack",
                "source": "WritingBench",
                "heldout_tasks": [
                    {"task_id": "task-1"},
                    {"task_id": "task-2"},
                ],
            }
        ]
        rows = [
            {
                "pack_id": "writingbench_pack",
                "task_id": "task-1",
                "mode": "examples_plus_feature_skill",
                "evaluator_kind": "writingbench_official_prompt_qwen_judge",
            }
        ]

        cells = expected_cells_for_eval(
            path=Path("runs/writingbench_ablation.jsonl"),
            rows=rows,
            packs=packs,
            modes=[
                "prompt_only",
                "few_shot_examples_only",
                "examples_plus_feature_skill",
            ],
            limit_heldout=None,
        )

        self.assertEqual(
            cells,
            [
                ("writingbench_pack", "task-1", "examples_plus_feature_skill"),
                ("writingbench_pack", "task-2", "examples_plus_feature_skill"),
            ],
        )

    def test_cross_eval_summary_pairs_split_baseline_and_ablation_files(self) -> None:
        packs = [
            {
                "pack_id": "writingbench_pack",
                "source": "WritingBench",
                "heldout_tasks": [{"task_id": "task-1"}],
            }
        ]
        baseline_rows = [
            {
                "pack_id": "writingbench_pack",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 7,
                "evaluator_kind": "writingbench_official_prompt_qwen_judge",
                "judge_model": "qwen3.5-plus",
            }
        ]
        ablation_rows = [
            {
                "pack_id": "writingbench_pack",
                "task_id": "task-1",
                "mode": "examples_plus_feature_skill",
                "status": "success",
                "overall_score": 9,
                "evaluator_kind": "writingbench_official_prompt_qwen_judge",
                "judge_model": "qwen3.5-plus",
            }
        ]

        summaries = summarize_cross_eval_groups(
            [
                (Path("runs/writingbench_baseline.jsonl"), baseline_rows),
                (Path("runs/writingbench_ablation.jsonl"), ablation_rows),
            ],
            packs=packs,
            modes=["prompt_only", "examples_plus_feature_skill"],
            baseline_mode="prompt_only",
            limit_heldout=None,
        )

        self.assertEqual(len(summaries), 1)
        paired = summaries[0]["summary"]["paired_deltas"]["examples_plus_feature_skill"]
        self.assertEqual(paired["count"], 1)
        self.assertEqual(paired["expected_pairs"], 1)
        self.assertEqual(paired["mean_delta"], 2)

    def test_model_inventory_keeps_diagnostics_out_of_paper_facing_inventory(
        self,
    ) -> None:
        inventories = metrics_model_inventories(
            skill_rows=[{"solver_model": "qwen3.5-plus"}],
            eval_rows=[
                {
                    "status": "success",
                    "solver_model": "qwen3.5-plus",
                    "judge_model": "mimo-v2.5-pro",
                }
            ],
            diagnostic_rows=[
                {
                    "status": "success",
                    "signature_generation": {"model": "qwen3.5-plus"},
                    "judge": {"model": "qwen3.5-plus"},
                }
            ],
        )

        self.assertEqual(
            inventories["heldout_eval"]["scored"]["eval_judge_models"],
            ["mimo-v2.5-pro"],
        )
        self.assertEqual(
            inventories["with_diagnostics"]["scored"]["eval_judge_models"],
            ["mimo-v2.5-pro", "qwen3.5-plus"],
        )
        self.assertEqual(
            inventories["heldout_eval"]["scored"]["successful_eval_rows"],
            1,
        )
        self.assertEqual(
            inventories["with_diagnostics"]["scored"]["successful_eval_rows"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
