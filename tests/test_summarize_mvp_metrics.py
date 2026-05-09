from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.metrics.summarize_mvp_metrics import (
    duplicate_score_cells,
    expected_cells_for_eval,
    main,
    metrics_model_inventories,
    row_judge_model,
    summarize_cross_eval_groups,
)


class SummarizeMvpMetricsTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_row_judge_model_falls_back_to_nested_judge_model(self) -> None:
        row = {"judge": {"model": "qwen3.5-plus"}}

        self.assertEqual(row_judge_model(row), "qwen3.5-plus")

    def test_expected_cells_do_not_mask_whole_missing_modes(self) -> None:
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
                ("writingbench_pack", "task-1", "prompt_only"),
                ("writingbench_pack", "task-1", "few_shot_examples_only"),
                ("writingbench_pack", "task-1", "examples_plus_feature_skill"),
                ("writingbench_pack", "task-2", "prompt_only"),
                ("writingbench_pack", "task-2", "few_shot_examples_only"),
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

    def test_cross_eval_summary_rejects_duplicate_cells(self) -> None:
        packs = [
            {
                "pack_id": "writingbench_pack",
                "source": "WritingBench",
                "heldout_tasks": [{"task_id": "task-1"}],
            }
        ]
        duplicate_rows = [
            {
                "pack_id": "writingbench_pack",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 7,
                "evaluator_kind": "writingbench_official_prompt_qwen_judge",
                "judge_model": "qwen3.5-plus",
            },
            {
                "pack_id": "writingbench_pack",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 8,
                "evaluator_kind": "writingbench_official_prompt_qwen_judge",
                "judge_model": "qwen3.5-plus",
            },
        ]

        self.assertEqual(
            duplicate_score_cells(duplicate_rows),
            [("writingbench_pack", "task-1", "prompt_only")],
        )
        with self.assertRaisesRegex(ValueError, "duplicate score cells"):
            summarize_cross_eval_groups(
                [(Path("runs/writingbench_baseline.jsonl"), duplicate_rows)],
                packs=packs,
                modes=["prompt_only"],
                baseline_mode="prompt_only",
                limit_heldout=None,
            )

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

    def test_cli_writes_separate_eval_and_diagnostic_model_inventories(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs_path = root / "packs.jsonl"
            skills_path = root / "skills.jsonl"
            eval_path = root / "eval.jsonl"
            self_consistency_path = root / "self_consistency.jsonl"
            out_path = root / "summary.json"

            self.write_jsonl(
                packs_path,
                [
                    {
                        "pack_id": "pack-1",
                        "source": "WritingBench",
                        "train_examples": [
                            {
                                "example_id": "e1",
                                "task_input": "task",
                                "desired_output": {"status": "generated", "text": "out"},
                            }
                        ],
                        "heldout_tasks": [{"task_id": "task-1", "task_input": "heldout"}],
                    }
                ],
            )
            self.write_jsonl(
                skills_path,
                [
                    {
                        "schema_version": "skill-induction/v1",
                        "status": "success",
                        "pack_id": "pack-1",
                        "mode": "one_shot_skill_from_examples",
                        "skill_md": "# Skill",
                        "solver_model": "qwen3.5-plus",
                        "model_calls": [],
                    },
                    {
                        "schema_version": "skill-induction/v1",
                        "status": "success",
                        "pack_id": "pack-1",
                        "mode": "auto_skill_feature_driven_no_validation",
                        "skill_md": "# Skill",
                        "solver_model": "qwen3.5-plus",
                        "model_calls": [],
                    },
                ],
            )
            self.write_jsonl(
                eval_path,
                [
                    {
                        "schema_version": "heldout-eval/v1",
                        "status": "success",
                        "pack_id": "pack-1",
                        "task_id": "task-1",
                        "mode": "prompt_only",
                        "evaluator_kind": "writingbench_official_prompt_qwen_judge",
                        "overall_score": 7,
                        "solver_model": "qwen3.5-plus",
                        "judge_model": "mimo-v2.5-pro",
                    }
                ],
            )
            self.write_jsonl(
                self_consistency_path,
                [
                    {
                        "schema_version": "skill-self-consistency/v1",
                        "diagnostic_kind": "skill_encoding_self_consistency",
                        "status": "success",
                        "pack_id": "pack-1",
                        "mode": "one_shot_skill_from_examples",
                        "signature_generation": {"model": "qwen3.5-plus"},
                        "judge": {"model": "qwen3.5-plus"},
                    }
                ],
            )

            argv = [
                "summarize_mvp_metrics.py",
                "--skills",
                str(skills_path),
                "--packs",
                str(packs_path),
                "--modes",
                "prompt_only",
                "--eval",
                str(eval_path),
                "--self-consistency",
                str(self_consistency_path),
                "--out",
                str(out_path),
            ]
            with patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(), 0)

            summary = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["model_inventory"]["scored"]["eval_judge_models"],
                ["mimo-v2.5-pro"],
            )
            self.assertEqual(
                summary["diagnostic_model_inventory"]["scored"]["eval_judge_models"],
                ["mimo-v2.5-pro", "qwen3.5-plus"],
            )


if __name__ == "__main__":
    unittest.main()
