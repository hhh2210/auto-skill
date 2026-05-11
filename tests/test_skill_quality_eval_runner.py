from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_skill.example_packs import load_jsonl
from auto_skill.llm import ChatCompletionConfig
from scripts.metrics import run_skill_quality_eval


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class SkillQualityEvalRunnerTests(unittest.TestCase):
    def test_parse_quality_report_keeps_derived_average_out_of_report(self) -> None:
        report = run_skill_quality_eval.parse_quality_report(
            json.dumps(
                {
                    "scores": {
                        "faithfulness": {"score": 4, "reason": "ok"},
                        "reusability": {"score": 3, "reason": "ok"},
                        "effectiveness": {"score": 5, "reason": "ok"},
                        "clarity": {"score": 4, "reason": "ok"},
                        "conciseness": {"score": 4, "reason": "ok"},
                    },
                    "summary": "ok",
                }
            )
        )

        self.assertNotIn("parse_error", report)
        self.assertNotIn("overall_skill_quality", report)

    def test_summarize_uses_top_level_overall_skill_quality_only(self) -> None:
        rows = [
            {
                "mode": "skill",
                "status": "success",
                "skill_quality_report": {"overall_skill_quality": 1},
                "overall_skill_quality": 80.0,
            },
            {
                "mode": "skill",
                "status": "model_error",
                "overall_skill_quality": None,
            },
        ]

        summary = run_skill_quality_eval.summarize(rows)

        self.assertEqual(summary["status_counts"]["success"], 1)
        self.assertEqual(summary["status_counts"]["model_error"], 1)
        self.assertEqual(summary["by_mode"]["skill"]["mean_skill_quality"], 80.0)

    def test_main_checkpoints_model_error_when_evaluate_one_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packs = root / "packs.jsonl"
            skills = root / "skills.jsonl"
            out = root / "quality.jsonl"
            summary = root / "quality.summary.json"
            _write_jsonl(
                packs,
                [
                    {
                        "pack_id": "pack",
                        "train_examples": [
                            {
                                "example_id": "ex1",
                                "task_input": "Write a memo.",
                                "desired_output": {"text": "Memo text."},
                            }
                        ],
                    }
                ],
            )
            _write_jsonl(
                skills,
                [
                    {
                        "schema_version": "skill-induction/v1",
                        "status": "success",
                        "pack_id": "pack",
                        "mode": "skill",
                        "solver_model": "solver",
                        "skill_md": "# Skill",
                        "model_calls": [],
                    }
                ],
            )
            argv = [
                "run_skill_quality_eval.py",
                "--packs",
                str(packs),
                "--skills",
                str(skills),
                "--out",
                str(out),
                "--summary-out",
                str(summary),
                "--modes",
                "skill",
            ]
            config = ChatCompletionConfig(
                base_url="https://example.test/v1",
                api_key="key",
                model="judge-model",
            )
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    run_skill_quality_eval.ChatCompletionConfig,
                    "from_env",
                    return_value=config,
                ),
                patch.object(
                    run_skill_quality_eval,
                    "evaluate_one",
                    side_effect=RuntimeError("network down"),
                ),
            ):
                rc = run_skill_quality_eval.main()

            self.assertEqual(rc, 0)
            rows = load_jsonl(out)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "model_error")
            self.assertEqual(rows[0]["error"], "RuntimeError: network down")
            self.assertEqual(rows[0]["overall_skill_quality"], None)
            summary_data = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(summary_data["status_counts"]["model_error"], 1)


if __name__ == "__main__":
    unittest.main()
