from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from auto_skill.llm import ChatCompletionConfig
from auto_skill.mvp import PromptRunResult
from scripts.eval.run_heldout_eval import (
    append_checkpoint_row,
    has_non_success_rows,
    is_valid_overall_score,
    judge_with_parse_retry,
    load_compatible_resume_success_rows,
    load_resume_success_rows,
    load_skill_rows,
    mode_needs_skill,
    mode_skill,
    runtime_metadata,
    score_row_status,
    skill_index,
    successful_score_cells,
    summarize_rows,
    write_empty_eval_result,
)


class SequencedCompletionClient:
    def __init__(self, texts: list[str], finish_reasons: list[str] | None = None) -> None:
        self.texts = list(texts)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(texts))

    def complete(self, *_args, **_kwargs):
        text = self.texts.pop(0)
        finish_reason = self.finish_reasons.pop(0)
        return type(
            "Completion",
            (),
            {
                "text": text,
                "model": "judge",
                "usage": {"total_tokens": 1},
                "finish_reason": finish_reason,
                "request_id": "req",
            },
        )()


class HeldoutEvalTests(unittest.TestCase):
    def test_auto_skill_does_not_fallback_to_no_validation(self) -> None:
        skills = {("pack-1", "auto_skill_feature_driven_no_validation"): "Skill"}

        self.assertIsNone(mode_skill("auto_skill", skills, "pack-1"))
        self.assertEqual(mode_skill("ours_no_validation", skills, "pack-1"), "Skill")

    def test_examples_plus_modes_select_expected_skill_artifact(self) -> None:
        skills = {
            ("pack-1", "one_shot_skill_from_examples"): "One-shot Skill",
            ("pack-1", "auto_skill_feature_driven_no_validation"): "Feature Skill",
        }

        self.assertEqual(
            mode_skill("examples_plus_one_shot_skill", skills, "pack-1"),
            "One-shot Skill",
        )
        self.assertEqual(
            mode_skill("examples_plus_feature_skill", skills, "pack-1"),
            "Feature Skill",
        )
        self.assertEqual(
            mode_skill("slide_constrained_examples_plus_feature_skill", skills, "pack-1"),
            "Feature Skill",
        )
        self.assertEqual(
            mode_skill("layout_plan_examples_plus_feature_skill", skills, "pack-1"),
            "Feature Skill",
        )

    def test_feature_signature_modes_select_compact_feature_context(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "mode": "auto_skill_feature_driven_no_validation",
                "skill_md": "Full Skill",
                "feature_reports": [
                    {
                        "content_features": {"key_variables": ["temperature"]},
                        "structure_features": {"sections": ["Methods"]},
                    }
                ],
                "cross_example_report": {
                    "stable_features": ["Use numbered sections"],
                    "candidate_rules": [{"rule": "Keep actions concrete", "support_count": 3}],
                },
            }
        ]
        skills = skill_index(rows)

        self.assertIn(
            "Abstract Feature Signatures",
            mode_skill("feature_signatures_only", skills, "pack-1") or "",
        )
        self.assertIn(
            "Keep actions concrete",
            mode_skill("examples_plus_feature_signatures", skills, "pack-1") or "",
        )
        self.assertIn(
            "Keep actions concrete",
            mode_skill("task_first_feature_signatures", skills, "pack-1") or "",
        )
        self.assertIn(
            "Task-Grounded Operational Anchors",
            mode_skill("task_first_operational_anchors", skills, "pack-1") or "",
        )

    def test_feature_signature_modes_require_context(self) -> None:
        self.assertTrue(mode_needs_skill("feature_signatures_only"))
        self.assertTrue(mode_needs_skill("examples_plus_feature_signatures"))
        self.assertTrue(mode_needs_skill("task_first_feature_signatures"))
        self.assertTrue(mode_needs_skill("task_first_operational_anchors"))
        self.assertFalse(mode_needs_skill("prompt_only"))
        self.assertFalse(mode_needs_skill("few_shot_examples_only"))

    def test_load_skill_rows_merges_repeated_skill_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "mvp.jsonl"
            second = Path(tmp) / "ours_full.jsonl"
            first.write_text(
                '{"pack_id":"pack-1","mode":"one_shot_skill_from_examples",'
                '"skill_md":"one shot"}\n',
                encoding="utf-8",
            )
            second.write_text(
                '{"pack_id":"pack-1","mode":"auto_skill_ours_full",'
                '"skill_md":"ours full"}\n',
                encoding="utf-8",
            )

            rows = load_skill_rows([first, second])

        self.assertEqual(
            [(row["pack_id"], row["mode"]) for row in rows],
            [
                ("pack-1", "one_shot_skill_from_examples"),
                ("pack-1", "auto_skill_ours_full"),
            ],
        )

    def test_score_row_status_rejects_truncated_generation(self) -> None:
        status = score_row_status(
            generation=PromptRunResult(text="partial", finish_reason="length"),
            judge=PromptRunResult(text='{"overall_score": 8}', finish_reason="stop"),
            judge_report={"overall_score": 8},
            overall_score=8.0,
        )

        self.assertEqual(status, "generation_incomplete")

    def test_score_row_status_rejects_truncated_judge(self) -> None:
        status = score_row_status(
            generation=PromptRunResult(text="answer", finish_reason="stop"),
            judge=PromptRunResult(text='{"overall_score": 8', finish_reason="length"),
            judge_report={"parse_error": "partial"},
            overall_score=None,
        )

        self.assertEqual(status, "judge_incomplete")

    def test_score_row_status_marks_content_filter_as_refusal(self) -> None:
        status = score_row_status(
            generation=PromptRunResult(text="answer", finish_reason="stop"),
            judge=PromptRunResult(text="blocked", finish_reason="content_filter"),
            judge_report={"parse_error": "no_json_object_found"},
            overall_score=None,
        )

        self.assertEqual(status, "judge_refusal")

    def test_score_row_status_requires_parseable_score(self) -> None:
        status = score_row_status(
            generation=PromptRunResult(text="answer", finish_reason="stop"),
            judge=PromptRunResult(text="not json", finish_reason="stop"),
            judge_report={"parse_error": "no_json_object_found"},
            overall_score=None,
        )

        self.assertEqual(status, "judge_parse_error")

    def test_judge_with_parse_retry_recovers_from_bad_json(self) -> None:
        judge, report, score, attempts = judge_with_parse_retry(
            judge_client=SequencedCompletionClient(["not json", '{"overall_score": 8}']),
            judge_prompt="score this",
            generation=PromptRunResult(text="answer", finish_reason="stop"),
            judge_max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(judge.text, '{"overall_score": 8}')
        self.assertEqual(report, {"overall_score": 8})
        self.assertEqual(score, 8.0)
        self.assertEqual(
            [attempt["status"] for attempt in attempts],
            ["judge_parse_error", "success"],
        )

    def test_judge_with_parse_retry_does_not_retry_refusal(self) -> None:
        judge, report, score, attempts = judge_with_parse_retry(
            judge_client=SequencedCompletionClient(
                [
                    "The request was rejected because it was considered high risk",
                    '{"overall_score": 8}',
                ],
                finish_reasons=["content_filter", "stop"],
            ),
            judge_prompt="score this",
            generation=PromptRunResult(text="answer", finish_reason="stop"),
            judge_max_tokens=128,
            parse_max_attempts=3,
        )

        self.assertEqual(judge.finish_reason, "content_filter")
        self.assertIn("parse_error", report)
        self.assertIsNone(score)
        self.assertEqual([attempt["status"] for attempt in attempts], ["judge_refusal"])

    def test_score_row_status_rejects_out_of_range_score(self) -> None:
        status = score_row_status(
            generation=PromptRunResult(text="answer", finish_reason="stop"),
            judge=PromptRunResult(text='{"overall_score": 99}', finish_reason="stop"),
            judge_report={"overall_score": 99},
            overall_score=99.0,
        )

        self.assertEqual(status, "judge_invalid_score")
        self.assertFalse(is_valid_overall_score(float("nan")))
        self.assertFalse(is_valid_overall_score(0.0))
        self.assertTrue(is_valid_overall_score(10.0))

    def test_summary_includes_status_counts_and_detects_failures(self) -> None:
        rows = [
            {"mode": "prompt_only", "status": "success", "overall_score": 8},
            {"mode": "auto_skill", "status": "missing_ours_full_skill"},
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["status_counts"], {"missing_ours_full_skill": 1, "success": 1})
        self.assertEqual(summary["modes"]["prompt_only"]["mean_overall_score"], 8)
        self.assertTrue(has_non_success_rows(rows))

    def test_write_empty_eval_result(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            summary_out = Path(tmp) / "summary.json"

            write_empty_eval_result(out, summary_out)

            self.assertEqual(out.read_text(), "")
            self.assertEqual(summarize_rows([]), __import__("json").loads(summary_out.read_text()))

    def test_resume_checkpoint_keeps_only_expected_success_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            rows = []
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                    "overall_score": 8,
                },
            )
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "auto_skill",
                    "status": "judge_parse_error",
                    "overall_score": None,
                },
            )
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "other",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                    "overall_score": 9,
                },
            )

            resumed = load_resume_success_rows(out, [("pack", "task", "prompt_only")])

            self.assertEqual(len(resumed), 1)
            self.assertEqual(successful_score_cells(resumed), {("pack", "task", "prompt_only")})

    def test_compatible_resume_requires_prompt_runtime_metadata(self) -> None:
        metadata = runtime_metadata(
            config=ChatCompletionConfig(
                base_url="https://example.test/v1",
                api_key="key",
                model="solver",
                enable_thinking=False,
            ),
            judge_config=None,
            temperature=0.2,
            max_tokens=4096,
            judge_max_tokens=1024,
            max_material_chars=4000,
        )
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            rows = []
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                    "overall_score": 8,
                    "judge_model": "solver",
                    **metadata,
                },
            )
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "old",
                    "mode": "prompt_only",
                    "status": "success",
                    "overall_score": 8,
                    "judge_model": "solver",
                },
            )

            resumed = load_compatible_resume_success_rows(
                out,
                [("pack", "task", "prompt_only"), ("pack", "old", "prompt_only")],
                expected_judge_model="solver",
                metadata=metadata,
            )

            self.assertEqual(successful_score_cells(resumed), {("pack", "task", "prompt_only")})


if __name__ == "__main__":
    unittest.main()
