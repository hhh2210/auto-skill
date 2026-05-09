from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.eval.audit_train_examples import (
    chat_result_json,
    desired_output_text,
    generic_audit_with_parse_retry,
    generic_judge_status,
    main,
    summarize_rows,
    train_private_index,
)


class SequencedCompletionClient:
    def __init__(self, texts: list[str], finish_reasons: list[str] | None = None) -> None:
        self.texts = list(texts)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(texts))

    def complete(self, *_args, **_kwargs):
        text = self.texts.pop(0)
        finish_reason = self.finish_reasons.pop(0)
        return SimpleNamespace(
            text=text,
            model="judge",
            finish_reason=finish_reason,
            usage={"total_tokens": 1},
            request_id="req",
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

    def test_generic_judge_status_fails_closed(self) -> None:
        self.assertEqual(
            generic_judge_status(
                finish_reason="stop",
                judge_report={"overall_score": 8},
                overall_score=8,
            ),
            "success",
        )
        self.assertEqual(
            generic_judge_status(
                finish_reason="length",
                judge_report={"overall_score": 8},
                overall_score=8,
            ),
            "judge_incomplete",
        )
        self.assertEqual(
            generic_judge_status(
                finish_reason="content_filter",
                judge_report={"overall_score": 8},
                overall_score=8,
            ),
            "judge_refusal",
        )
        self.assertEqual(
            generic_judge_status(
                finish_reason="stop",
                judge_report={"parse_error": "bad"},
                overall_score=None,
            ),
            "judge_parse_error",
        )
        self.assertEqual(
            generic_judge_status(
                finish_reason="stop",
                judge_report={"overall_score": True},
                overall_score=True,
            ),
            "judge_invalid_score",
        )

    def test_chat_result_json_serializes_completion_result_shape(self) -> None:
        result = SimpleNamespace(
            text="{}",
            model="judge",
            finish_reason="stop",
            usage={"total_tokens": 1},
            request_id="req",
        )

        self.assertEqual(
            chat_result_json(result),
            {
                "text": "{}",
                "model": "judge",
                "finish_reason": "stop",
                "usage": {"total_tokens": 1},
                "request_id": "req",
            },
        )

    def test_generic_audit_retries_parse_error(self) -> None:
        judge, report, score, status, calls = generic_audit_with_parse_retry(
            judge_client=SequencedCompletionClient(["bad", '{"overall_score": 7}']),
            judge_prompt="score",
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(judge.text, '{"overall_score": 7}')
        self.assertEqual(report, {"overall_score": 7})
        self.assertEqual(score, 7.0)
        self.assertEqual(status, "success")
        self.assertEqual([call["status"] for call in calls], ["judge_parse_error", "success"])

    def test_generic_audit_records_refusal_without_retry(self) -> None:
        judge, report, score, status, calls = generic_audit_with_parse_retry(
            judge_client=SequencedCompletionClient(
                ["blocked", '{"overall_score": 7}'],
                finish_reasons=["safety", "stop"],
            ),
            judge_prompt="score",
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(judge.text, "blocked")
        self.assertEqual(status, "judge_refusal")
        self.assertIn("parse_error", report)
        self.assertIsNone(score)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["finish_reason"], "safety")
        self.assertEqual(calls[0]["status"], "judge_refusal")

    def test_cli_rejects_invalid_parse_attempts_before_dry_run(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["audit_train_examples.py", "--parse-max-attempts", "0", "--dry-run"],
        ):
            self.assertEqual(main(), 2)

    def test_summary_accepts_presentbench_audit_success(self) -> None:
        summary = summarize_rows(
            [
                {
                    "schema_version": "train-example-quality-audit/v1",
                    "pack_id": "pack",
                    "task_id": "train-1",
                    "source": "PresentBench",
                    "mode": "desired_output",
                    "evaluator_kind": "private_train_example_quality_audit",
                    "status": "success",
                    "overall_score": 7.5,
                }
            ]
        )

        self.assertEqual(
            summary["modes"]["desired_output"]["mean_train_example_quality_score"],
            7.5,
        )


if __name__ == "__main__":
    unittest.main()
