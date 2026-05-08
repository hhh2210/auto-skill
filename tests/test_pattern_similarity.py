from __future__ import annotations

import unittest
from types import SimpleNamespace

from auto_skill.pattern_similarity import (
    SKILL_AWARE_EVALUATOR_KIND,
    parse_pattern_similarity_report,
    pattern_similarity_status,
    summarize_pattern_similarity_rows,
)
from scripts.metrics.run_pattern_similarity_eval import judge_with_parse_retry


class SequencedCompletionClient:
    def __init__(self, texts: list[str], finish_reasons: list[str] | None = None) -> None:
        self.texts = list(texts)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(texts))

    def complete(self, *_args, **_kwargs):
        return SimpleNamespace(
            text=self.texts.pop(0),
            model="judge",
            finish_reason=self.finish_reasons.pop(0),
            usage={"total_tokens": 1},
            request_id="req",
        )


class PatternSimilarityTests(unittest.TestCase):
    def test_parse_pattern_similarity_report_accepts_scores(self) -> None:
        report = parse_pattern_similarity_report(
            """{
              "pattern_similarity_score": 8,
              "structural_similarity_score": 7,
              "style_similarity_score": 9,
              "constraint_transfer_score": 8,
              "unsupported_pattern_risk": "low",
              "supported_patterns": ["numbered sections"],
              "missed_patterns": [],
              "unsupported_patterns": [],
              "rationale": "ok"
            }"""
        )

        self.assertEqual(report["pattern_similarity_score"], 8.0)
        self.assertNotIn("parse_error", report)
        self.assertEqual(pattern_similarity_status(finish_reason="stop", report=report), "success")

    def test_parse_pattern_similarity_report_rejects_bool_score(self) -> None:
        report = parse_pattern_similarity_report(
            """{
              "pattern_similarity_score": true,
              "structural_similarity_score": 7,
              "style_similarity_score": 9,
              "constraint_transfer_score": 8,
              "unsupported_pattern_risk": "low",
              "supported_patterns": [],
              "missed_patterns": [],
              "unsupported_patterns": []
            }"""
        )

        self.assertEqual(
            report["parse_error"],
            "pattern_similarity_score_must_be_number_1_to_10",
        )
        self.assertEqual(
            pattern_similarity_status(finish_reason="stop", report=report),
            "judge_parse_error",
        )

    def test_summary_uses_pattern_similarity_key(self) -> None:
        summary = summarize_pattern_similarity_rows(
            [
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                    "overall_score": 6,
                    "evaluator_kind": "qwen_example_pattern_similarity",
                }
            ]
        )

        self.assertEqual(summary["modes"]["prompt_only"]["mean_pattern_similarity_score"], 6)

    def test_summary_can_use_skill_aware_debug_evaluator_kind(self) -> None:
        summary = summarize_pattern_similarity_rows(
            [
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "auto_skill",
                    "status": "success",
                    "overall_score": 7,
                    "evaluator_kind": SKILL_AWARE_EVALUATOR_KIND,
                }
            ],
            evaluator_kind=SKILL_AWARE_EVALUATOR_KIND,
        )

        self.assertEqual(summary["evaluator_kind"], SKILL_AWARE_EVALUATOR_KIND)
        self.assertEqual(summary["modes"]["auto_skill"]["mean_pattern_similarity_score"], 7)

    def test_judge_with_parse_retry_recovers_from_bad_json(self) -> None:
        judge, report, status, attempts = judge_with_parse_retry(
            client=SequencedCompletionClient(
                [
                    "not json",
                    """{
                      "pattern_similarity_score": 8,
                      "structural_similarity_score": 7,
                      "style_similarity_score": 9,
                      "constraint_transfer_score": 8,
                      "unsupported_pattern_risk": "low",
                      "supported_patterns": [],
                      "missed_patterns": [],
                      "unsupported_patterns": []
                    }""",
                ]
            ),
            prompt="judge",
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(judge.finish_reason, "stop")
        self.assertEqual(report["pattern_similarity_score"], 8.0)
        self.assertEqual(status, "success")
        self.assertEqual(
            [attempt["status"] for attempt in attempts],
            ["judge_parse_error", "success"],
        )

    def test_judge_with_parse_retry_does_not_retry_incomplete_judge(self) -> None:
        judge, report, status, attempts = judge_with_parse_retry(
            client=SequencedCompletionClient(
                ["partial", '{"pattern_similarity_score": 8}'],
                finish_reasons=["length", "stop"],
            ),
            prompt="judge",
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(judge.finish_reason, "length")
        self.assertIn("parse_error", report)
        self.assertEqual(status, "judge_incomplete")
        self.assertEqual([attempt["status"] for attempt in attempts], ["judge_incomplete"])


if __name__ == "__main__":
    unittest.main()
