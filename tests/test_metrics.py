from __future__ import annotations

import unittest
from types import SimpleNamespace

from auto_skill.metrics import (
    build_abstract_signatures_prompt,
    build_signature_consistency_judge_prompt,
    count_skill_section_rules,
    literal_leakage_report,
    parse_abstract_signatures,
    parse_self_consistency_report,
    score_and_negative_transfer_summary,
    self_consistency_summary,
    skill_artifact_metric,
    skill_artifact_summary,
    skill_induction_token_usage_summary,
    token_usage_summary,
)
from scripts.metrics.run_self_consistency_metric import (
    generate_signatures_with_parse_retry,
    judge_with_parse_retry,
)


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


class MetricsTests(unittest.TestCase):
    def test_score_summary_adds_negative_transfer_rate(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 8,
            },
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "auto_skill",
                "status": "success",
                "overall_score": 6,
            },
        ]

        summary = score_and_negative_transfer_summary(rows, evaluator_kind="test")

        self.assertEqual(summary["paired_deltas"]["auto_skill"]["mean_delta"], -2)
        self.assertEqual(
            summary["negative_transfer"]["auto_skill"]["negative_transfer_rate"],
            1.0,
        )
        self.assertEqual(
            summary["negative_transfer"]["auto_skill"]["negative_transfer_rate_denominator"],
            "paired_count",
        )
        self.assertEqual(summary["negative_transfer"]["auto_skill"]["expected_pair_count"], 1)
        self.assertEqual(summary["negative_transfer"]["auto_skill"]["missing_pair_count"], 0)
        self.assertEqual(
            summary["negative_transfer"]["auto_skill"][
                "negative_transfer_rate_on_expected_pairs"
            ],
            1.0,
        )

    def test_score_summary_reports_missing_expected_pairs(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 8,
            },
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "auto_skill",
                "status": "success",
                "overall_score": 9,
            },
        ]

        summary = score_and_negative_transfer_summary(
            rows,
            evaluator_kind="test",
            expected_cells=[
                ("pack-1", "task-1", "prompt_only"),
                ("pack-1", "task-1", "auto_skill"),
                ("pack-2", "task-2", "prompt_only"),
                ("pack-2", "task-2", "auto_skill"),
            ],
        )

        transfer = summary["negative_transfer"]["auto_skill"]
        self.assertEqual(transfer["expected_pair_count"], 2)
        self.assertEqual(transfer["paired_count"], 1)
        self.assertEqual(transfer["missing_pair_count"], 1)
        self.assertEqual(transfer["missing_pair_rate"], 0.5)
        self.assertFalse(summary["coverage"]["complete"])

    def test_skill_section_rule_counts(self) -> None:
        skill_md = """# Skill

## Required rules
- First
* Second

## Optional rules
1. Maybe

## Do not generalize
- Do not copy IDs
"""
        self.assertEqual(
            count_skill_section_rules(skill_md),
            {
                "required_rules": 2,
                "optional_rules": 1,
                "do_not_generalize_rules": 1,
            },
        )

    def test_skill_artifact_metric_counts_cross_example_fields(self) -> None:
        row = {
            "pack_id": "pack-1",
            "mode": "auto_skill_feature_driven_no_validation",
            "status": "success",
            "skill_md": "## Required rules\n- A\n## Optional rules\n- B\n## Do not generalize\n- C",
            "cross_example_report": {
                "conflicts": ["x"],
                "outliers": ["y", "z"],
                "candidate_rules": [
                    {"rule": "A", "supporting_examples": ["e1", "e2"]},
                    {"rule": "B", "support_count": 1},
                ],
            },
        }

        metric = skill_artifact_metric(row)

        self.assertEqual(metric["required_rules"], 1)
        self.assertEqual(metric["conflicts_count"], 1)
        self.assertEqual(metric["outliers_count"], 2)
        self.assertEqual(metric["candidate_rule_support_distribution"], {1: 1, 2: 1})

    def test_skill_artifact_summary_compares_modes(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "mode": "one_shot_skill_from_examples",
                "status": "success",
                "skill_md": "## Required rules\n- A",
            },
            {
                "pack_id": "pack-1",
                "mode": "auto_skill_feature_driven_no_validation",
                "status": "success",
                "skill_md": "## Required rules\n- A\n- B",
            },
        ]

        summary = skill_artifact_summary(rows)

        self.assertEqual(summary["comparison"]["paired_pack_count"], 1)
        self.assertEqual(summary["comparison"]["mean_deltas"]["required_rules"], 1)

    def test_token_usage_summary_uses_generation_and_judge_usage(self) -> None:
        rows = [
            {
                "mode": "few_shot_examples_only",
                "status": "success",
                "generation": {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                    }
                },
            },
            {
                "mode": "auto_skill",
                "status": "success",
                "layout_plan": {
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    }
                },
                "generation": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    }
                },
                "judge_calls": [
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 2,
                            "total_tokens": 22,
                        }
                    }
                ],
            },
            {
                "mode": "auto_skill",
                "status": "model_error",
                "generation": {
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    }
                },
            },
        ]

        summary = token_usage_summary(rows)

        mode = summary["by_mode"]["auto_skill"]
        self.assertEqual(mode["attempted_rows"], 2)
        self.assertEqual(mode["successful_rows"], 1)
        self.assertEqual(mode["status_counts"], {"model_error": 1, "success": 1})
        self.assertEqual(mode["layout_plan"]["total_tokens"], 10)
        self.assertEqual(mode["generation"]["total_tokens"], 19)
        self.assertEqual(mode["judge"]["total_tokens"], 22)
        self.assertEqual(mode["combined_model_calls"]["total_tokens"], 51)
        self.assertEqual(mode["combined_model_calls_avg_per_success"]["total_tokens"], 51)
        self.assertEqual(mode["combined_model_calls_avg_per_attempt"]["total_tokens"], 25.5)
        self.assertEqual(
            summary["comparison_to_examples_only_generation"]["modes"]["auto_skill"][
                "input_tokens"
            ]["delta"],
            -7,
        )

    def test_skill_induction_token_usage_summary_counts_model_calls(self) -> None:
        summary = skill_induction_token_usage_summary(
            [
                {
                    "mode": "auto_skill",
                    "status": "success",
                    "model_calls": [
                        {
                            "stage": "feature",
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
                        },
                        {
                            "stage": "merge",
                            "usage": {
                                "prompt_tokens": 20,
                                "completion_tokens": 5,
                                "total_tokens": 25,
                            },
                        },
                    ],
                },
                {
                    "mode": "auto_skill",
                    "status": "model_error",
                    "model_calls": [
                        {
                            "stage": "feature",
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ],
                },
            ]
        )

        mode = summary["by_mode"]["auto_skill"]
        self.assertEqual(mode["attempted_rows"], 2)
        self.assertEqual(mode["successful_rows"], 1)
        self.assertEqual(mode["model_calls"]["total_tokens"], 42)
        self.assertEqual(mode["model_calls_avg_per_attempt"]["total_tokens"], 21)
        self.assertEqual(mode["by_stage"]["feature"]["total_tokens"], 17)

    def test_abstract_signature_prompt_is_not_literal_reconstruction(self) -> None:
        prompt = build_abstract_signatures_prompt(skill_md="# Skill", n=2)

        self.assertIn("abstract_example_signatures", prompt)
        self.assertIn("official scores", prompt)
        self.assertNotIn("synthetic_examples", prompt)

    def test_parse_abstract_signatures(self) -> None:
        signatures = parse_abstract_signatures(
            '{"abstract_example_signatures":[{"task_family":"deck"}]}',
            limit=1,
        )

        self.assertEqual(signatures, [{"task_family": "deck"}])

    def test_literal_leakage_detector_finds_sample_specific_literals(self) -> None:
        pack = {
            "pack_id": "pack-1",
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "source_task_id": "498",
                    "task_input": (
                        "Write a 30-30-30-10 payment ratio contract for "
                        "Acme Ceramic Company."
                    ),
                    "desired_output": {
                        "status": "generated",
                        "text": "Contact finance@example.com.",
                    },
                }
            ],
        }

        report = literal_leakage_report(
            pack=pack,
            generated_signatures=[
                {"stable_constraints": ["Keep the 30-30-30-10 payment ratio."]}
            ],
        )

        self.assertTrue(report["has_literal_leakage"])
        self.assertIn("30-30-30-10", report["matched_literals"])

    def test_self_consistency_summary_uses_structured_scores_as_diagnostic(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "mode": "auto_skill_feature_driven_no_validation",
                "status": "success",
                "judge_report": {
                    "stable_feature_recall": 8,
                    "constraint_recall": 7,
                    "structure_recall": 6,
                    "style_signature": 5,
                    "leakage_penalty": 1,
                    "unsupported_specificity_penalty": 2,
                    "overall_self_consistency": 7,
                    "matched_constraints": ["structure"],
                    "missing_or_distorted_constraints": [],
                    "rationale": "ok",
                },
                "literal_leakage": {
                    "has_literal_leakage": True,
                    "matched_literal_count": 1,
                },
            },
            {
                "pack_id": "pack-2",
                "mode": "auto_skill_feature_driven_no_validation",
                "status": "success",
                "judge_report": {
                    "stable_feature_recall": 8,
                    "constraint_recall": 7,
                    "structure_recall": 6,
                    "style_signature": 5,
                    "leakage_penalty": 1,
                    "unsupported_specificity_penalty": 2,
                    "overall_self_consistency": 7,
                },
            },
        ]

        summary = self_consistency_summary(rows)

        self.assertFalse(summary["main_performance_metric"])
        self.assertEqual(summary["status_counts"], {"success": 2})
        self.assertEqual(
            summary["by_mode"]["auto_skill_feature_driven_no_validation"][
                "overall_self_consistency"
            ]["mean"],
            7,
        )
        self.assertEqual(
            summary["literal_leakage_by_mode"]["auto_skill_feature_driven_no_validation"][
                "literal_leakage_rate"
            ],
            1.0,
        )
        self.assertEqual(
            summary["invalid_success_rows"][0]["errors"],
            [
                "matched_constraints_must_be_list",
                "missing_or_distorted_constraints_must_be_list",
                "rationale_must_be_string",
            ],
        )

    def test_parse_self_consistency_report_rejects_invalid_scores(self) -> None:
        report = parse_self_consistency_report(
            """{
              "stable_feature_recall": true,
              "constraint_recall": 7,
              "structure_recall": 6,
              "style_signature": 5,
              "leakage_penalty": 1,
              "unsupported_specificity_penalty": 2,
              "overall_self_consistency": 7,
              "matched_constraints": [],
              "missing_or_distorted_constraints": [],
              "rationale": "bad"
            }"""
        )

        self.assertEqual(
            report["parse_error"],
            "stable_feature_recall_must_be_number_0_to_10",
        )

    def test_signature_judge_prompt_mentions_private_data_boundary(self) -> None:
        pack = {
            "pack_id": "pack-1",
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "task_input": "Make a report.",
                    "desired_output": {"status": "generated", "text": "Report"},
                }
            ],
        }

        prompt = build_signature_consistency_judge_prompt(
            train_pack=pack,
            abstract_signatures=[{"task_family": "report"}],
            literal_leakage={"matched_literals": []},
        )

        self.assertIn("stable_feature_recall", prompt)
        self.assertIn("official benchmark scores", prompt)

    def test_generate_signatures_with_parse_retry_recovers_from_bad_json(self) -> None:
        generation, signatures, attempts = generate_signatures_with_parse_retry(
            client=SequencedCompletionClient(
                [
                    "not json",
                    '{"abstract_example_signatures":[{"task_family":"deck"}]}',
                ]
            ),
            prompt="signatures",
            n=1,
            temperature=0.2,
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(generation.finish_reason, "stop")
        self.assertEqual(signatures, [{"task_family": "deck"}])
        self.assertEqual(
            [attempt["status"] for attempt in attempts],
            ["signature_parse_error", "success"],
        )

    def test_generate_signatures_with_parse_retry_does_not_retry_truncation(self) -> None:
        generation, signatures, attempts = generate_signatures_with_parse_retry(
            client=SequencedCompletionClient(
                ["partial", '{"abstract_example_signatures":[{"task_family":"deck"}]}'],
                finish_reasons=["length", "stop"],
            ),
            prompt="signatures",
            n=1,
            temperature=0.2,
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(generation.finish_reason, "length")
        self.assertEqual(signatures, [])
        self.assertEqual(
            [attempt["status"] for attempt in attempts],
            ["signature_generation_incomplete"],
        )

    def test_self_consistency_judge_with_parse_retry_recovers_from_bad_json(self) -> None:
        judge, report, status, attempts = judge_with_parse_retry(
            client=SequencedCompletionClient(
                [
                    "not json",
                    """{
                      "stable_feature_recall": 8,
                      "constraint_recall": 7,
                      "structure_recall": 6,
                      "style_signature": 5,
                      "leakage_penalty": 1,
                      "unsupported_specificity_penalty": 2,
                      "overall_self_consistency": 7,
                      "matched_constraints": [],
                      "missing_or_distorted_constraints": [],
                      "rationale": "ok"
                    }""",
                ]
            ),
            prompt="judge",
            max_tokens=128,
            parse_max_attempts=2,
        )

        self.assertEqual(judge.finish_reason, "stop")
        self.assertEqual(report["overall_self_consistency"], 7.0)
        self.assertEqual(status, "success")
        self.assertEqual(
            [attempt["status"] for attempt in attempts],
            ["judge_parse_error", "success"],
        )


if __name__ == "__main__":
    unittest.main()
