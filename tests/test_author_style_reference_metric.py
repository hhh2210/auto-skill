from __future__ import annotations

import unittest

from auto_skill.metrics.author_style_reference import (
    ReferenceCandidate,
    build_reference_retrieval_jobs,
    build_reference_retrieval_prompt,
    build_success_row,
    parse_reference_retrieval_report,
    summarize_reference_retrieval_rows,
)


def _pack() -> dict[str, object]:
    return {
        "pack_id": "author_style_personal_blog_0001",
        "source": "personal_blog_history",
        "train_examples": [
            {
                "example_id": "author_style_personal_blog_0001::train::0",
                "source_task_id": "train-source-0",
                "desired_output": {
                    "text": (
                        "same author writes with ellipses... tiny jokes, "
                        "lowercase i, and diary drift."
                    )
                },
            }
        ],
        "heldout_tasks": [
            {
                "task_id": "author_style_personal_blog_0001::heldout::0",
                "source_task_id": "heldout-source-0",
            }
        ],
    }


class AuthorStyleReferenceMetricTests(unittest.TestCase):
    def test_build_jobs_use_source_target_reference_and_other_author_negatives(self) -> None:
        jobs, skipped = build_reference_retrieval_jobs(
            packs=[_pack()],
            private_eval=[
                {
                    "pack_id": "author_style_personal_blog_0001",
                    "heldout_private": [
                        {
                            "task_ref": "author_style_personal_blog_0001::heldout::0",
                            "reference_output_private": (
                                "target has same ellipses... lowercase i and diary-like drift."
                            ),
                        }
                    ],
                }
            ],
            hard_negatives=[
                {
                    "target_task_ref": "author_style_personal_blog_0001::heldout::0",
                    "negative_id": "neg-1",
                    "negative_type": "topic_time_length_style_matched_impostor",
                    "public_negative_text": "Other author writes in clean formal sentences.",
                }
            ],
        )

        self.assertEqual(skipped, [])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].target_text[:6], "target")
        self.assertEqual(
            jobs[0].expected_candidate_id,
            "author_style_personal_blog_0001::train::0",
        )
        self.assertEqual({candidate.kind for candidate in jobs[0].candidates}, {
            "same_author_reference",
            "topic_time_length_style_matched_impostor",
        })

    def test_prompt_names_target_candidates_and_not_private_labels(self) -> None:
        prompt = build_reference_retrieval_prompt(
            target_text="target text",
            candidates=[
                ReferenceCandidate("same-ref", "same_author_reference", "reference text"),
                ReferenceCandidate("negative::1", "hard_negative", "negative text"),
            ],
        )

        self.assertIn("Target text:", prompt)
        self.assertIn("Candidate same-ref", prompt)
        self.assertIn("Candidate negative::1", prompt)
        self.assertIn("hard negatives", prompt)
        self.assertNotIn("author_hash", prompt)

    def test_parse_report_requires_selected_candidate_id(self) -> None:
        report = parse_reference_retrieval_report(
            '{"most_similar_candidate_id":"same-ref",'
            '"ranked_candidate_ids":["same-ref","negative::1"],'
            '"confidence":"high","rationale":"style match"}',
            candidate_ids={"same-ref", "negative::1"},
        )

        self.assertNotIn("parse_error", report)
        self.assertEqual(report["most_similar_candidate_id"], "same-ref")

    def test_parse_report_normalizes_candidate_prefix(self) -> None:
        report = parse_reference_retrieval_report(
            '{"most_similar_candidate_id":"candidate same-ref",'
            '"ranked_candidate_ids":["Candidate same-ref","candidate negative::1"],'
            '"confidence":"high","rationale":"style match"}',
            candidate_ids={"same-ref", "negative::1"},
        )

        self.assertNotIn("parse_error", report)
        self.assertEqual(report["most_similar_candidate_id"], "same-ref")
        self.assertEqual(report["ranked_candidate_ids"], ["same-ref", "negative::1"])

    def test_parse_report_rejects_unknown_candidate(self) -> None:
        report = parse_reference_retrieval_report(
            '{"most_similar_candidate_id":"missing",'
            '"ranked_candidate_ids":["missing"],'
            '"confidence":"high","rationale":"bad"}',
            candidate_ids={"same-ref"},
        )

        self.assertEqual(
            report["parse_error"],
            "most_similar_candidate_id_must_match_candidate_id",
        )

    def test_build_jobs_reject_multiple_references_per_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "references_per_target must be 1"):
            build_reference_retrieval_jobs(
                packs=[_pack()],
                private_eval=[],
                hard_negatives=[],
                references_per_target=2,
            )

    def test_success_row_redacts_rationale_and_counts_hard_negatives(self) -> None:
        jobs, _ = build_reference_retrieval_jobs(
            packs=[_pack()],
            private_eval=[
                {
                    "pack_id": "author_style_personal_blog_0001",
                    "heldout_private": [
                        {
                            "task_ref": "author_style_personal_blog_0001::heldout::0",
                            "reference_output_private": "PRIVATE_TARGET_SENTINEL",
                        }
                    ],
                }
            ],
            hard_negatives=[
                {
                    "target_task_ref": "author_style_personal_blog_0001::heldout::0",
                    "negative_id": "neg-1",
                    "negative_type": "topic_time_length_style_matched_impostor",
                    "public_negative_text": "PRIVATE_NEGATIVE_SENTINEL",
                }
            ],
        )
        row = build_success_row(
            job=jobs[0],
            judge={"model": "judge"},
            report={
                "most_similar_candidate_id": "author_style_personal_blog_0001::train::0",
                "ranked_candidate_ids": [
                    "author_style_personal_blog_0001::train::0",
                    "negative::1",
                ],
                "confidence": "high",
                "rationale": "PRIVATE_TARGET_SENTINEL PRIVATE_NEGATIVE_SENTINEL",
            },
            status="success",
            attempts=[],
        )

        self.assertEqual(row["hard_negative_count"], 1)
        self.assertNotIn("rationale", row["judge_report"])
        self.assertNotIn("PRIVATE_TARGET_SENTINEL", str(row))
        self.assertNotIn("PRIVATE_NEGATIVE_SENTINEL", str(row))

    def test_parse_error_row_redacts_model_candidate_fields(self) -> None:
        jobs, _ = build_reference_retrieval_jobs(
            packs=[_pack()],
            private_eval=[
                {
                    "pack_id": "author_style_personal_blog_0001",
                    "heldout_private": [
                        {
                            "task_ref": "author_style_personal_blog_0001::heldout::0",
                            "reference_output_private": "PRIVATE_TARGET_SENTINEL",
                        }
                    ],
                }
            ],
            hard_negatives=[
                {
                    "target_task_ref": "author_style_personal_blog_0001::heldout::0",
                    "negative_id": "neg-1",
                    "negative_type": "topic_time_length_style_matched_impostor",
                    "public_negative_text": "negative text",
                }
            ],
        )
        row = build_success_row(
            job=jobs[0],
            judge={"model": "judge"},
            report={
                "most_similar_candidate_id": "PRIVATE_TARGET_SENTINEL",
                "ranked_candidate_ids": ["PRIVATE_TARGET_SENTINEL"],
                "confidence": "high",
                "parse_error": "most_similar_candidate_id_must_match_candidate_id",
            },
            status="judge_parse_error",
            attempts=[
                {
                    "attempt": 1,
                    "status": "judge_parse_error",
                    "selected_candidate_id": "PRIVATE_TARGET_SENTINEL",
                    "parse_error": "most_similar_candidate_id_must_match_candidate_id",
                }
            ],
        )

        self.assertEqual(
            row["judge_report"],
            {"parse_error": "most_similar_candidate_id_must_match_candidate_id"},
        )
        self.assertNotIn("selected_candidate_id", row["judge_parse_attempts"][0])
        self.assertNotIn("PRIVATE_TARGET_SENTINEL", str(row))

    def test_model_controlled_parse_error_is_redacted(self) -> None:
        report = parse_reference_retrieval_report(
            '{"parse_error":"PRIVATE_TARGET_SENTINEL"}',
            candidate_ids={"same-ref"},
        )

        self.assertEqual(report, {"parse_error": "json_parse_error"})

    def test_summary_reports_oracle_accuracy(self) -> None:
        summary = summarize_reference_retrieval_rows(
            [
                {
                    "pack_id": "p1",
                    "task_id": "t1",
                    "source": "personal_blog_history",
                    "status": "success",
                    "oracle_correct": True,
                    "expected_rank": 1,
                },
                {
                    "pack_id": "p2",
                    "task_id": "t1",
                    "source": "cross_topic_online_comment_history",
                    "status": "success",
                    "oracle_correct": False,
                    "expected_rank": 3,
                },
            ],
            skipped=[{"reason": "missing_negatives"}],
        )

        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["median_expected_rank"], 2)
        self.assertEqual(summary["skipped"]["reason_counts"], {"missing_negatives": 1})


if __name__ == "__main__":
    unittest.main()
