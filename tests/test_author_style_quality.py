from __future__ import annotations

import unittest

from auto_skill.author_style_quality import (
    normalize_style_tag,
    pack_quality_rows,
    summarize_quality_datasets,
)


class AuthorStyleQualityTests(unittest.TestCase):
    def test_pack_quality_flags_topic_leakage_and_negative_coverage(self) -> None:
        packs = [
            {
                "pack_id": "p1",
                "source": "personal_blog_history",
                "domain": {"kind": "personal_blog"},
                "train_examples": [{"example_id": "e1"}],
                "heldout_tasks": [{"task_id": "p1::heldout::0"}],
            }
        ]
        private_rows = [{"pack_id": "p1", "cluster_tags": ["short_sentence"]}]
        private_rows[0]["train_private"] = [
            {
                "task_ref": "p1::train::0",
                "private_topic": "school",
                "content_tags": ["school", "friend"],
                "word_count": 100,
                "style_features": {
                    "avg_sentence_words": 10,
                    "question_per_100w": 1,
                    "exclamation_per_100w": 1,
                    "ellipsis_per_100w": 0,
                    "paren_per_100w": 0,
                    "first_person_per_100w": 2,
                    "contraction_per_100w": 1,
                    "hedge_marker": 0,
                },
            }
        ]
        private_rows[0]["heldout_private"] = [
            {
                "task_ref": "p1::heldout::0",
                "private_topic": "school",
                "content_tags": ["school", "music"],
                "word_count": 90,
                "style_features": {
                    "avg_sentence_words": 11,
                    "question_per_100w": 1,
                    "exclamation_per_100w": 1,
                    "ellipsis_per_100w": 0,
                    "paren_per_100w": 0,
                    "first_person_per_100w": 2,
                    "contraction_per_100w": 1,
                    "hedge_marker": 0,
                },
            }
        ]
        audits = [
            {
                "pack_id": "p1",
                "status": "success",
                "audit": {
                    "style_cluster_tags": ["ellipsis_heavy", "teen_diary_voice"],
                    "style_extractability_1_to_5": 4,
                    "topic_leakage_risk_1_to_5": 4,
                    "model_familiarity_risk_1_to_5": 1,
                    "negative_strength_1_to_5": 3,
                    "usable": True,
                    "should_use_for_smoke": True,
                    "reject_reasons": ["topic leakage"],
                },
                "public_negative_text": (
                    "This is a plain sentence. It has little punctuation and no quirks."
                ),
            }
        ]
        negatives = [
            {
                "target_task_ref": "p1::heldout::0",
                "negative_type": "topic_time_length_style_matched_impostor",
                "gpt_rerank": {
                    "status": "selected",
                    "style_confusability_1_to_5": 4,
                    "topic_shortcut_risk_1_to_5": 3,
                },
                "match_features": {
                    "same_topic": True,
                    "style_similarity": 0.6,
                    "length_ratio": 0.9,
                    "content_tag_jaccard": 0.2,
                    "time_score": 0.8,
                },
            }
        ]

        rows = pack_quality_rows(
            label="blog",
            packs=packs,
            private_rows=private_rows,
            audit_rows=audits,
            hard_negative_rows=negatives,
        )

        self.assertEqual(rows[0]["gpt_style_cluster_tags"], ["ellipsis_heavy", "teen_diary_voice"])
        self.assertEqual(
            rows[0]["canonical_gpt_style_tags"],
            ["ellipsis_heavy", "first_person_diary"],
        )
        self.assertIn("high_topic_leakage", rows[0]["quality_flags"])
        self.assertIn("weak_negative_set", rows[0]["quality_flags"])
        self.assertIn("insufficient_hard_negative_coverage", rows[0]["quality_flags"])
        self.assertEqual(
            rows[0]["hard_negative_metrics"]["mean_style_confusability_1_to_5"],
            4.0,
        )
        self.assertEqual(
            rows[0]["deterministic_metrics"]["heldout_topic_seen_in_train_rate"],
            1.0,
        )
        self.assertEqual(rows[0]["quality_gate"]["status"], "fail")
        self.assertEqual(
            rows[0]["quality_gate"]["tier"],
            "temporal_context_dominated",
        )
        self.assertIn(
            "insufficient_hard_negative_coverage",
            rows[0]["quality_gate"]["reasons"],
        )
        self.assertIn(
            "same_train_heldout_topic_by_design",
            rows[0]["diagnostic_caveats"],
        )
        self.assertFalse(rows[0]["paper_plot_eligible"])

    def test_mendeley_cross_topic_pack_can_be_paper_main_candidate(self) -> None:
        packs = [
            {
                "pack_id": "r1",
                "source": "cross_topic_online_comment_history",
                "domain": {
                    "kind": "cross_topic_online_comment",
                    "split_kind": "known_unknown_cross_topic",
                },
                "train_examples": [{"example_id": f"e{index}"} for index in range(4)],
                "heldout_tasks": [{"task_id": "r1::heldout::0"}],
            }
        ]
        style_features = {
            "avg_sentence_words": 12,
            "question_per_100w": 1,
            "exclamation_per_100w": 1,
            "ellipsis_per_100w": 0,
            "paren_per_100w": 0,
            "first_person_per_100w": 2,
            "contraction_per_100w": 1,
            "hedge_marker": 0,
        }
        private_rows = [
            {
                "pack_id": "r1",
                "author_hash": "author_a",
                "cluster_tags": ["short_sentence"],
                "train_private": [
                    {
                        "task_ref": f"r1::train::{index}",
                        "private_topic": f"subreddit_{index}",
                        "content_tags": [f"topic_{index}"],
                        "word_count": 100,
                        "style_features": style_features,
                    }
                    for index in range(4)
                ],
                "heldout_private": [
                    {
                        "task_ref": "r1::heldout::0",
                        "private_topic": "subreddit_heldout",
                        "content_tags": ["heldout_topic"],
                        "word_count": 105,
                        "style_features": style_features,
                    }
                ],
            },
            {
                "pack_id": "r2",
                "author_hash": "author_b",
                "cluster_tags": ["long_sentence"],
                "train_private": [],
                "heldout_private": [
                    {
                        "task_ref": f"r2::heldout::{index}",
                        "private_topic": f"other_subreddit_{index}",
                        "content_tags": [f"other_topic_{index}"],
                        "word_count": 105,
                        "style_features": {
                            "avg_sentence_words": 30,
                            "question_per_100w": 0,
                            "exclamation_per_100w": 0,
                            "ellipsis_per_100w": 0,
                            "paren_per_100w": 0,
                            "first_person_per_100w": 0,
                            "contraction_per_100w": 0,
                            "hedge_marker": 0,
                        },
                    }
                    for index in range(8)
                ],
            },
        ]
        audits = [
            {
                "pack_id": "r1",
                "status": "success",
                "audit": {
                    "style_cluster_tags": ["short_sentences"],
                    "style_extractability_1_to_5": 4,
                    "topic_leakage_risk_1_to_5": 3,
                    "model_familiarity_risk_1_to_5": 2,
                    "negative_strength_1_to_5": 4,
                    "usable": True,
                    "should_use_for_smoke": True,
                    "reject_reasons": [],
                },
            }
        ]
        negatives = [
            {
                "target_task_ref": "r1::heldout::0",
                "negative_type": "cross_author_style_matched_impostor",
                "public_negative_text": "I do not know, but I think it works.",
                "gpt_rerank": {
                    "status": "selected",
                    "style_confusability_1_to_5": 4,
                    "topic_shortcut_risk_1_to_5": 1,
                },
                "match_features": {
                    "same_topic": False,
                    "style_similarity": 0.8,
                    "length_ratio": 1.0,
                    "content_tag_jaccard": 0.0,
                    "time_score": 0.5,
                },
            }
            for _ in range(4)
        ]

        rows = pack_quality_rows(
            label="mendeley",
            packs=packs,
            private_rows=private_rows,
            audit_rows=audits,
            hard_negative_rows=negatives,
        )

        self.assertEqual(rows[0]["quality_gate"]["status"], "pass")
        self.assertEqual(rows[0]["quality_gate"]["policy"], "mendeley_cross_topic_v1")
        self.assertEqual(rows[0]["paper_quality_tier"], "cross_topic_high_signal")
        self.assertEqual(rows[0]["paper_plot_role"], "main_cross_topic")
        self.assertTrue(rows[0]["paper_plot_eligible"])
        self.assertTrue(rows[0]["paper_main_candidate"])
        self.assertEqual(
            rows[0]["source_pool_impostor_baseline"]["pool_kind"],
            "accepted_private_heldout_pool",
        )
        self.assertEqual(
            rows[0]["deterministic_metrics"]["heldout_topic_seen_in_train_rate"],
            0.0,
        )
        self.assertGreater(
            rows[0]["deterministic_metrics"]["margin_vs_source_pool_impostors"],
            0.0,
        )
        self.assertEqual(
            rows[0]["deterministic_metrics"]["source_pool_pairwise_win_rate"],
            1.0,
        )

    def test_summarize_quality_datasets_counts_tags(self) -> None:
        rows = [
            {
                "pack_id": "p1",
                "source": "s",
                "domain_kind": "d",
                "train_examples": 4,
                "heldout_tasks": 1,
                "deterministic_cluster_tags": ["short_sentence"],
                "gpt_style_cluster_tags": ["short_sentence", "question_heavy"],
                "canonical_gpt_style_tags": ["short_sentences", "question_heavy"],
                "topic_like_gpt_tags": [],
                "style_extractability_1_to_5": 4,
                "topic_leakage_risk_1_to_5": 2,
                "model_familiarity_risk_1_to_5": 1,
                "negative_strength_1_to_5": 4,
                "hard_negative_coverage_by_task": {"t1": 4},
                "hard_negative_metrics": {"mean_style_confusability_1_to_5": 4.5},
                "deterministic_metrics": {
                    "mean_train_heldout_style_similarity": 0.9,
                    "mean_train_negative_style_similarity": 0.6,
                    "style_similarity_margin_vs_negatives": 0.3,
                    "heldout_topic_seen_in_train_rate": 0.0,
                    "mean_train_heldout_content_tag_jaccard": 0.1,
                },
                "quality_flags": [],
                "diagnostic_caveats": [],
                "quality_gate": {
                    "status": "pass",
                    "tier": "cross_topic_high_signal",
                    "policy": "mendeley_cross_topic_v1",
                },
                "paper_quality_tier": "cross_topic_high_signal",
                "paper_plot_role": "main_cross_topic",
                "paper_plot_eligible": True,
                "paper_main_candidate": True,
            }
        ]

        summary = summarize_quality_datasets([("sample", rows)])

        self.assertEqual(summary["datasets"][0]["packs"], 1)
        self.assertEqual(
            summary["aggregate"]["gpt_style_cluster_tag_counts"],
            {"short_sentence": 1, "question_heavy": 1},
        )
        self.assertEqual(
            summary["aggregate"]["canonical_gpt_style_tag_counts"],
            {"short_sentences": 1, "question_heavy": 1},
        )
        self.assertEqual(summary["aggregate"]["hard_negative_coverage"]["min"], 4)
        self.assertEqual(
            summary["aggregate"]["deterministic_style_stability"][
                "mean_style_similarity_margin_vs_negatives"
            ],
            0.3,
        )
        self.assertIn("source_pool_impostor_baseline", summary["aggregate"])
        self.assertEqual(
            summary["aggregate"]["paper_quality_tier_counts"],
            {"cross_topic_high_signal": 1},
        )
        self.assertEqual(summary["aggregate"]["paper_main_candidate_count"], 1)

    def test_normalize_style_tag_drops_topic_like_tags(self) -> None:
        self.assertEqual(normalize_style_tag("short_punchy_sentences"), "short_sentences")
        self.assertEqual(normalize_style_tag("sarcastic_asides"), "sarcasm_snark")
        self.assertIsNone(normalize_style_tag("gaming_pc_vocab"))


if __name__ == "__main__":
    unittest.main()
