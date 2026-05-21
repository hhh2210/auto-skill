from __future__ import annotations

import unittest

from auto_skill.author_style_cluster_manifest import build_cluster_manifest


def row(
    pack_id: str,
    tags: list[str],
    *,
    topic_like: list[str] | None = None,
    source_pool_margin: float = 0.2,
) -> dict[str, object]:
    deterministic_tags = ["short_sentence"]
    if "question_heavy" in tags:
        deterministic_tags.append("question_heavy")
    return {
        "schema_version": "author-style-pack-quality/v3",
        "pack_id": pack_id,
        "source": "cross_topic_online_comment_history",
        "paper_quality_tier": "cross_topic_high_signal",
        "paper_plot_role": "main_cross_topic",
        "canonical_gpt_style_tags": tags,
        "topic_like_gpt_tags": topic_like or [],
        "deterministic_cluster_tags": deterministic_tags,
        "style_extractability_1_to_5": 4,
        "topic_leakage_risk_1_to_5": 2,
        "model_familiarity_risk_1_to_5": 2,
        "negative_strength_1_to_5": 4,
        "hard_negative_metrics": {
            "mean_style_confusability_1_to_5": 4.5,
            "mean_topic_shortcut_risk_1_to_5": 1.0,
        },
        "deterministic_metrics": {
            "mean_train_heldout_style_similarity": 0.9,
            "mean_train_negative_style_similarity": 0.8,
            "style_similarity_margin_vs_negatives": 0.1,
            "margin_vs_source_pool_impostors": source_pool_margin,
            "source_pool_pairwise_win_rate": 1.0,
            "heldout_topic_seen_in_train_rate": 0.0,
            "mean_train_heldout_content_tag_jaccard": 0.0,
        },
        "quality_flags": [],
        "diagnostic_caveats": [],
    }


class AuthorStyleClusterManifestTests(unittest.TestCase):
    def test_build_cluster_manifest_keeps_topic_like_tags_as_exclusions(self) -> None:
        manifest, pack_profiles = build_cluster_manifest(
            [
                row(
                    "p2",
                    ["question_heavy", "casual_conversational"],
                    topic_like=["gaming_pc_vocab"],
                    source_pool_margin=0.1,
                ),
                row("p1", ["question_heavy", "ellipsis_heavy"], source_pool_margin=0.3),
                row("p3", ["ellipsis_heavy"], topic_like=["sports_fandom_energy"]),
            ],
            min_cluster_size=2,
            representative_limit=1,
        )

        self.assertEqual(manifest["schema_version"], "author-style-cluster-manifest/v1")
        self.assertEqual(manifest["pack_count"], 3)
        self.assertTrue(manifest["artifact_boundary"]["must_not_use_for_induction"])
        self.assertEqual(manifest["cluster_readiness"]["status"], "ok")
        self.assertEqual(manifest["cluster_readiness"]["paper_eligible_cluster_count"], 2)
        self.assertEqual(
            manifest["cluster_readiness"][
                "pack_rate_with_at_least_2_paper_eligible_style_tags"
            ],
            0.333,
        )
        self.assertEqual(
            manifest["style_tag_counts"],
            {"ellipsis_heavy": 2, "question_heavy": 2, "casual_conversational": 1},
        )
        self.assertEqual(
            manifest["background_style_tag_counts"],
            {"casual_conversational": 1},
        )
        self.assertEqual(
            manifest["topic_like_exclusion"]["topic_like_tag_counts"],
            {"gaming_pc_vocab": 1, "sports_fandom_energy": 1},
        )
        eligible_tags = [
            profile["tag"] for profile in manifest["paper_eligible_cluster_tags"]
        ]
        self.assertEqual(eligible_tags, ["ellipsis_heavy", "question_heavy"])
        question_profile = next(
            profile
            for profile in manifest["paper_eligible_cluster_tags"]
            if profile["tag"] == "question_heavy"
        )
        self.assertEqual(question_profile["family"], "pacing_punctuation")
        self.assertTrue(question_profile["paper_eligible"])
        self.assertEqual(question_profile["cluster_kind"], "deterministic_plus_llm")
        self.assertEqual(question_profile["supporting_deterministic_tags"], ["question_heavy"])
        self.assertEqual(question_profile["deterministic_support_pack_count"], 2)
        self.assertEqual(question_profile["co_style_tag_counts"]["ellipsis_heavy"], 1)
        self.assertEqual(
            question_profile["metric_means"]["source_pool_margin"],
            0.2,
        )
        self.assertEqual(question_profile["representative_pack_ids"], ["p1"])
        self.assertEqual([profile["pack_id"] for profile in pack_profiles], ["p1", "p2", "p3"])
        self.assertEqual(
            pack_profiles[1]["topic_like_exclusions"],
            ["gaming_pc_vocab"],
        )
        self.assertEqual(
            pack_profiles[1]["background_style_tags"],
            ["casual_conversational"],
        )
        self.assertEqual(pack_profiles[1]["excluded_topic_like_tags"], ["gaming_pc_vocab"])
        self.assertTrue(pack_profiles[1]["artifact_boundary"]["must_not_use_for_induction"])


if __name__ == "__main__":
    unittest.main()
