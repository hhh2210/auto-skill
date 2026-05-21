from __future__ import annotations

import unittest

from auto_skill.author_style_selection import (
    build_author_style_audit_subset,
    build_author_style_subset,
    pack_id_from_task_ref,
    select_audit_rows,
    select_quality_rows,
)


class AuthorStyleSelectionTests(unittest.TestCase):
    def test_select_quality_rows_filters_and_ranks(self) -> None:
        rows = [
            {
                "pack_id": "p-low",
                "dataset_label": "mendeley_full",
                "paper_plot_role": "main_cross_topic",
                "paper_plot_eligible": True,
                "paper_quality_tier": "cross_topic_moderate_signal",
                "deterministic_metrics": {"margin_vs_source_pool_impostors": 0.2},
            },
            {
                "pack_id": "p-high",
                "dataset_label": "mendeley_full",
                "paper_plot_role": "main_cross_topic",
                "paper_plot_eligible": True,
                "paper_quality_tier": "cross_topic_high_signal",
                "deterministic_metrics": {"margin_vs_source_pool_impostors": 0.1},
            },
            {
                "pack_id": "p-debug",
                "dataset_label": "mendeley_full",
                "paper_plot_role": "debug_only",
                "paper_plot_eligible": False,
                "paper_quality_tier": "cross_topic_low_signal",
            },
        ]

        selected = select_quality_rows(
            rows,
            dataset_label="mendeley_full",
            paper_plot_role="main_cross_topic",
            require_paper_eligible=True,
            limit=2,
            sort_by="quality_rank",
        )

        self.assertEqual([row["pack_id"] for row in selected], ["p-high", "p-low"])

    def test_build_subset_filters_private_eval_and_negatives_by_pack(self) -> None:
        quality_rows = [
            {
                "pack_id": "p1",
                "dataset_label": "mendeley_full",
                "paper_plot_role": "main_cross_topic",
                "paper_plot_eligible": True,
                "paper_quality_tier": "cross_topic_high_signal",
            }
        ]
        packs = [{"pack_id": "p1", "heldout_tasks": [{"task_id": "p1::heldout::0"}]}]
        private_eval = [{"pack_id": "p1"}, {"pack_id": "p2"}]
        hard_negatives = [
            {"target_task_ref": "p1::heldout::0", "negative_id": "n1"},
            {"target_task_ref": "p2::heldout::0", "negative_id": "n2"},
        ]

        subset = build_author_style_subset(
            quality_rows=quality_rows,
            packs=packs,
            private_eval=private_eval,
            hard_negatives=hard_negatives,
            dataset_label="mendeley_full",
            paper_plot_role="main_cross_topic",
            require_paper_eligible=True,
            limit=None,
            sort_by="quality_rank",
        )

        self.assertEqual(subset["summary"]["packs"], 1)
        self.assertEqual(subset["summary"]["private_eval_rows"], 1)
        self.assertEqual(subset["summary"]["hard_negatives"], 1)
        self.assertEqual(subset["summary"]["task_refs_with_hard_negatives"], 1)

    def test_pack_id_from_task_ref(self) -> None:
        self.assertEqual(pack_id_from_task_ref("p1::heldout::0"), "p1")
        self.assertEqual(pack_id_from_task_ref("p1::other"), "p1")

    def test_select_audit_rows_filters_and_ranks(self) -> None:
        rows = [
            {
                "pack_id": "p-low",
                "status": "success",
                "audit": {
                    "usable": True,
                    "should_use_for_smoke": True,
                    "style_extractability_1_to_5": 3,
                    "negative_strength_1_to_5": 4,
                    "topic_leakage_risk_1_to_5": 3,
                    "model_familiarity_risk_1_to_5": 2,
                },
            },
            {
                "pack_id": "p-high",
                "status": "success",
                "audit": {
                    "usable": True,
                    "should_use_for_smoke": True,
                    "style_extractability_1_to_5": 4,
                    "negative_strength_1_to_5": 4,
                    "topic_leakage_risk_1_to_5": 3,
                    "model_familiarity_risk_1_to_5": 2,
                },
            },
            {
                "pack_id": "p-reject",
                "status": "success",
                "audit": {
                    "usable": False,
                    "should_use_for_smoke": True,
                    "style_extractability_1_to_5": 5,
                    "negative_strength_1_to_5": 5,
                    "topic_leakage_risk_1_to_5": 1,
                    "model_familiarity_risk_1_to_5": 1,
                },
            },
        ]

        selected = select_audit_rows(rows, limit=2)

        self.assertEqual([row["pack_id"] for row in selected], ["p-high", "p-low"])

    def test_build_audit_subset_requires_selected_negative_coverage(self) -> None:
        audit_rows = [
            {
                "pack_id": "p1",
                "status": "success",
                "audit": {
                    "usable": True,
                    "should_use_for_smoke": True,
                    "style_extractability_1_to_5": 3,
                    "negative_strength_1_to_5": 4,
                    "topic_leakage_risk_1_to_5": 4,
                    "model_familiarity_risk_1_to_5": 3,
                },
            },
            {
                "pack_id": "p2",
                "status": "success",
                "audit": {
                    "usable": True,
                    "should_use_for_smoke": True,
                    "style_extractability_1_to_5": 3,
                    "negative_strength_1_to_5": 4,
                    "topic_leakage_risk_1_to_5": 4,
                    "model_familiarity_risk_1_to_5": 3,
                },
            },
        ]
        packs = [
            {"pack_id": "p1", "heldout_tasks": [{"task_id": "p1::heldout::0"}]},
            {"pack_id": "p2", "heldout_tasks": [{"task_id": "p2::heldout::0"}]},
        ]
        private_eval = [{"pack_id": "p1"}, {"pack_id": "p2"}]
        hard_negatives = [
            {
                "target_task_ref": "p1::heldout::0",
                "gpt_rerank": {"status": "selected"},
            },
            {
                "target_task_ref": "p1::heldout::0",
                "gpt_rerank": {"status": "selected"},
            },
            {
                "target_task_ref": "p2::heldout::0",
                "gpt_rerank": {"status": "fallback_prefilter"},
            },
        ]

        subset = build_author_style_audit_subset(
            audit_rows=audit_rows,
            packs=packs,
            private_eval=private_eval,
            hard_negatives=hard_negatives,
            require_usable=True,
            require_should_use=True,
            min_style_extractability=3,
            min_negative_strength=4,
            max_topic_leakage=4,
            max_model_familiarity=3,
            limit=None,
            sort_by="source_order",
            min_negatives_per_heldout=2,
            require_gpt_selected_negatives=True,
        )

        self.assertEqual(subset["summary"]["packs"], 1)
        self.assertEqual(subset["summary"]["hard_negatives"], 2)
        self.assertEqual(subset["summary"]["selected_pack_ids"], ["p1"])


if __name__ == "__main__":
    unittest.main()
