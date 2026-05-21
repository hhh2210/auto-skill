from __future__ import annotations

import unittest

from auto_skill.author_style_stratified_eval import (
    summarize_stratified_author_style_eval,
)


def eval_row(pack_id: str, mode: str, score: int, wins: int) -> dict[str, object]:
    return {
        "schema_version": "author-style-heldout-eval/v1",
        "status": "success",
        "pack_id": pack_id,
        "task_id": f"{pack_id}::heldout::0",
        "mode": mode,
        "style_likeness_1_to_10": score,
        "candidate_beats_negatives": wins,
        "hard_negative_count": 4,
    }


def pack_profile(
    pack_id: str,
    tags: list[str],
    *,
    background: list[str] | None = None,
    topic_like: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "author-style-pack-cluster-profile/v1",
        "pack_id": pack_id,
        "eligible_style_tags": tags,
        "background_style_tags": background or [],
        "style_families": ["pacing_punctuation"],
        "excluded_topic_like_tags": topic_like or [],
    }


class AuthorStyleStratifiedEvalTests(unittest.TestCase):
    def test_stratified_summary_reports_cluster_macro_without_single_count_claim(self) -> None:
        manifest = {
            "paper_eligible_cluster_tags": [
                {"tag": "question_heavy"},
                {"tag": "ellipsis_heavy"},
            ]
        }
        pack_profiles = [
            pack_profile("p1", ["question_heavy", "ellipsis_heavy"]),
            pack_profile(
                "p2",
                ["question_heavy"],
                background=["casual_conversational"],
                topic_like=["gaming_pc_vocab"],
            ),
            pack_profile("p3", ["ellipsis_heavy"]),
        ]
        eval_rows = [
            eval_row("p1", "prompt_only", 2, 0),
            eval_row("p1", "few_shot_examples_only", 5, 3),
            eval_row("p2", "prompt_only", 3, 1),
            eval_row("p2", "few_shot_examples_only", 5, 4),
            eval_row("p3", "prompt_only", 4, 2),
            eval_row("p3", "few_shot_examples_only", 5, 4),
        ]

        summary = summarize_stratified_author_style_eval(
            eval_rows=eval_rows,
            pack_profiles=pack_profiles,
            manifest=manifest,
        )

        self.assertEqual(
            summary["schema_version"],
            "author-style-stratified-eval-summary/v1",
        )
        self.assertTrue(summary["artifact_boundary"]["must_not_use_for_induction"])
        self.assertEqual(
            summary["comparison"]["delta_direction"],
            "few_shot_examples_only - prompt_only",
        )
        self.assertEqual(summary["coverage"]["paired_cells"], 3)
        population_delta = summary["overall_unique_pack_effect"]["paired_deltas"][
            "few_shot_examples_only"
        ]
        self.assertEqual(population_delta["mean_style_likeness_delta"], 2.0)
        self.assertEqual(population_delta["style_delta_sign_counts"], {"positive": 3})
        question = next(
            stratum
            for stratum in summary["strata"]
            if stratum["stratum_id"] == "question_heavy"
        )
        self.assertTrue(question["paper_facing"])
        self.assertEqual(question["pack_count"], 2)
        self.assertEqual(
            question["paired_deltas"]["few_shot_examples_only"][
                "mean_style_likeness_delta"
            ],
            2.5,
        )
        macro = summary["macro_averages"]["paper_eligible_style_tag"][
            "few_shot_examples_only"
        ]
        self.assertEqual(summary["cluster_balanced_macro"], macro)
        self.assertEqual(macro["strata"], 2)
        self.assertEqual(macro["mean_of_mean_style_likeness_delta"], 2.25)
        self.assertEqual(macro["positive_style_delta_strata"], 2)
        self.assertEqual(
            summary["multi_label_counting"]["paper_eligible_style_tag"][
                "duplicate_factor"
            ],
            1.333,
        )
        self.assertEqual(
            summary["multi_label_counting"]["paper_eligible_style_tag"][
                "membership_count_distribution"
            ],
            {"2": 1, "1": 2},
        )
        topic_group = next(
            stratum
            for stratum in summary["strata"]
            if stratum["stratum_id"] == "has_topic_like_exclusion"
        )
        self.assertFalse(topic_group["paper_facing"])
        self.assertEqual(topic_group["pack_count"], 1)


if __name__ == "__main__":
    unittest.main()
