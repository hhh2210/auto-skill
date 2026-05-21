from __future__ import annotations

import unittest

from auto_skill.author_style_compression_diagnostic import (
    summarize_author_style_compression_failures,
)


def eval_row(
    pack_id: str,
    mode: str,
    *,
    score: float,
    wins: int,
    matched: list[str],
    missed: list[str],
) -> dict:
    return {
        "schema_version": "author-style-heldout-eval/v1",
        "status": "success",
        "pack_id": pack_id,
        "task_id": f"{pack_id}::heldout::0",
        "mode": mode,
        "solver_model": "solver",
        "solver_backend": "backend",
        "judge_model": "judge",
        "judge_backend": "judge_backend",
        "judge_config_prefix": None,
        "judge_config_model": "judge",
        "style_likeness_1_to_10": score,
        "candidate_beats_negatives": wins,
        "hard_negative_count": 4,
        "judge_report": {
            "style_likeness_1_to_10": score,
            "candidate_beats_negatives": wins,
            "style_tags_matched": matched,
            "style_tags_missed": missed,
        },
    }


class AuthorStyleCompressionDiagnosticTests(unittest.TestCase):
    def test_summarizes_loss_tags_without_raw_generation_text(self) -> None:
        eval_rows = [
            eval_row(
                "pack-a",
                "prompt_only",
                score=2,
                wins=0,
                matched=["casual register"],
                missed=["question-heavy rhythm", "ellipses"],
            ),
            eval_row(
                "pack-a",
                "few_shot_examples_only",
                score=8,
                wins=4,
                matched=["question-heavy rhythm", "ellipses", "personal anecdote"],
                missed=[],
            ),
            eval_row(
                "pack-a",
                "auto_skill_feature_driven",
                score=5,
                wins=2,
                matched=["casual register"],
                missed=["misses question-heavy rhythm", "no ellipses", "too polished"],
            ),
        ]
        skill_rows = [
            {
                "schema_version": "skill-induction/v1",
                "status": "success",
                "pack_id": "pack-a",
                "mode": "auto_skill_feature_driven_no_validation",
                "skill_md": "Use a casual voice with personal anecdotes.",
                "feature_reports": [{"style_features": {"ellipsis": "present"}}],
                "cross_example_report": {"stable_features": [{"feature": "question-heavy"}]},
            }
        ]
        pack_profiles = [
            {
                "schema_version": "author-style-pack-cluster-profile/v1",
                "pack_id": "pack-a",
                "paper_quality_tier": "cross_topic_high_signal",
                "paper_plot_role": "main_cross_topic",
                "canonical_style_tags": [
                    "question_heavy",
                    "ellipsis_heavy",
                    "first_person_diary",
                ],
                "eligible_style_tags": ["question_heavy", "ellipsis_heavy"],
                "style_families": ["pacing_punctuation"],
                "hard_negatives": {"mean_style_confusability_1_to_5": 4.5},
                "source_pool": {"margin_vs_source_pool_impostors": 0.1},
            }
        ]

        summary = summarize_author_style_compression_failures(
            eval_rows=eval_rows,
            skill_rows=skill_rows,
            pack_profiles=pack_profiles,
            treatment_modes=["auto_skill_feature_driven"],
        )

        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["missing_pairs"], {})
        mode = summary["modes"]["auto_skill_feature_driven"]
        self.assertEqual(mode["mean_style_likeness_delta"], -3.0)
        self.assertEqual(mode["mean_compression_retention_ratio_vs_prompt_only"], 0.5)
        self.assertEqual(mode["loss_bucket_counts"]["severe_loss"], 1)
        self.assertEqual(
            mode["top_candidate_missed_canonical_tags_on_losses"]["question_heavy"],
            1,
        )
        self.assertEqual(
            mode["top_feature_pipeline_hits_not_compiled_to_skill_md"]["ellipsis_heavy"],
            1,
        )
        per_pack = summary["per_pack"][0]
        self.assertIn("question_heavy", per_pack["baseline_matched_canonical_tags"])
        self.assertIn("ellipsis_heavy", per_pack["canonical_tags_missing_from_skill_md"])
        self.assertEqual(per_pack["compression_retention_ratio_vs_prompt_only"], 0.5)
        self.assertNotIn("generation", per_pack)
        self.assertIn("generation_shape", per_pack)
        self.assertTrue(summary["artifact_boundary"]["must_not_use_for_induction"])


if __name__ == "__main__":
    unittest.main()
