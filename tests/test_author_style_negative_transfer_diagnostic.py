from __future__ import annotations

import unittest

from auto_skill.author_style_negative_transfer_diagnostic import (
    build_pack_diagnostics,
    summarize_negative_transfer_diagnostic,
)


class AuthorStyleNegativeTransferDiagnosticTests(unittest.TestCase):
    def test_builds_pack_and_stratum_diagnostics(self) -> None:
        pack_profiles = [
            {
                "pack_id": "p1",
                "eligible_style_tags": ["ellipsis_heavy"],
                "background_style_tags": ["casual_conversational"],
                "style_families": ["pacing_punctuation"],
                "quality_flags": ["low_style_extractability"],
                "metric_means": {"source_pool_margin": 0.2},
            },
            {
                "pack_id": "p2",
                "eligible_style_tags": ["profanity"],
                "background_style_tags": [],
                "style_families": ["register_voice"],
                "quality_flags": [],
                "metric_means": {"source_pool_margin": -0.1},
            },
        ]
        scalar_rows = [
            row("p1", "few_shot_examples_only", 8, 4),
            row("p1", "wrong_author_examples_same_source", 6, 2),
            row("p1", "prompt_only", 3, 0),
            row("p2", "few_shot_examples_only", 5, 3),
            row("p2", "wrong_author_examples_same_source", 7, 4),
            row("p2", "prompt_only", 2, 0),
        ]
        swap_audit = {
            "status_counts": {"success": 4},
            "by_candidate": {},
            "pairs": [
                {
                    "pack_id": "p1",
                    "task_id": "p1::heldout::0",
                    "status": "swap_stable_anchor_wins",
                },
                {
                    "pack_id": "p2",
                    "task_id": "p2::heldout::0",
                    "status": "swap_stable_candidate_wins",
                },
            ],
        }
        manifest = {
            "policy": "public_match",
            "pairs": [
                {"target_pack_id": "p1", "impostor_pack_id": "p2"},
                {"target_pack_id": "p2", "impostor_pack_id": "p1"},
            ],
        }

        pack_rows = build_pack_diagnostics(
            pack_profiles=pack_profiles,
            scalar_eval_sets={"judge": scalar_rows},
            swap_audit=swap_audit,
            negative_transfer_manifest=manifest,
        )
        summary = summarize_negative_transfer_diagnostic(
            pack_rows=pack_rows,
            scalar_labels=["judge"],
            negative_transfer_manifest=manifest,
            swap_audit=swap_audit,
        )

        self.assertEqual(pack_rows[0]["pairwise_verdict"], "target_stable_win")
        self.assertEqual(
            pack_rows[0]["scalar_eval"]["judge"]["target_minus_wrong_style_likeness"],
            2,
        )
        self.assertEqual(
            summary["population"]["pairwise_verdict_counts"],
            {"target_stable_win": 1, "wrong_author_stable_win": 1},
        )
        self.assertEqual(
            summary["population"]["scalar_summaries"]["judge"][
                "target_minus_wrong_style_sign_counts"
            ],
            {"target_higher": 1, "wrong_author_higher": 1},
        )


def row(pack_id: str, mode: str, score: int, wins: int) -> dict:
    return {
        "status": "success",
        "pack_id": pack_id,
        "task_id": f"{pack_id}::heldout::0",
        "mode": mode,
        "style_likeness_1_to_10": score,
        "candidate_beats_negatives": wins,
        "hard_negative_count": 4,
    }


if __name__ == "__main__":
    unittest.main()
