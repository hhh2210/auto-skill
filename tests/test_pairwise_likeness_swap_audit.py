from __future__ import annotations

import unittest

from auto_skill.pairwise_likeness import swap_flip_audit

ANCHOR = "few_shot_examples_only"


def _row(
    *,
    task_id: str,
    candidate_mode: str = "ours_no_validation",
    order: str,
    winner: str,
    status: str = "success",
    confidence: str = "high",
) -> dict[str, object]:
    return {
        "pack_id": "pack",
        "task_id": task_id,
        "anchor_mode": ANCHOR,
        "candidate_mode": candidate_mode,
        "order": order,
        "status": status,
        "winner_mode": winner,
        "pairwise_report": {"confidence": confidence},
    }


def _pair(
    *,
    task_id: str,
    first_winner: str,
    second_winner: str,
    candidate_mode: str = "ours_no_validation",
) -> list[dict[str, object]]:
    return [
        _row(
            task_id=task_id,
            candidate_mode=candidate_mode,
            order="anchor_first",
            winner=first_winner,
            confidence="low",
        ),
        _row(
            task_id=task_id,
            candidate_mode=candidate_mode,
            order="candidate_first",
            winner=second_winner,
            confidence="medium",
        ),
    ]


class PairwiseSwapAuditTests(unittest.TestCase):
    def test_classifies_stable_candidate_anchor_and_tie_pairs(self) -> None:
        rows = [
            *_pair(
                task_id="candidate",
                first_winner="ours_no_validation",
                second_winner="ours_no_validation",
            ),
            *_pair(task_id="anchor", first_winner=ANCHOR, second_winner=ANCHOR),
            *_pair(task_id="tie", first_winner="tie", second_winner="tie"),
        ]
        audit = swap_flip_audit(rows)
        counts = audit["by_candidate"]["ours_no_validation"]["counts"]
        self.assertEqual(counts["swap_stable_candidate_wins"], 1)
        self.assertEqual(counts["swap_stable_anchor_wins"], 1)
        self.assertEqual(counts["swap_stable_tie"], 1)
        self.assertEqual(
            audit["by_candidate"]["ours_no_validation"][
                "swap_stable_candidate_win_rate_decisive"
            ],
            0.5,
        )

    def test_classifies_decisive_flip(self) -> None:
        audit = swap_flip_audit(
            _pair(task_id="flip", first_winner="ours_no_validation", second_winner=ANCHOR)
        )
        entry = audit["by_candidate"]["ours_no_validation"]
        self.assertEqual(entry["counts"]["swap_flip_decisive"], 1)
        self.assertEqual(entry["decisive_pair_count"], 1)
        self.assertEqual(entry["swap_flip_rate_decisive_pairs"], 1.0)

    def test_classifies_one_decisive_one_tie(self) -> None:
        audit = swap_flip_audit(
            _pair(task_id="mixed", first_winner="ours_no_validation", second_winner="tie")
        )
        counts = audit["by_candidate"]["ours_no_validation"]["counts"]
        self.assertEqual(counts["swap_one_decisive_one_tie"], 1)
        self.assertEqual(
            audit["by_candidate"]["ours_no_validation"]["decisive_pair_count"],
            0,
        )

    def test_classifies_missing_and_non_success_as_incomplete(self) -> None:
        rows = [
            _row(
                task_id="missing",
                order="anchor_first",
                winner="ours_no_validation",
            ),
            *_pair(task_id="parse_error", first_winner="ours_no_validation", second_winner=ANCHOR),
        ]
        rows[-1]["status"] = "judge_parse_error"
        audit = swap_flip_audit(rows)
        counts = audit["by_candidate"]["ours_no_validation"]["counts"]
        self.assertEqual(counts["swap_pair_incomplete"], 2)
        self.assertEqual(
            audit["by_candidate"]["ours_no_validation"]["confidence_by_pair_status"][
                "swap_pair_incomplete"
            ]["unknown"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
