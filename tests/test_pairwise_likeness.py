from __future__ import annotations

import unittest

from auto_skill.pairwise_likeness import (
    build_pairwise_likeness_prompt,
    parse_pairwise_likeness_report,
    summarize_pairwise_rows,
    truncate_text,
    winner_mode,
)
from auto_skill.schemas import UserExample


def _example(example_id: str = "ex1") -> UserExample:
    return UserExample(
        example_id=example_id,
        task_input="Write a structured report with specific operational details.",
        output="## Summary\nConcrete detail and clear sections.",
    )


def _row(
    *,
    task_id: str,
    candidate_mode: str,
    winner: str,
    order: str = "anchor_first",
    status: str = "success",
) -> dict[str, object]:
    return {
        "pack_id": "p",
        "task_id": task_id,
        "anchor_mode": "few_shot_examples_only",
        "candidate_mode": candidate_mode,
        "order": order,
        "status": status,
        "winner_mode": winner,
    }


class PairwiseLikenessTests(unittest.TestCase):
    def test_prompt_is_blind_and_contains_examples_task_outputs(self) -> None:
        prompt = build_pairwise_likeness_prompt(
            examples=[_example()],
            heldout_task={"task_input": "Heldout task"},
            output_a="A output",
            output_b="B output",
        )
        self.assertIn("User examples:", prompt)
        self.assertIn("Heldout task:", prompt)
        self.assertIn("Output A:", prompt)
        self.assertIn("Output B:", prompt)
        self.assertNotIn("few_shot_examples_only", prompt)
        self.assertNotIn("ours_no_validation", prompt)

    def test_parse_report_validates_enum_fields(self) -> None:
        report = parse_pairwise_likeness_report(
            '{"winner":"A","confidence":"high",'
            '"pattern_fidelity_winner":"B","task_fit_winner":"tie",'
            '"rationale":"A is better overall."}'
        )
        self.assertNotIn("parse_error", report)
        self.assertEqual(report["winner"], "A")

    def test_parse_report_rejects_bad_winner(self) -> None:
        report = parse_pairwise_likeness_report(
            '{"winner":"C","confidence":"high",'
            '"pattern_fidelity_winner":"B","task_fit_winner":"tie",'
            '"rationale":"bad"}'
        )
        self.assertEqual(report["parse_error"], "winner_must_be_A_B_or_tie")

    def test_winner_mode_maps_position_to_mode(self) -> None:
        self.assertEqual(
            winner_mode({"winner": "B"}, mode_a="anchor", mode_b="candidate"),
            "candidate",
        )
        self.assertEqual(
            winner_mode({"winner": "tie"}, mode_a="anchor", mode_b="candidate"),
            "tie",
        )

    def test_summarize_pairwise_rows_counts_call_and_task_levels(self) -> None:
        rows = [
            _row(task_id="t1", candidate_mode="ours", winner="ours"),
            _row(
                task_id="t1",
                candidate_mode="ours",
                winner="ours",
                order="candidate_first",
            ),
            _row(task_id="t2", candidate_mode="ours", winner="few_shot_examples_only"),
            _row(
                task_id="t2",
                candidate_mode="ours",
                winner="tie",
                order="candidate_first",
            ),
            _row(task_id="t3", candidate_mode="ours", winner="", status="model_error"),
        ]
        summary = summarize_pairwise_rows(rows)
        call = summary["call_level"]["ours"]
        self.assertEqual(call["success_calls"], 4)
        self.assertEqual(call["candidate_wins"], 2)
        self.assertEqual(call["anchor_wins"], 1)
        self.assertEqual(call["ties"], 1)
        paired = summary["paired_task_level"]["ours"]
        self.assertEqual(paired["paired_tasks"], 2)
        self.assertEqual(paired["candidate_wins"], 1)
        self.assertEqual(paired["anchor_wins"], 1)

    def test_truncate_text_marks_omitted_chars(self) -> None:
        self.assertEqual(truncate_text("abc", 5), "abc")
        self.assertIn("[truncated", truncate_text("abcdef", 3))


if __name__ == "__main__":
    unittest.main()
