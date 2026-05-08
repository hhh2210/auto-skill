from __future__ import annotations

import unittest

from scripts.metrics.export_judge_disagreements import build_disagreement_packets


def row(mode: str, score: float, *, text: str = "candidate") -> dict:
    return {
        "pack_id": "pack-1",
        "task_id": "task-1",
        "mode": mode,
        "status": "success",
        "overall_score": score,
        "judge_model": "judge",
        "generation": {"text": text},
        "scores": {"quality": [{"score": score, "reason": "ok"}]},
    }


class ExportJudgeDisagreementsTests(unittest.TestCase):
    def test_exports_only_delta_sign_disagreements(self) -> None:
        left_rows = [
            row("prompt_only", 5, text="baseline"),
            row("skill", 4, text="skill output"),
            row("same", 6),
        ]
        right_rows = [
            row("prompt_only", 5, text="baseline"),
            row("skill", 7, text="skill output"),
            row("same", 8),
        ]

        packets = build_disagreement_packets(
            left_rows=left_rows,
            right_rows=right_rows,
            left_label="qwen",
            right_label="mimo",
            baseline_mode="prompt_only",
        )

        self.assertEqual(len(packets), 1)
        packet = packets[0]
        self.assertEqual(packet["mode"], "skill")
        self.assertEqual(packet["disagreement_kind"], "sign_flip")
        self.assertEqual(packet["left"]["delta"], -1)
        self.assertEqual(packet["right"]["delta"], 2)
        self.assertEqual(packet["candidate_output"], "skill output")
        self.assertEqual(packet["baseline_output"], "baseline")

    def test_respects_mode_filter(self) -> None:
        packets = build_disagreement_packets(
            left_rows=[row("prompt_only", 5), row("skill", 4)],
            right_rows=[row("prompt_only", 5), row("skill", 7)],
            left_label="qwen",
            right_label="mimo",
            baseline_mode="prompt_only",
            modes=["other"],
        )

        self.assertEqual(packets, [])

    def test_marks_tie_disagreement_separately(self) -> None:
        packets = build_disagreement_packets(
            left_rows=[row("prompt_only", 5), row("skill", 5)],
            right_rows=[row("prompt_only", 5), row("skill", 7)],
            left_label="qwen",
            right_label="mimo",
            baseline_mode="prompt_only",
        )

        self.assertEqual(packets[0]["disagreement_kind"], "sign_or_tie_disagreement")

    def test_rejects_output_mismatch_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical candidate outputs"):
            build_disagreement_packets(
                left_rows=[row("prompt_only", 5), row("skill", 4, text="left")],
                right_rows=[row("prompt_only", 5), row("skill", 7, text="right")],
                left_label="qwen",
                right_label="mimo",
                baseline_mode="prompt_only",
            )

    def test_rejects_output_mismatch_after_truncation_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical candidate outputs"):
            build_disagreement_packets(
                left_rows=[row("prompt_only", 5), row("skill", 4, text="prefix-left")],
                right_rows=[row("prompt_only", 5), row("skill", 7, text="prefix-right")],
                left_label="qwen",
                right_label="mimo",
                baseline_mode="prompt_only",
                max_output_chars=6,
            )

    def test_can_export_task_and_private_eval_context(self) -> None:
        packets = build_disagreement_packets(
            left_rows=[row("prompt_only", 5), row("skill", 4)],
            right_rows=[row("prompt_only", 5), row("skill", 7)],
            left_label="qwen",
            right_label="mimo",
            baseline_mode="prompt_only",
            packs=[
                {
                    "pack_id": "pack-1",
                    "source": "WritingBench",
                    "domain": {"primary": "demo"},
                    "heldout_tasks": [
                        {
                            "task_id": "task-1",
                            "source_task_id": "source-1",
                            "task_input": "Do the task",
                            "materials": [{"text": "material"}],
                        }
                    ],
                }
            ],
            private_rows=[
                {
                    "pack_id": "pack-1",
                    "heldout_private": [
                        {
                            "task_ref": "task-1",
                            "supervision": {"items": [{"name": "Quality"}]},
                            "judge": {"type": "llm"},
                        }
                    ],
                }
            ],
        )

        packet = packets[0]
        self.assertTrue(packet["same_candidate_output"])
        self.assertEqual(packet["task_context"]["task_input"], "Do the task")
        self.assertEqual(
            packet["private_eval_context"]["supervision"]["items"][0]["name"],
            "Quality",
        )


if __name__ == "__main__":
    unittest.main()
