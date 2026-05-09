from __future__ import annotations

import unittest

from scripts.metrics.report_example_contamination import (
    build_contamination_report,
    text_ngrams,
)


class ExampleContaminationTests(unittest.TestCase):
    def test_text_ngrams_tokenizes_english_and_cjk(self) -> None:
        phrases = text_ngrams("Alpha beta gamma 中文测试", n=3)

        self.assertIn("alpha beta gamma", phrases)
        self.assertIn("中 文 测", phrases)

    def test_reports_train_only_phrase_copied_to_candidate(self) -> None:
        packs = [
            {
                "pack_id": "pack-1",
                "train_examples": [
                    {
                        "task_input": "Write about a toy store.",
                        "desired_output": {
                            "status": "generated",
                            "text": "Use the unique phrase traffic security platform.",
                        },
                    }
                ],
                "heldout_tasks": [
                    {
                        "task_id": "task-1",
                        "task_input": "Write about parent education consulting.",
                        "materials": [],
                    }
                ],
            }
        ]
        eval_rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "prompt_only",
                "status": "success",
                "generation": {"text": "No copied phrase here."},
            },
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "skill",
                "status": "success",
                "generation": {
                    "text": "The memo mentions traffic security platform by mistake."
                },
            }
        ]

        report = build_contamination_report(
            packs=packs,
            eval_rows=eval_rows,
            n=3,
            top_k=5,
        )

        self.assertEqual(len(report), 2)
        skill_row = next(row for row in report if row["mode"] == "skill")
        self.assertGreater(skill_row["suspicious_train_only_candidate_ngrams"], 0)
        self.assertGreater(skill_row["excess_suspicious_ngrams_vs_baseline"], 0)
        self.assertIn(
            "traffic security platform",
            {item["phrase"] for item in skill_row["top_matches"]},
        )

    def test_ignores_phrase_present_in_heldout_task(self) -> None:
        packs = [
            {
                "pack_id": "pack-1",
                "train_examples": [
                    {
                        "task_input": "Write about parent education consulting.",
                        "desired_output": {
                            "status": "generated",
                            "text": "parent education consulting",
                        },
                    }
                ],
                "heldout_tasks": [
                    {
                        "task_id": "task-1",
                        "task_input": "Need parent education consulting advice.",
                        "materials": [],
                    }
                ],
            }
        ]
        eval_rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "skill",
                "status": "success",
                "generation": {"text": "parent education consulting advice"},
            }
        ]

        report = build_contamination_report(packs=packs, eval_rows=eval_rows, n=3)

        self.assertEqual(report[0]["suspicious_train_only_candidate_ngrams"], 0)

    def test_train_material_paths_are_user_visible_example_text(self) -> None:
        packs = [
            {
                "pack_id": "pack-1",
                "train_examples": [
                    {
                        "task_input": "Write with a source file.",
                        "materials": [
                            {
                                "path": "materials/special traffic security platform.pdf",
                            }
                        ],
                        "desired_output": {"status": "generated", "text": "Output."},
                    }
                ],
                "heldout_tasks": [
                    {
                        "task_id": "task-1",
                        "task_input": "Write unrelated advice.",
                        "materials": [],
                    }
                ],
            }
        ]
        eval_rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "skill",
                "status": "success",
                "generation": {"text": "Mentions traffic security platform."},
            }
        ]

        report = build_contamination_report(packs=packs, eval_rows=eval_rows, n=3)

        self.assertIn(
            "traffic security platform",
            {item["phrase"] for item in report[0]["top_matches"]},
        )

    def test_train_material_text_is_user_visible_example_text(self) -> None:
        packs = [
            {
                "pack_id": "pack-1",
                "train_examples": [
                    {
                        "task_input": "Write with a source file.",
                        "materials": [
                            {
                                "path": "materials/source.pdf",
                                "text": "source-only carbon audit protocol",
                            }
                        ],
                        "desired_output": {"status": "generated", "text": "Output."},
                    }
                ],
                "heldout_tasks": [
                    {
                        "task_id": "task-1",
                        "task_input": "Write unrelated advice.",
                        "materials": [],
                    }
                ],
            }
        ]
        eval_rows = [
            {
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "skill",
                "status": "success",
                "generation": {"text": "Mentions carbon audit protocol."},
            }
        ]

        report = build_contamination_report(packs=packs, eval_rows=eval_rows, n=3)

        self.assertIn(
            "carbon audit protocol",
            {item["phrase"] for item in report[0]["top_matches"]},
        )


if __name__ == "__main__":
    unittest.main()
