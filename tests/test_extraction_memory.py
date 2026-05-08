from __future__ import annotations

import unittest

from auto_skill.extraction_memory import (
    entries_from_skill_row,
    entries_from_skill_rows,
    summarize_memory_entries,
)


class ExtractionMemoryTests(unittest.TestCase):
    def test_extracts_public_example_lessons_from_skill_row(self) -> None:
        row = {
            "pack_id": "pack",
            "mode": "auto_skill_feature_driven_no_validation",
            "status": "success",
            "cross_example_report": {
                "stable_features": ["Use numbered headings"],
                "candidate_rules": [
                    {
                        "rule": "Ground every metric in the input.",
                        "supporting_example_ids": ["ex1", "ex2"],
                    }
                ],
                "conflicts": [
                    {
                        "feature": "format",
                        "description": "Reports and press releases need separate structure rules.",
                    }
                ],
            },
            "feature_reports": [
                {
                    "example_id": "ex1",
                    "must_not_generalize": ["Do not copy example-specific company names."],
                }
            ],
        }

        entries = entries_from_skill_row(row)
        lessons = {(entry.lesson_kind, entry.lesson) for entry in entries}

        self.assertIn(("stable_feature", "Use numbered headings"), lessons)
        self.assertIn(("candidate_rule", "Ground every metric in the input."), lessons)
        self.assertIn(
            ("conflict", "format: Reports and press releases need separate structure rules."),
            lessons,
        )
        self.assertIn(
            ("do_not_generalize", "Do not copy example-specific company names."),
            lessons,
        )
        candidate = next(entry for entry in entries if entry.lesson_kind == "candidate_rule")
        self.assertEqual(candidate.evidence_examples, ("ex1", "ex2"))

    def test_deduplicates_and_summarizes_entries(self) -> None:
        row = {
            "pack_id": "pack",
            "mode": "auto_skill_feature_driven_no_validation",
            "status": "success",
            "cross_example_report": {"stable_features": ["Use numbered headings"]},
        }

        entries = entries_from_skill_rows([row, row])
        summary = summarize_memory_entries(entries)

        self.assertEqual(len(entries), 1)
        self.assertEqual(summary["entries"], 1)
        self.assertEqual(summary["lesson_kinds"], {"stable_feature": 1})


if __name__ == "__main__":
    unittest.main()
