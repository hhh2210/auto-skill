from __future__ import annotations

import unittest

from auto_skill.extraction_memory import (
    DERIVATION_FEATURE_REPORTS,
    DERIVATION_MINIMAL_SUPERVISOR_REPORT,
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
        self.assertEqual(candidate.derivation, DERIVATION_FEATURE_REPORTS)

    def test_extracts_minimal_supervisor_memory_deterministically(self) -> None:
        row = {
            "pack_id": "pack",
            "mode": "auto_skill_minimal",
            "status": "success",
            "supervisor_report": {
                "artifact_critique": {
                    "too_generic": ["This should not become memory."],
                    "over_specific": ["Do not copy the company name."],
                    "missing": ["This prescriptive item is skipped."],
                },
                "rule_grounding": [
                    {
                        "rule": "Ground every metric in the input.",
                        "supported_by": ["ex1", "ex2"],
                        "contradicted_by": ["ex3"],
                        "out_of_scope": ["ex4"],
                    }
                ],
            },
        }

        entries = entries_from_skill_row(row)
        lessons = {(entry.lesson_kind, entry.lesson, entry.evidence_examples) for entry in entries}

        self.assertEqual(len(entries), 4)
        self.assertIn(
            ("candidate_rule", "Ground every metric in the input.", ("ex1", "ex2")),
            lessons,
        )
        self.assertIn(("conflict", "Ground every metric in the input.", ("ex3",)), lessons)
        self.assertIn(("outlier", "Ground every metric in the input.", ("ex4",)), lessons)
        self.assertIn(("do_not_generalize", "Do not copy the company name.", ()), lessons)
        self.assertTrue(
            all(entry.derivation == DERIVATION_MINIMAL_SUPERVISOR_REPORT for entry in entries)
        )

    def test_minimal_supervisor_memory_handles_missing_report(self) -> None:
        self.assertEqual(
            entries_from_skill_row(
                {
                    "pack_id": "pack",
                    "mode": "auto_skill_minimal",
                    "status": "success",
                    "supervisor_report": None,
                }
            ),
            [],
        )

    def test_minimal_supervisor_memory_parses_string_report(self) -> None:
        row = {
            "pack_id": "pack",
            "mode": "auto_skill_minimal",
            "status": "success",
            "supervisor_report": (
                '{"artifact_critique": {}, "rule_grounding": '
                '[{"rule": "Use concise bullets.", "supported_by": ["ex1"], '
                '"contradicted_by": [], "out_of_scope": []}]}'
            ),
        }

        entries = entries_from_skill_row(row)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].lesson_kind, "candidate_rule")
        self.assertEqual(entries[0].evidence_examples, ("ex1",))

    def test_minimal_supervisor_memory_skips_unanchored_grounding(self) -> None:
        row = {
            "pack_id": "pack",
            "mode": "auto_skill_minimal",
            "status": "success",
            "supervisor_report": {
                "artifact_critique": {},
                "rule_grounding": [
                    {
                        "rule": "Use concise bullets.",
                        "supported_by": [],
                        "contradicted_by": [],
                        "out_of_scope": [],
                    }
                ],
            },
        }

        self.assertEqual(entries_from_skill_row(row), [])

    def test_minimal_supervisor_memory_handles_non_object_critique(self) -> None:
        row = {
            "pack_id": "pack",
            "mode": "auto_skill_minimal",
            "status": "success",
            "supervisor_report": {
                "artifact_critique": [],
                "rule_grounding": [
                    {
                        "rule": "Use concise bullets.",
                        "supported_by": ["ex1"],
                        "contradicted_by": [],
                        "out_of_scope": [],
                    }
                ],
            },
        }

        entries = entries_from_skill_row(row)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].lesson_kind, "candidate_rule")

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
