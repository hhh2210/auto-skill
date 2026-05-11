from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_skill.example_packs import write_jsonl
from auto_skill.extraction_memory import SCHEMA_VERSION
from auto_skill.skill_memory import (
    format_memory_for_prompt,
    load_memory_for_pack,
    polarity_for_lesson_kind,
    select_memory_entries,
)


class SkillMemoryTests(unittest.TestCase):
    def test_polarity_mapping_from_lesson_kind(self) -> None:
        self.assertEqual(polarity_for_lesson_kind("stable_feature"), "positive")
        self.assertEqual(polarity_for_lesson_kind("candidate_rule"), "positive")
        self.assertEqual(polarity_for_lesson_kind("optional_feature"), "positive")
        self.assertEqual(polarity_for_lesson_kind("do_not_generalize"), "negative")
        self.assertEqual(polarity_for_lesson_kind("outlier"), "negative")
        self.assertEqual(polarity_for_lesson_kind("conflict"), "negative")

    def test_load_filters_by_scope_and_formats_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            write_jsonl(
                path,
                [
                    self._row("m1", "pack-a", "stable_feature", "Use concise bullets."),
                    self._row("m2", "pack-a", "do_not_generalize", "Do not copy names."),
                    self._row("m3", "pack-b", "candidate_rule", "Start with outcome."),
                    {
                        **self._row("m4", "pack-a", "candidate_rule", "Private rule."),
                        "evidence_source": "rubric",
                    },
                ],
            )

            within = load_memory_for_pack(path, "pack-a", "within_pack")
            cross = load_memory_for_pack(path, "pack-a", "cross_pack")
            holdout = load_memory_for_pack(path, "pack-a", "cross_pack_holdout")

            self.assertEqual([entry.memory_id for entry in within], ["m1", "m2"])
            self.assertEqual([entry.memory_id for entry in cross], ["m1", "m2", "m3"])
            self.assertEqual([entry.memory_id for entry in holdout], ["m3"])
            formatted = format_memory_for_prompt(within)
            self.assertIn("Positive priors", formatted)
            self.assertIn("[pack-a/stable_feature] Use concise bullets.", formatted)
            self.assertIn("Negative warnings", formatted)
            self.assertIn("[pack-a/do_not_generalize] Do not copy names.", formatted)

    def test_format_empty_memory(self) -> None:
        self.assertEqual(format_memory_for_prompt([]), "No prior extraction memory.")

    def test_select_memory_entries_prefers_more_evidence_then_positive(self) -> None:
        entries = [
            self._entry("m-neg-2", "pack-a", "conflict", "negative two", ["ex-1", "ex-2"]),
            self._entry("m-pos-1", "pack-a", "candidate_rule", "positive one", ["ex-1"]),
            self._entry("m-pos-2", "pack-a", "candidate_rule", "positive two", ["ex-1", "ex-2"]),
        ]

        selected = select_memory_entries(entries, 2)

        self.assertEqual([entry.memory_id for entry in selected], ["m-pos-2", "m-neg-2"])

    @staticmethod
    def _row(memory_id: str, pack_id: str, lesson_kind: str, lesson: str) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "memory_id": memory_id,
            "pack_id": pack_id,
            "mode": "auto_skill_feature_driven_no_validation",
            "lesson_kind": lesson_kind,
            "lesson": lesson,
            "evidence_examples": ["ex-1"],
            "evidence_source": "user_examples",
        }

    @classmethod
    def _entry(
        cls,
        memory_id: str,
        pack_id: str,
        lesson_kind: str,
        lesson: str,
        evidence_examples: list[str],
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            row = cls._row(memory_id, pack_id, lesson_kind, lesson)
            row["evidence_examples"] = evidence_examples
            write_jsonl(path, [row])
            return load_memory_for_pack(path, pack_id, "within_pack")[0]


if __name__ == "__main__":
    unittest.main()
