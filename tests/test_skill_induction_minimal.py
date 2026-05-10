from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any

from auto_skill.schemas import UserExample, validate_skill_row
from auto_skill.skill_induction_minimal import (
    MinimalInductionError,
    build_extraction_prompt,
    build_revision_prompt,
    build_supervisor_prompt,
    induce_minimal,
    validate_supervisor_report,
)
from auto_skill.skill_memory import SkillMemoryEntry


@dataclass
class MockResult:
    text: str
    model: str
    finish_reason: str | None = "stop"
    usage: dict[str, int] | None = None
    request_id: str | None = "req"


class MockClient:
    def __init__(
        self,
        model: str,
        texts: list[str],
        finish_reasons: list[str | None] | None = None,
    ):
        self.model = model
        self.texts = list(texts)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(texts))
        self.prompts: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **_: Any) -> MockResult:
        self.prompts.append(messages)
        return MockResult(
            self.texts.pop(0),
            self.model,
            finish_reason=self.finish_reasons.pop(0),
            usage={"total_tokens": 1},
        )


class MinimalInductionTests(unittest.TestCase):
    def test_extraction_prompt_injects_positive_negative_and_empty_memory(self) -> None:
        examples = [self.example("ex-1", pack_id="pack-a")]
        both = build_extraction_prompt(
            examples,
            [
                self.memory("m1", "pack-a", "stable_feature", "Use bullets.", "positive"),
                self.memory("m2", "pack-a", "outlier", "Do not copy names.", "negative"),
            ],
            "within_pack",
        )
        self.assertIn("Reusable patterns from prior packs", both)
        self.assertIn("[pack-a/stable_feature] Use bullets.", both)
        self.assertIn("Past failure modes — do not generalize", both)
        self.assertIn("[pack-a/outlier] Do not copy names.", both)

        empty = build_extraction_prompt(examples, [], "within_pack")
        self.assertIn("- None", empty)

    def test_extraction_prompt_filters_within_pack_but_allows_cross_pack(self) -> None:
        examples = [self.example("ex-1", pack_id="pack-a")]
        entries = [
            self.memory("m1", "pack-a", "candidate_rule", "Match tone.", "positive"),
            self.memory("m2", "pack-b", "candidate_rule", "Use headings.", "positive"),
        ]
        within = build_extraction_prompt(examples, entries, "within_pack")
        cross = build_extraction_prompt(examples, entries, "cross_pack")

        self.assertIn("Match tone.", within)
        self.assertNotIn("Use headings.", within)
        self.assertIn("Use headings.", cross)

    def test_extraction_prompt_fails_closed_when_within_pack_has_no_pack_id(self) -> None:
        examples = [self.example("ex-1", pack_id="")]
        entries = [self.memory("m1", "pack-b", "candidate_rule", "Use headings.", "positive")]

        within = build_extraction_prompt(examples, entries, "within_pack")
        cross = build_extraction_prompt(examples, entries, "cross_pack")

        self.assertNotIn("Use headings.", within)
        self.assertIn("Use headings.", cross)

    def test_supervisor_and_revision_prompts_include_required_contract(self) -> None:
        examples = [self.example("ex-1")]
        supervisor = build_supervisor_prompt("# Candidate", examples)
        self.assertIn("artifact_critique", supervisor)
        self.assertIn("rule_grounding", supervisor)
        self.assertIn("supported_by", supervisor)
        self.assertIn("Do not hold out examples", supervisor)

        revision = build_revision_prompt("# Candidate", {"artifact_critique": {"missing": []}})
        self.assertIn("# Candidate", revision)
        self.assertIn("Supervisor report JSON", revision)

    def test_induce_minimal_returns_skill_row_shape(self) -> None:
        supervisor_report = {
            "artifact_critique": {"too_generic": [], "over_specific": [], "missing": []},
            "rule_grounding": [
                {
                    "rule": "Use concise bullets.",
                    "supported_by": ["ex-1"],
                    "contradicted_by": [],
                    "out_of_scope": [],
                }
            ],
        }
        solver = MockClient("solver-model", ["# Candidate", "# Final Skill"])
        supervisor = MockClient("mimo-model", [json.dumps(supervisor_report)])

        row = induce_minimal(
            self.pack(),
            solver_client=solver,
            supervisor_client=supervisor,
            memory=[],
            scope="within_pack",
        )

        validate_skill_row(row)
        self.assertEqual(row["mode"], "auto_skill_minimal")
        self.assertEqual(row["skill_md"], "# Final Skill")
        self.assertEqual(row["solver_model"], "solver-model")
        self.assertEqual(row["supervisor_model"], "mimo-model")
        self.assertEqual(row["supervisor_report"], supervisor_report)
        self.assertEqual([call["stage"] for call in row["model_calls"]], [
            "minimal_extraction",
            "minimal_supervisor",
            "minimal_revision",
        ])

    def test_induce_minimal_rejects_truncated_model_response(self) -> None:
        supervisor_report = {
            "artifact_critique": {"too_generic": [], "over_specific": [], "missing": []},
            "rule_grounding": [],
        }
        solver = MockClient("solver-model", ["# Candidate"], finish_reasons=["length"])
        supervisor = MockClient("mimo-model", [json.dumps(supervisor_report)])

        with self.assertRaisesRegex(MinimalInductionError, "minimal_extraction"):
            induce_minimal(
                self.pack(),
                solver_client=solver,
                supervisor_client=supervisor,
                memory=[],
                scope="within_pack",
            )

    def test_supervisor_report_validation_fails_closed(self) -> None:
        with self.assertRaises(MinimalInductionError):
            validate_supervisor_report({"artifact_critique": {}})

        with self.assertRaises(MinimalInductionError):
            validate_supervisor_report(
                {
                    "artifact_critique": {
                        "too_generic": [],
                        "over_specific": [],
                        "missing": [],
                    },
                    "rule_grounding": [{"rule": "Missing lists"}],
                }
            )

    @staticmethod
    def example(example_id: str, pack_id: str = "pack-a") -> UserExample:
        return UserExample(
            example_id=example_id,
            task_input="Write a brief update.",
            output="A concise update.",
            metadata={"pack_id": pack_id},
        )

    @staticmethod
    def memory(
        memory_id: str,
        pack_id: str,
        lesson_kind: str,
        lesson: str,
        polarity: str,
    ) -> SkillMemoryEntry:
        return SkillMemoryEntry(
            memory_id=memory_id,
            pack_id=pack_id,
            mode="auto_skill_feature_driven_no_validation",
            lesson_kind=lesson_kind,
            lesson=lesson,
            evidence_examples=("ex-1",),
            polarity=polarity,  # type: ignore[arg-type]
        )

    @staticmethod
    def pack() -> dict[str, object]:
        return {
            "pack_id": "pack-a",
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "task_input": "Write a brief update.",
                    "desired_output": {"status": "generated", "text": "A concise update."},
                    "materials": [],
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
