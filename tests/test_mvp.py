from __future__ import annotations

import unittest

from auto_skill.mvp import (
    build_feature_signature_context,
    build_heldout_generation_prompt,
    build_judge_prompt,
    build_validation_aware_skill_merge_prompt,
    evaluation_criteria,
    extract_overall_score,
    parse_json_object,
    user_examples_from_pack,
)


class MVPTests(unittest.TestCase):
    def test_user_examples_from_pack_uses_generated_outputs_only(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "task_input": "Write a report.",
                    "desired_output": {"status": "generated", "text": "Report text."},
                    "materials": [{"path": "data/a.txt"}],
                },
                {
                    "example_id": "ex-2",
                    "task_input": "Write another report.",
                    "desired_output": {"status": "needs_generation"},
                    "materials": [],
                },
            ]
        }

        examples = user_examples_from_pack(pack)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].example_id, "ex-1")
        self.assertEqual(examples[0].output, "Report text.")
        self.assertEqual(examples[0].materials, ("data/a.txt",))

    def test_parse_json_object_extracts_wrapped_json(self) -> None:
        parsed = parse_json_object('prefix {"overall_score": 8, "rationale": "ok"} suffix')

        self.assertEqual(parsed["overall_score"], 8)

    def test_parse_json_object_extracts_fenced_json(self) -> None:
        parsed = parse_json_object('```json\n{"overall_score": 8, "rationale": "ok"}\n```')

        self.assertEqual(parsed["rationale"], "ok")

    def test_extract_overall_score_accepts_numeric_string(self) -> None:
        self.assertEqual(extract_overall_score({"overall_score": "7.5"}), 7.5)

    def test_extract_overall_score_rejects_boolean(self) -> None:
        self.assertIsNone(extract_overall_score({"overall_score": True}))

    def test_judge_prompt_marks_rubric_as_eval_only(self) -> None:
        prompt = build_judge_prompt(
            task={"task_input": "Write a report."},
            candidate_output="Report.",
            private_eval={"supervision": {"items": [{"name": "Depth"}]}},
        )

        self.assertIn("only for evaluation", prompt)
        self.assertIn("surrogate judge", prompt)
        self.assertIn("Depth", prompt)

    def test_judge_prompt_fails_closed_without_criteria(self) -> None:
        with self.assertRaisesRegex(ValueError, "no rubric/checklist"):
            build_judge_prompt(
                task={"task_input": "Write a report."},
                candidate_output="Report.",
                private_eval={"supervision": {"type": "empty"}},
            )

    def test_evaluation_criteria_flattens_presentbench_checklists(self) -> None:
        criteria = evaluation_criteria(
            {
                "supervision": {
                    "checklists": {
                        "material_independent_checklist_1": ["A", {"func": "check"}],
                    }
                }
            }
        )

        self.assertEqual(len(criteria), 2)
        self.assertEqual(criteria[0]["group"], "material_independent_checklist_1")

    def test_validation_merge_prompt_does_not_take_full_example_base_skill(self) -> None:
        prompt = build_validation_aware_skill_merge_prompt(
            candidate_skills=[{"validation_example_id": "ex-1", "skill_md": "Skill A"}],
            leave_one_out_reports=[{"validation_example_id": "ex-1"}],
        )

        self.assertIn("n-1 user-visible examples", prompt)
        self.assertNotIn("Base SKILL.md", prompt)

    def test_heldout_generation_prompt_prioritizes_final_deliverable(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={
                "task_id": "task-1",
                "task_input": "Write the complete consulting memo.",
                "materials": [],
            },
            mode="examples_plus_feature_skill",
            examples=[],
            skill_md="Use sections from examples.",
        )

        self.assertIn("Produce the completed artifact requested", prompt)
        self.assertIn("Do not answer with a plan, outline, checklist", prompt)
        self.assertIn("higher priority than", prompt)
        self.assertIn("not import example-specific facts", prompt)

    def test_feature_signature_context_uses_structured_example_features(self) -> None:
        context = build_feature_signature_context(
            {
                "cross_example_report": {
                    "stable_features": ["Use numbered sections"],
                    "candidate_rules": [
                        {
                            "rule": "Keep action items concrete",
                            "supporting_examples": ["ex-1", "ex-2"],
                        }
                    ],
                    "conflicts": [
                        {
                            "feature": "length",
                            "description": "Some examples are brief and some are long.",
                        }
                    ],
                    "outliers": [
                        {"example_id": "ex-3", "reason": "Uses a special audience."}
                    ],
                }
            }
        )

        assert context is not None
        self.assertIn("Abstract Feature Signatures", context)
        self.assertIn("Use numbered sections", context)
        self.assertIn("Keep action items concrete (support=2)", context)
        self.assertIn("not private rubrics", context)

    def test_feature_signature_context_returns_none_without_cross_report(self) -> None:
        self.assertIsNone(build_feature_signature_context({"skill_md": "# Skill"}))

    def test_examples_plus_feature_signatures_includes_examples(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={"task_id": "task-1", "task_input": "Write a memo.", "materials": []},
            mode="examples_plus_feature_signatures",
            examples=[],
            skill_md="# Abstract Feature Signatures",
        )

        self.assertIn("User examples:", prompt)
        self.assertIn("Reusable skill:", prompt)


if __name__ == "__main__":
    unittest.main()
