from __future__ import annotations

import unittest

from auto_skill.mvp import (
    build_current_evidence_inventory,
    build_evidence_scaffolding_plan_prompt,
    build_feature_signature_context,
    build_heldout_generation_prompt,
    build_judge_prompt,
    build_operational_anchor_context,
    build_planned_operational_anchor_prompt,
    build_task_first_feature_signature_prompt,
    build_validation_aware_skill_merge_prompt,
    evaluation_criteria,
    extract_overall_score,
    parse_json_object,
    user_examples_from_pack,
    validate_evidence_plan,
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

    def test_task_first_feature_signature_prompt_places_task_before_signatures(self) -> None:
        prompt = build_task_first_feature_signature_prompt(
            task={
                "task_id": "task-1",
                "task_input": "Write the current memo about Alpha.",
                "materials": [],
            },
            feature_signatures="Do not copy Beta.",
        )

        self.assertIn("Mode: task_first_feature_signatures", prompt)
        self.assertIn("higher priority than all", prompt)
        self.assertLess(
            prompt.index("Heldout task input:"),
            prompt.index("Reusable feature signatures:"),
        )
        self.assertIn("Do not import task-specific facts", prompt)

    def test_task_first_feature_signature_mode_uses_custom_prompt(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={"task_id": "task-1", "task_input": "Write a memo.", "materials": []},
            mode="task_first_feature_signatures",
            examples=[],
            skill_md="# Abstract Feature Signatures",
        )

        self.assertIn("Mode: task_first_feature_signatures", prompt)
        self.assertNotIn("User examples:", prompt)
        self.assertLess(
            prompt.index("Heldout task input:"),
            prompt.index("Reusable feature signatures:"),
        )

    def test_operational_anchor_context_preserves_detail_slots_not_example_facts(self) -> None:
        context = build_operational_anchor_context(
            {
                "feature_reports": [
                    {
                        "content_features": {
                            "key_variables": ["Indoor temperature", "CO2 concentration"],
                            "evaluation_metrics": ["Energy reduction percentages"],
                        },
                        "structure_features": {
                            "outline_sections": ["Abstract", "Methodology", "Results"]
                        },
                        "style_features": {"tone": "Formal, academic, and technical"},
                        "must_not_generalize": [
                            "Do not copy Building and Environment journal details.",
                        ],
                    }
                ]
            }
        )

        assert context is not None
        self.assertIn("Task-Grounded Operational Anchors", context)
        self.assertIn("key variables", context)
        self.assertIn("evaluation metrics", context)
        self.assertIn("current-task entities", context)
        self.assertIn("Only fill a detail slot", context)
        self.assertIn("current evidence lacks a concrete value", context)
        self.assertIn("Anti-leakage checks", context)
        self.assertNotIn("Indoor temperature", context)
        self.assertNotIn("CO2 concentration", context)
        self.assertNotIn("Building and Environment", context)

    def test_task_first_operational_anchor_mode_uses_task_first_prompt(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={"task_id": "task-1", "task_input": "Write a memo.", "materials": []},
            mode="task_first_operational_anchors",
            examples=[],
            skill_md="# Task-Grounded Operational Anchors",
        )

        self.assertIn("Mode: task_first_operational_anchors", prompt)
        self.assertIn("Reusable operational anchors:", prompt)
        self.assertIn("Evidence policy for operational anchors", prompt)
        self.assertIn("Treat anchors as requests for detail types", prompt)
        self.assertLess(
            prompt.index("Heldout task input:"),
            prompt.index("Reusable operational anchors:"),
        )

    def test_current_evidence_inventory_extracts_current_task_facts(self) -> None:
        inventory = build_current_evidence_inventory(
            {
                "task_input": "Write about SUPOR SW-CFB301 using 950W and 30 cm specs.",
                "materials": [
                    {
                        "path": "spec.md",
                        "text": "The product supports Espresso and American Drip.",
                    }
                ],
            }
        )

        self.assertIn("950W", inventory)
        self.assertIn("30", inventory)
        self.assertIn("SUPOR", inventory)
        self.assertIn("Espresso", inventory)

    def test_evidence_anchored_operational_anchor_mode_includes_inventory(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={
                "task_id": "task-1",
                "task_input": "Write about SUPOR SW-CFB301 using 950W.",
                "materials": [],
            },
            mode="task_first_evidence_anchored_operational_anchors",
            examples=[],
            skill_md="# Task-Grounded Operational Anchors",
        )

        self.assertIn("Mode: task_first_evidence_anchored_operational_anchors", prompt)
        self.assertIn("Current evidence inventory for concrete facts", prompt)
        self.assertIn("Use this inventory as a whitelist", prompt)
        self.assertIn("Reusable operational anchors:", prompt)
        self.assertIn("950W", prompt)
        self.assertLess(
            prompt.index("Heldout task input:"),
            prompt.index("Current evidence inventory for concrete facts"),
        )

    def test_two_level_operational_anchor_mode_allows_generic_scaffolding(self) -> None:
        prompt = build_heldout_generation_prompt(
            task={
                "task_id": "task-1",
                "task_input": "Write about SUPOR SW-CFB301 using 950W.",
                "materials": [],
            },
            mode="task_first_two_level_operational_anchors",
            examples=[],
            skill_md="# Task-Grounded Operational Anchors",
        )

        self.assertIn("Mode: task_first_two_level_operational_anchors", prompt)
        self.assertIn("Current evidence inventory for specific factual claims", prompt)
        self.assertIn("Two-level evidence policy", prompt)
        self.assertIn("Generic scaffolding is allowed only when it stays generic", prompt)
        self.assertIn("950W", prompt)
        self.assertIn("Reusable operational anchors:", prompt)

    def test_evidence_scaffolding_plan_prompt_requires_json_not_final_answer(self) -> None:
        prompt = build_evidence_scaffolding_plan_prompt(
            task={"task_id": "task-1", "task_input": "Write about SUPOR 950W.", "materials": []},
            operational_anchors="# Task-Grounded Operational Anchors",
        )

        self.assertIn("Return strict JSON only", prompt)
        self.assertIn("grounded_facts", prompt)
        self.assertIn("generic_scaffolding", prompt)
        self.assertIn("missing_specifics", prompt)
        self.assertIn("SUPOR", prompt)
        self.assertIn("950W", prompt)

    def test_planned_operational_anchor_prompt_uses_plan_fields(self) -> None:
        prompt = build_planned_operational_anchor_prompt(
            task={"task_id": "task-1", "task_input": "Write about SUPOR 950W.", "materials": []},
            operational_anchors="# Task-Grounded Operational Anchors",
            evidence_plan={
                "grounded_facts": ["SUPOR uses 950W"],
                "generic_scaffolding": ["Compare use cases generically"],
                "missing_specifics": ["exact extraction time"],
                "generation_constraints": ["Do not invent missing specifics"],
            },
        )

        self.assertIn("Mode: task_first_planned_operational_anchors", prompt)
        self.assertIn("SUPOR uses 950W", prompt)
        self.assertIn("Compare use cases generically", prompt)
        self.assertIn("Do not invent missing_specifics", prompt)
        self.assertIn("Reusable operational anchors:", prompt)

    def test_validate_evidence_plan_requires_list_strings(self) -> None:
        self.assertEqual(
            validate_evidence_plan(
                {
                    "grounded_facts": ["SUPOR uses 950W"],
                    "generic_scaffolding": ["Use a generic buying-guide structure"],
                    "missing_specifics": ["exact extraction time"],
                    "generation_constraints": ["Do not invent missing specifics"],
                }
            ),
            [],
        )
        self.assertEqual(
            validate_evidence_plan(
                {
                    "grounded_facts": [],
                    "generic_scaffolding": "not a list",
                    "missing_specifics": [""],
                }
            ),
            [
                "generic_scaffolding_not_list",
                "missing_specifics_contains_non_string",
                "generation_constraints_not_list",
            ],
        )


if __name__ == "__main__":
    unittest.main()
