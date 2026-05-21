from __future__ import annotations

import unittest

from auto_skill.baselines import (
    FEATURE_SIGNATURE_MODE,
    FEATURE_SKILL_MODE,
    FULL_SKILL_MODE,
    ONE_SHOT_SKILL_MODE,
    OPERATIONAL_ANCHOR_MODE,
    REGISTRY,
    baseline_for_mode,
    mode_requires_skill,
    requested_induction_outputs,
    skill_for_mode,
)
from auto_skill.schemas import UserExample

EXAMPLES = [
    UserExample(
        example_id="ex-1",
        task_input="Write a concise update.",
        output="Done: shipped the parser fix.",
    )
]

TASK = {
    "task_id": "task-1",
    "task_input": "Write a concise update for the benchmark run.",
    "materials": [],
}


class BaselinesRegistryTests(unittest.TestCase):
    def test_every_registered_mode_builds_a_prompt(self) -> None:
        for mode, baseline in REGISTRY.items():
            with self.subTest(mode=mode):
                prompt = baseline.build_heldout_prompt(
                    task=TASK,
                    examples=EXAMPLES,
                    skill_md="# Reusable Skill\n- Keep updates concrete.",
                    max_material_chars=500,
                )

                self.assertIsInstance(prompt, str)
                self.assertIn(f"Mode: {mode}", prompt)
                self.assertIn(TASK["task_input"], prompt)

    def test_skill_lookup_matches_legacy_eval_dispatch(self) -> None:
        skills = {
            ("pack-1", ONE_SHOT_SKILL_MODE): "one-shot",
            ("pack-1", FEATURE_SKILL_MODE): "feature",
            ("pack-1", FEATURE_SIGNATURE_MODE): "signatures",
            ("pack-1", OPERATIONAL_ANCHOR_MODE): "anchors",
            ("pack-1", FULL_SKILL_MODE): "full",
        }

        cases = {
            "one_shot_skill_from_examples": "one-shot",
            "examples_plus_one_shot_skill": "one-shot",
            "ours_no_validation": "feature",
            "examples_plus_feature_skill": "feature",
            "examples_plus_operational_skill": None,
            "slide_constrained_examples_plus_feature_skill": "feature",
            "layout_plan_examples_plus_feature_skill": "feature",
            "feature_signatures_only": "signatures",
            "examples_plus_feature_signatures": "signatures",
            "task_first_feature_signatures": "signatures",
            "task_first_operational_anchors": "anchors",
            "task_first_evidence_anchored_operational_anchors": "anchors",
            "task_first_two_level_operational_anchors": "anchors",
            "task_first_planned_operational_anchors": "anchors",
            "auto_skill": "full",
            "prompt_only": None,
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode):
                self.assertEqual(skill_for_mode(mode, skills, "pack-1"), expected)

    def test_skill_requirement_and_special_plan_metadata(self) -> None:
        self.assertFalse(mode_requires_skill("prompt_only"))
        self.assertFalse(mode_requires_skill("few_shot_examples_only"))
        self.assertFalse(mode_requires_skill("examples_plus_operational_skill"))
        self.assertTrue(mode_requires_skill("auto_skill"))
        self.assertTrue(mode_requires_skill("task_first_planned_operational_anchors"))
        self.assertTrue(
            baseline_for_mode("layout_plan_examples_plus_feature_skill").spec.needs_layout_plan
        )
        self.assertTrue(
            baseline_for_mode("task_first_planned_operational_anchors").spec.needs_evidence_plan
        )
        self.assertEqual(
            baseline_for_mode("task_first_planned_operational_anchors").spec.benchmark_scope,
            ("writingbench",),
        )

    def test_feature_driven_and_ours_full_induction_metadata(self) -> None:
        self.assertEqual(
            requested_induction_outputs(
                {"one_shot_skill_from_examples", "auto_skill_feature_driven"},
                include_full=False,
            ),
            {"one_shot_skill_from_examples", "auto_skill_feature_driven_no_validation"},
        )
        self.assertEqual(
            requested_induction_outputs({"auto_skill_feature_driven"}, include_full=True),
            {"auto_skill_feature_driven_no_validation", "auto_skill_ours_full"},
        )
        self.assertEqual(
            REGISTRY["auto_skill_feature_driven_no_validation"].spec.induction_outputs,
            ("auto_skill_feature_driven_no_validation",),
        )
        self.assertEqual(
            REGISTRY["auto_skill_ours_full"].spec.induction_outputs,
            ("auto_skill_feature_driven_no_validation", "auto_skill_ours_full"),
        )

    def test_unknown_mode_falls_back_to_legacy_prompt_behavior(self) -> None:
        baseline = baseline_for_mode("legacy_debug_mode")

        prompt = baseline.build_heldout_prompt(task=TASK, examples=EXAMPLES)

        self.assertFalse(baseline.spec.requires_skill)
        self.assertIn("Mode: legacy_debug_mode", prompt)


if __name__ == "__main__":
    unittest.main()
