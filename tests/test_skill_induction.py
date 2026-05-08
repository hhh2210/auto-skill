from __future__ import annotations

import unittest

from auto_skill.schemas import UserExample
from auto_skill.skill_induction import (
    RuleSupport,
    build_feature_extraction_prompt,
    classify_rule_support,
    merge_rule_supports,
)


class SkillInductionTests(unittest.TestCase):
    def test_classify_rule_support(self) -> None:
        self.assertEqual(classify_rule_support(4, 4), "required")
        self.assertEqual(classify_rule_support(2, 4), "optional")
        self.assertEqual(classify_rule_support(1, 4), "reject")

    def test_merge_rule_supports_accounts_for_contradictions(self) -> None:
        merged = merge_rule_supports(
            [
                RuleSupport("Always include a summary", ("a", "b", "c"), ()),
                RuleSupport("Use exactly five slides", ("a", "b"), ("c",)),
            ],
            total_examples=3,
        )

        self.assertEqual([rule.rule for rule in merged["required"]], ["Always include a summary"])
        self.assertEqual([rule.rule for rule in merged["reject"]], ["Use exactly five slides"])

    def test_feature_prompt_states_no_hidden_rubrics(self) -> None:
        prompt = build_feature_extraction_prompt(
            UserExample(
                example_id="ex1",
                task_input="Create a research presentation.",
                output="A 10-slide deck with problem, method, and results.",
                user_notes="Keep the structure.",
            )
        )

        self.assertIn("Do not infer hidden rubrics", prompt)
        self.assertIn("Create a research presentation.", prompt)
        self.assertIn("Keep the structure.", prompt)


if __name__ == "__main__":
    unittest.main()
