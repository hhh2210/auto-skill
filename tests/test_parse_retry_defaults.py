import unittest

from scripts.eval import audit_train_examples, run_heldout_eval, run_writingbench_official_eval
from scripts.metrics import (
    run_grounding_eval,
    run_pattern_similarity_eval,
    run_self_consistency_metric,
    run_skill_quality_eval,
)
from scripts.skills import run_skill_mvp


class ParseRetryDefaultsTests(unittest.TestCase):
    def test_llm_json_and_judge_parse_retries_default_to_three(self) -> None:
        modules = [
            audit_train_examples,
            run_grounding_eval,
            run_heldout_eval,
            run_pattern_similarity_eval,
            run_self_consistency_metric,
            run_skill_quality_eval,
            run_skill_mvp,
            run_writingbench_official_eval,
        ]

        for module in modules:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.DEFAULT_PARSE_MAX_ATTEMPTS, 3)

    def test_writingbench_planner_parse_retry_default_stays_three(self) -> None:
        self.assertEqual(run_writingbench_official_eval.DEFAULT_PLAN_PARSE_MAX_ATTEMPTS, 3)

    def test_skill_stage_config_uses_default_parse_retry_budget(self) -> None:
        config = run_skill_mvp.StageCallConfig(
            model="qwen3.5-plus",
            temperature=0.0,
            max_tokens=1024,
            enable_thinking=False,
            thinking_budget=None,
            stream=False,
        )

        self.assertEqual(config.parse_max_attempts, run_skill_mvp.DEFAULT_PARSE_MAX_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
