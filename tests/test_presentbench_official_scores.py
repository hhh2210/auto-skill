from __future__ import annotations

import unittest

from scripts.eval.summarize_presentbench_official_scores import (
    model_identity_fields,
    parse_score_roots,
)


class PresentBenchOfficialScoresTests(unittest.TestCase):
    def test_parse_score_roots_requires_mode_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "MODE=PATH"):
            parse_score_roots(["/tmp/results"])

    def test_parse_score_roots_maps_modes_to_paths(self) -> None:
        roots = parse_score_roots(["prompt_only=/tmp/prompt", "auto_skill=/tmp/skill"])

        self.assertEqual(str(roots["prompt_only"]), "/tmp/prompt")
        self.assertEqual(str(roots["auto_skill"]), "/tmp/skill")

    def test_model_identity_fields_omits_unknown_models(self) -> None:
        self.assertEqual(
            model_identity_fields(solver_model=None, judge_model="gemini-3-flash-preview"),
            {"judge_model": "gemini-3-flash-preview"},
        )

    def test_model_identity_fields_records_solver_and_judge_models(self) -> None:
        self.assertEqual(
            model_identity_fields(
                solver_model="qwen3.5-plus",
                judge_model="gemini-3-flash-preview",
            ),
            {
                "solver_model": "qwen3.5-plus",
                "judge_model": "gemini-3-flash-preview",
            },
        )


if __name__ == "__main__":
    unittest.main()
