from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from scripts.metrics.demo_skill_quality_clarity import (
    DIMENSIONS,
    build_skill_quality_prompt,
    compute_skill_quality_avg,
    load_examples,
)


def _result(score: object) -> dict[str, object]:
    return {
        "scores": {
            dimension: {"score": score, "reason": "ok"} for dimension in DIMENSIONS
        },
        "summary": "ok",
    }


class SkillQualityClarityDemoTests(unittest.TestCase):
    def test_prompt_uses_ctx2skill_five_dimension_schema(self) -> None:
        prompt = build_skill_quality_prompt(
            examples_text="Example text",
            skill_md="---\nname: demo\ndescription: demo\n---\nBody",
        )
        for dimension in DIMENSIONS:
            self.assertIn(f'"{dimension}"', prompt)
        self.assertIn('"summary"', prompt)
        self.assertNotIn("overall_skill_quality", prompt)
        self.assertNotIn("fatal_issues", prompt)
        self.assertNotIn("triggerability", prompt)

    def test_compute_skill_quality_avg_scales_to_100(self) -> None:
        self.assertEqual(compute_skill_quality_avg(_result(5)), 100.0)
        self.assertEqual(compute_skill_quality_avg(_result(1)), 20.0)

    def test_compute_skill_quality_avg_uses_arithmetic_mean(self) -> None:
        result = {
            "scores": {
                "faithfulness": {"score": 4},
                "reusability": {"score": 3},
                "effectiveness": {"score": 5},
                "clarity": {"score": 4},
                "conciseness": {"score": 4},
            }
        }
        self.assertEqual(compute_skill_quality_avg(result), 80.0)

    def test_compute_skill_quality_avg_rejects_missing_dimension(self) -> None:
        result = _result(3)
        del result["scores"]["faithfulness"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "missing score object"):
            compute_skill_quality_avg(result)

    def test_compute_skill_quality_avg_rejects_out_of_range_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "number from 1 to 5"):
            compute_skill_quality_avg(_result(6))

    def test_compute_skill_quality_avg_rejects_bool_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "number from 1 to 5"):
            compute_skill_quality_avg(_result(True))

    def test_compute_skill_quality_avg_rejects_nan_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "number from 1 to 5"):
            compute_skill_quality_avg(_result(math.nan))

    def test_load_examples_projects_public_fields_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pack.jsonl"
            path.write_text(
                (
                    '{"pack_id":"p","train_examples":[{'
                    '"example_id":"ex1",'
                    '"task_input":"Write a memo.",'
                    '"materials":[{"path":"public.pdf"}],'
                    '"desired_output":{"text":"Memo text","usage":{"secret":1}},'
                    '"private_rubrics":["do not leak"],'
                    '"heldout_tasks":[{"task_input":"private"}]'
                    "}]}\n"
                ),
                encoding="utf-8",
            )
            text = load_examples(path)
        self.assertIn("Write a memo.", text)
        self.assertIn("Memo text", text)
        self.assertIn("public.pdf", text)
        self.assertNotIn("private_rubrics", text)
        self.assertNotIn("heldout_tasks", text)
        self.assertNotIn("secret", text)

    def test_load_examples_rejects_non_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text('["not", "an", "object"]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object rows"):
                load_examples(path)

    def test_load_examples_rejects_non_object_train_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_pack.jsonl"
            path.write_text('{"train_examples":["bad"]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_examples"):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()
