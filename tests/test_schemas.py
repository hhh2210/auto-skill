from __future__ import annotations

import unittest

from auto_skill.schemas import (
    SchemaValidationError,
    validate_eval_row,
    validate_generated_output_row,
    validate_skill_row,
)


class ArtifactSchemaTests(unittest.TestCase):
    def test_generated_output_success_requires_stop_finish_reason(self) -> None:
        row = {
            "schema_version": "generated-desired-output/v1",
            "status": "success",
            "job_id": "job-1",
            "pack_id": "pack-1",
            "example_id": "example-1",
            "desired_output": "output",
            "finish_reason": "length",
        }

        with self.assertRaisesRegex(SchemaValidationError, "finish_reason"):
            validate_generated_output_row(row)

    def test_generated_output_error_accepts_structured_error(self) -> None:
        validate_generated_output_row(
            {
                "schema_version": "generated-desired-output/v1",
                "status": "error",
                "job_id": "job-1",
                "pack_id": "pack-1",
                "example_id": "example-1",
                "error": {"type": "APITimeoutError"},
            }
        )

    def test_generated_output_rejection_statuses_match_generator(self) -> None:
        validate_generated_output_row(
            {
                "schema_version": "generated-desired-output/v1",
                "status": "rejected_private_leak",
                "job_id": "job-1",
                "pack_id": "pack-1",
                "example_id": "example-1",
                "finish_reason": "stop",
                "leak_matches": ["hidden"],
            }
        )
        validate_generated_output_row(
            {
                "schema_version": "generated-desired-output/v1",
                "status": "rejected_incomplete_generation",
                "job_id": "job-1",
                "pack_id": "pack-1",
                "example_id": "example-1",
                "finish_reason": "length",
            }
        )

    def test_skill_success_requires_schema_status_and_skill_text(self) -> None:
        validate_skill_row(
            {
                "schema_version": "skill-induction/v1",
                "status": "success",
                "pack_id": "pack-1",
                "mode": "auto_skill_ours_full",
                "skill_md": "# Skill",
                "model_calls": [],
            }
        )

    def test_eval_failure_row_has_null_score(self) -> None:
        validate_eval_row(
            {
                "schema_version": "heldout-eval/v1",
                "pack_id": "pack-1",
                "task_id": "task-1",
                "mode": "auto_skill",
                "evaluator_kind": "qwen_llm_rubric_surrogate",
                "status": "missing_ours_full_skill",
                "overall_score": None,
            }
        )

    def test_eval_non_success_rejects_score(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "non-success"):
            validate_eval_row(
                {
                    "schema_version": "heldout-eval/v1",
                    "pack_id": "pack-1",
                    "task_id": "task-1",
                    "mode": "auto_skill",
                    "evaluator_kind": "qwen_llm_rubric_surrogate",
                    "status": "missing_ours_full_skill",
                    "overall_score": 3,
                }
            )

    def test_eval_success_rejects_boolean_score(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "numeric"):
            validate_eval_row(
                {
                    "schema_version": "heldout-eval/v1",
                    "pack_id": "pack-1",
                    "task_id": "task-1",
                    "mode": "auto_skill",
                    "evaluator_kind": "qwen_llm_rubric_surrogate",
                    "status": "success",
                    "overall_score": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
