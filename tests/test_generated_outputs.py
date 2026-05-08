from __future__ import annotations

import unittest

from auto_skill.generated_outputs import (
    apply_outputs_to_pack,
    index_successful_outputs,
    private_leak_matches,
)


class GeneratedOutputsTests(unittest.TestCase):
    def test_index_successful_outputs_ignores_errors(self) -> None:
        rows = [
            {
                "status": "success",
                "job_id": "job-1",
                "desired_output": "ok",
                "finish_reason": "stop",
            },
            {"status": "error", "job_id": "job-2", "error": "failed"},
            {
                "status": "success",
                "job_id": "job-3",
                "desired_output": "truncated",
                "finish_reason": "length",
            },
            {"status": "success", "job_id": "job-4", "desired_output": "missing finish"},
        ]

        self.assertEqual(list(index_successful_outputs(rows)), ["job-1", "job-3", "job-4"])

    def test_apply_outputs_to_pack_replaces_placeholder(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "desired_output": {
                        "status": "needs_generation",
                        "generation_job_id": "job-1",
                        "prompt_sha256": "sha-1",
                        "prompt_template_version": "template-v1",
                    },
                }
            ]
        }
        outputs = {
            "job-1": {
                "desired_output": "final answer",
                "prompt_sha256": "sha-1",
                "prompt_template_version": "template-v1",
                "model": "qwen-plus",
                "finish_reason": "stop",
                "usage": {"total_tokens": 10},
            }
        }

        updated, applied, missing, rejected = apply_outputs_to_pack(pack, outputs)

        self.assertEqual(applied, 1)
        self.assertEqual(missing, 0)
        self.assertEqual(rejected, 0)
        self.assertEqual(updated["train_examples"][0]["desired_output"]["status"], "generated")
        self.assertEqual(updated["train_examples"][0]["desired_output"]["text"], "final answer")

    def test_index_then_apply_replaces_valid_success_output(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "desired_output": {
                        "status": "needs_generation",
                        "generation_job_id": "job-1",
                        "prompt_sha256": "sha-1",
                    },
                }
            ]
        }
        rows = [
            {
                "status": "success",
                "job_id": "job-1",
                "desired_output": "final answer",
                "prompt_sha256": "sha-1",
                "finish_reason": "stop",
            }
        ]

        updated, applied, missing, rejected = apply_outputs_to_pack(
            pack,
            index_successful_outputs(rows),
        )

        self.assertEqual(applied, 1)
        self.assertEqual(missing, 0)
        self.assertEqual(rejected, 0)
        self.assertEqual(updated["train_examples"][0]["desired_output"]["text"], "final answer")

    def test_apply_outputs_rejects_prompt_mismatch(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "desired_output": {
                        "status": "needs_generation",
                        "generation_job_id": "job-1",
                        "prompt_sha256": "new-sha",
                    },
                }
            ]
        }
        outputs = {
            "job-1": {
                "desired_output": "final answer",
                "prompt_sha256": "old-sha",
                "finish_reason": "stop",
            }
        }

        updated, applied, missing, rejected = apply_outputs_to_pack(pack, outputs)

        self.assertEqual(applied, 0)
        self.assertEqual(missing, 0)
        self.assertEqual(rejected, 1)
        self.assertEqual(
            updated["example_pack_status"]["rejections"][0]["reason"],
            "prompt_sha256_mismatch",
        )

    def test_apply_outputs_rejects_private_leak(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "desired_output": {
                        "status": "needs_generation",
                        "generation_job_id": "job-1",
                    },
                }
            ]
        }
        outputs = {
            "job-1": {
                "desired_output": "This includes benchmark rubric criteria.",
            }
        }

        _, applied, missing, rejected = apply_outputs_to_pack(pack, outputs)

        self.assertEqual(applied, 0)
        self.assertEqual(missing, 0)
        self.assertEqual(rejected, 1)

    def test_apply_outputs_rejects_non_stop_finish_reason(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "desired_output": {
                        "status": "needs_generation",
                        "generation_job_id": "job-1",
                    },
                }
            ]
        }
        outputs = {
            "job-1": {
                "desired_output": "truncated output",
                "finish_reason": "length",
            }
        }

        _, applied, missing, rejected = apply_outputs_to_pack(pack, outputs)

        self.assertEqual(applied, 0)
        self.assertEqual(missing, 0)
        self.assertEqual(rejected, 1)

    def test_apply_outputs_rejects_missing_finish_reason(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "desired_output": {
                        "status": "needs_generation",
                        "generation_job_id": "job-1",
                    },
                }
            ]
        }
        outputs = {
            "job-1": {
                "desired_output": "output without finish metadata",
            }
        }

        updated, applied, missing, rejected = apply_outputs_to_pack(pack, outputs)

        self.assertEqual(applied, 0)
        self.assertEqual(missing, 0)
        self.assertEqual(rejected, 1)
        self.assertEqual(
            updated["example_pack_status"]["rejections"][0]["reason"],
            "missing_or_non_stop_finish_reason",
        )

    def test_index_then_apply_counts_missing_finish_reason_as_rejected(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "desired_output": {
                        "status": "needs_generation",
                        "generation_job_id": "job-1",
                    },
                }
            ]
        }
        rows = [{"status": "success", "job_id": "job-1", "desired_output": "output"}]

        updated, applied, missing, rejected = apply_outputs_to_pack(
            pack,
            index_successful_outputs(rows),
        )

        self.assertEqual(applied, 0)
        self.assertEqual(missing, 0)
        self.assertEqual(rejected, 1)
        self.assertEqual(
            updated["example_pack_status"]["rejections"][0]["reason"],
            "missing_or_non_stop_finish_reason",
        )

    def test_private_leak_matches_detects_construction_terms(self) -> None:
        self.assertTrue(private_leak_matches("This mentions a benchmark rubric."))
        self.assertTrue(private_leak_matches("This mentions a hidden checklist."))
        self.assertTrue(private_leak_matches("This follows grading criteria."))
        self.assertFalse(private_leak_matches("This compares against an industry benchmark."))
        self.assertFalse(private_leak_matches("请按以下检查项完成交付。"))


if __name__ == "__main__":
    unittest.main()
