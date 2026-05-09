from __future__ import annotations

import unittest

from auto_skill.generated_outputs import (
    apply_outputs_to_pack,
    index_latest_outputs,
    index_successful_outputs,
    latest_generation_rows,
    latest_successful_generation_rows,
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

    def test_latest_generation_rows_uses_job_and_prompt_sha(self) -> None:
        rows = [
            {"status": "error", "job_id": "job-1", "prompt_sha256": "sha-1"},
            {"status": "success", "job_id": "job-1", "prompt_sha256": "sha-1"},
            {"status": "success", "job_id": "job-1", "prompt_sha256": "sha-2"},
            {"status": "success", "job_id": "missing-sha"},
        ]

        latest = latest_generation_rows(rows)

        self.assertEqual(set(latest), {("job-1", "sha-1"), ("job-1", "sha-2")})
        self.assertEqual(latest[("job-1", "sha-1")]["status"], "success")

    def test_index_latest_outputs_keeps_rejected_latest_rows(self) -> None:
        rows = [
            {"status": "success", "job_id": "job-1", "prompt_sha256": "sha-1"},
            {
                "status": "rejected_incomplete_generation",
                "job_id": "job-1",
                "prompt_sha256": "sha-1",
                "finish_reason": "length",
            },
            {"status": "error", "job_id": "job-2"},
        ]

        latest = index_latest_outputs(rows)

        self.assertEqual(latest["job-1"]["status"], "rejected_incomplete_generation")
        self.assertEqual(latest["job-2"]["status"], "error")

    def test_latest_successful_generation_rows_filters_historical_failures(self) -> None:
        rows = [
            {
                "status": "rejected_incomplete_generation",
                "job_id": "job-1",
                "prompt_sha256": "sha-1",
                "finish_reason": "length",
            },
            {
                "status": "success",
                "job_id": "job-1",
                "prompt_sha256": "sha-1",
                "finish_reason": "stop",
            },
            {
                "status": "rejected_incomplete_generation",
                "job_id": "job-2",
                "prompt_sha256": "sha-2",
                "finish_reason": "length",
            },
            {
                "status": "success",
                "job_id": "job-3",
                "prompt_sha256": "sha-3",
                "finish_reason": "length",
            },
        ]

        successful, status_counts = latest_successful_generation_rows(rows)

        self.assertEqual([row["job_id"] for row in successful], ["job-1"])
        self.assertEqual(status_counts, {"rejected_incomplete_generation": 1, "success": 2})

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
        self.assertFalse(private_leak_matches("The report defines evaluation criteria."))
        self.assertFalse(private_leak_matches("请按以下检查项完成交付。"))


if __name__ == "__main__":
    unittest.main()
