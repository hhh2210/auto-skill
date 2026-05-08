"""Helpers for applying generated desired outputs to clean example packs."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

PRIVATE_LEAK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bbenchmark training example\b",
        r"\bprivate supervision\b",
        r"\bconstruction prompt\b",
        r"\bbenchmark rubric\b",
        r"\bjudge metadata\b",
        r"\brubric criteria\b",
        r"\bhidden rubric\b",
        r"\bhidden checklist\b",
        r"\bgrading criteria\b",
        r"\b(private|benchmark|grading)\s+(rubric|checklist|criteria)\b",
        r"\b(rubric|checklist|criteria)\s+(private|benchmark|grading)\b",
        r"\bchecklist groups\b",
        r"\bteacher critique\b",
        r"\bheldout feedback\b",
        r"\bevaluation metadata\b",
        r"benchmark\s*评分标准",
        r"隐藏.*评分标准",
        r"内部.*评分标准",
        r"评审提示",
        r"隐藏.*(rubric|评分|标准|checklist|检查)",
        r"BAILIAN_API_KEY",
        r"OPENAI_API_KEY",
    )
]


def private_leak_matches(text: str) -> list[str]:
    return [pattern.pattern for pattern in PRIVATE_LEAK_PATTERNS if pattern.search(text)]


def index_successful_outputs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        job_id = row.get("job_id")
        if not job_id:
            continue
        outputs[str(job_id)] = row
    return outputs


def apply_outputs_to_pack(
    pack: dict[str, Any],
    outputs_by_job_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], int, int, int]:
    updated = deepcopy(pack)
    applied = 0
    missing = 0
    rejected = 0
    rejection_details = []

    for example in updated.get("train_examples", []):
        desired_output = example.get("desired_output")
        if not isinstance(desired_output, dict):
            continue
        job_id = desired_output.get("generation_job_id")
        if not job_id:
            continue
        generated = outputs_by_job_id.get(str(job_id))
        if generated is None:
            missing += 1
            continue
        finish_reason = generated.get("finish_reason")
        if finish_reason != "stop":
            rejected += 1
            rejection_details.append(
                {
                    "job_id": job_id,
                    "reason": "missing_or_non_stop_finish_reason",
                    "finish_reason": finish_reason,
                }
            )
            continue
        text = generated.get("desired_output")
        if not isinstance(text, str) or not text.strip():
            rejected += 1
            rejection_details.append(
                {
                    "job_id": job_id,
                    "reason": "empty_or_non_string_desired_output",
                }
            )
            continue
        expected_prompt_sha = desired_output.get("prompt_sha256")
        actual_prompt_sha = generated.get("prompt_sha256")
        if expected_prompt_sha and actual_prompt_sha != expected_prompt_sha:
            rejected += 1
            rejection_details.append(
                {
                    "job_id": job_id,
                    "reason": "prompt_sha256_mismatch",
                    "expected": expected_prompt_sha,
                    "actual": actual_prompt_sha,
                }
            )
            continue
        expected_template = desired_output.get("prompt_template_version")
        actual_template = generated.get("prompt_template_version")
        if expected_template and actual_template != expected_template:
            rejected += 1
            rejection_details.append(
                {
                    "job_id": job_id,
                    "reason": "prompt_template_version_mismatch",
                    "expected": expected_template,
                    "actual": actual_template,
                }
            )
            continue
        leak_matches = private_leak_matches(text)
        if leak_matches:
            rejected += 1
            rejection_details.append(
                {
                    "job_id": job_id,
                    "reason": "private_leak_matches",
                    "matches": leak_matches,
                }
            )
            continue
        example["desired_output"] = {
            "schema_version": "generated-desired-output/v1",
            "status": "generated",
            "text": text,
            "generation_job_id": job_id,
            "prompt_sha256": generated.get("prompt_sha256"),
            "prompt_template_version": generated.get("prompt_template_version"),
            "model": generated.get("model"),
            "usage": generated.get("usage"),
        }
        applied += 1

    updated["example_pack_status"] = {
        "desired_outputs_applied": applied,
        "desired_outputs_missing": missing,
        "desired_outputs_rejected": rejected,
        "rejections": rejection_details,
    }
    return updated, applied, missing, rejected
