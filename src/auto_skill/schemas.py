"""Shared schemas for auto-skill inputs and outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

GENERATED_OUTPUT_SCHEMA_VERSION = "generated-desired-output/v1"
SKILL_INDUCTION_SCHEMA_VERSION = "skill-induction/v1"
EVAL_SCHEMA_VERSIONS = {
    "heldout-eval/v1",
    "pattern-similarity-eval/v1",
    "train-example-quality-audit/v1",
    "writingbench-official-eval/v1",
    "presentbench-official-score/v1",
}


class SchemaValidationError(ValueError):
    """Raised when a JSONL artifact row does not match its declared schema."""


def _require_key(row: dict[str, Any], key: str, *, label: str) -> Any:
    if key not in row:
        raise SchemaValidationError(f"{label}: missing required field: {key}")
    return row[key]


def _require_str(row: dict[str, Any], key: str, *, label: str) -> str:
    value = _require_key(row, key, label=label)
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{label}: {key} must be a non-empty string")
    return value


def _require_nullable_key(row: dict[str, Any], key: str, *, label: str) -> None:
    if key not in row:
        raise SchemaValidationError(f"{label}: missing required nullable field: {key}")


def _check_optional_str_or_null(row: dict[str, Any], key: str, *, label: str) -> None:
    if key not in row:
        return
    value = row[key]
    if value is None:
        return
    if not isinstance(value, str):
        raise SchemaValidationError(f"{label}: {key} must be a string or null")


def validate_generated_output_row(row: dict[str, Any], *, label: str = "row") -> None:
    """Validate one desired-output generation JSONL row."""

    schema_version = _require_str(row, "schema_version", label=label)
    if schema_version != GENERATED_OUTPUT_SCHEMA_VERSION:
        raise SchemaValidationError(f"{label}: unsupported schema_version: {schema_version}")
    status = _require_str(row, "status", label=label)
    _require_str(row, "job_id", label=label)
    _require_str(row, "pack_id", label=label)
    _require_str(row, "example_id", label=label)
    if status == "success":
        _require_str(row, "desired_output", label=label)
        _require_str(row, "finish_reason", label=label)
        if row["finish_reason"] != "stop":
            raise SchemaValidationError(f"{label}: success row finish_reason must be stop")
    elif status == "error":
        _require_key(row, "error", label=label)
    elif status == "rejected_private_leak":
        _require_key(row, "leak_matches", label=label)
        _require_str(row, "finish_reason", label=label)
    elif status == "rejected_incomplete_generation":
        _require_str(row, "finish_reason", label=label)
    else:
        raise SchemaValidationError(f"{label}: unsupported generated output status: {status}")


def validate_skill_row(row: dict[str, Any], *, label: str = "row") -> None:
    """Validate one skill-induction JSONL row."""

    schema_version = _require_str(row, "schema_version", label=label)
    if schema_version != SKILL_INDUCTION_SCHEMA_VERSION:
        raise SchemaValidationError(f"{label}: unsupported schema_version: {schema_version}")
    status = _require_str(row, "status", label=label)
    _require_str(row, "pack_id", label=label)
    _require_str(row, "mode", label=label)
    model_calls = _require_key(row, "model_calls", label=label)
    if not isinstance(model_calls, list):
        raise SchemaValidationError(f"{label}: model_calls must be a list")
    _check_optional_str_or_null(row, "solver_model", label=label)
    if status == "success":
        _require_str(row, "skill_md", label=label)
    elif status == "induction_error":
        _require_nullable_key(row, "skill_md", label=label)
        _require_str(row, "error", label=label)
    else:
        raise SchemaValidationError(f"{label}: unsupported skill row status: {status}")


def validate_eval_row(row: dict[str, Any], *, label: str = "row") -> None:
    """Validate one heldout evaluation or official score JSONL row."""

    schema_version = _require_str(row, "schema_version", label=label)
    if schema_version not in EVAL_SCHEMA_VERSIONS:
        raise SchemaValidationError(f"{label}: unsupported schema_version: {schema_version}")
    status = _require_str(row, "status", label=label)
    _require_str(row, "pack_id", label=label)
    _require_str(row, "task_id", label=label)
    _require_str(row, "mode", label=label)
    _require_str(row, "evaluator_kind", label=label)
    _require_nullable_key(row, "overall_score", label=label)
    _check_optional_str_or_null(row, "solver_model", label=label)
    _check_optional_str_or_null(row, "judge_model", label=label)
    if status == "success":
        score = row["overall_score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise SchemaValidationError(f"{label}: success overall_score must be numeric")
    elif row["overall_score"] is not None:
        raise SchemaValidationError(f"{label}: non-success overall_score must be null")


def validate_artifact_rows(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    label: str = "artifact",
) -> None:
    validators = {
        "generated_outputs": validate_generated_output_row,
        "skills": validate_skill_row,
        "eval": validate_eval_row,
    }
    validator = validators.get(kind)
    if validator is None:
        raise ValueError(f"unknown artifact kind: {kind}")
    for index, row in enumerate(rows, start=1):
        validator(row, label=f"{label}:{index}")


@dataclass(frozen=True)
class UserExample:
    """A user-visible example that can be used for skill induction."""

    example_id: str
    task_input: str
    output: str | None = None
    user_notes: str | None = None
    materials: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must be non-empty")
        if not self.task_input.strip():
            raise ValueError("task_input must be non-empty")


@dataclass(frozen=True)
class SkillPackage:
    """A generated skill package before it is written to disk."""

    name: str
    skill_md: str
    metadata: dict[str, Any] = field(default_factory=dict)
    templates: dict[str, str] = field(default_factory=dict)
    tests: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.skill_md.strip():
            raise ValueError("skill_md must be non-empty")
