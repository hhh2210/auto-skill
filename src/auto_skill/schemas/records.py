"""Shared schema constants and validation helpers."""

from __future__ import annotations

from typing import Any

GENERATED_OUTPUT_SCHEMA_VERSION = "generated-desired-output/v1"
SKILL_INDUCTION_SCHEMA_VERSION = "skill-induction/v1"
EVAL_SCHEMA_VERSIONS = {
    "heldout-eval/v1",
    "pattern-similarity-eval/v1",
    "train-example-quality-audit/v1",
    "writingbench-official-eval/v1",
    "presentbench-official-score/v1",
    "author-style-heldout-eval/v1",
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
