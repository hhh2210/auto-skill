"""Schema validation for skill induction rows."""

from __future__ import annotations

from typing import Any

from auto_skill.schemas.records import (
    SKILL_INDUCTION_SCHEMA_VERSION,
    SchemaValidationError,
    _check_optional_str_or_null,
    _require_key,
    _require_nullable_key,
    _require_str,
)


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
