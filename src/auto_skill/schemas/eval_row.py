"""Schema validation for heldout evaluation rows."""

from __future__ import annotations

import math
from typing import Any

from auto_skill.schemas.records import (
    EVAL_SCHEMA_VERSIONS,
    SchemaValidationError,
    _check_optional_str_or_null,
    _require_nullable_key,
    _require_str,
)


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
