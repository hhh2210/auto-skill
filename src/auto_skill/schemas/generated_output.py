"""Schema validation for generated desired-output rows."""

from __future__ import annotations

from typing import Any

from auto_skill.schemas.records import (
    GENERATED_OUTPUT_SCHEMA_VERSION,
    SchemaValidationError,
    _require_key,
    _require_str,
)


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
