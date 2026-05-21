"""Schema validation for heldout evaluation rows."""

from __future__ import annotations

import math
from typing import Any

from auto_skill.schemas.records import (
    EVAL_SCHEMA_VERSIONS,
    SchemaValidationError,
    _check_optional_str_or_null,
    _require_key,
    _require_nullable_key,
    _require_str,
)


def validate_eval_row(row: dict[str, Any], *, label: str = "row") -> None:
    """Validate one heldout evaluation or official score JSONL row."""

    schema_version = _require_str(row, "schema_version", label=label)
    if schema_version not in EVAL_SCHEMA_VERSIONS:
        raise SchemaValidationError(f"{label}: unsupported schema_version: {schema_version}")
    if schema_version == "author-style-heldout-eval/v1":
        validate_author_style_eval_row(row, label=label)
        return
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


def validate_author_style_eval_row(row: dict[str, Any], *, label: str = "row") -> None:
    """Validate one personal author-style eval row."""

    status = _require_str(row, "status", label=label)
    _require_str(row, "pack_id", label=label)
    _require_str(row, "task_id", label=label)
    _require_str(row, "mode", label=label)
    _check_optional_str_or_null(row, "solver_model", label=label)
    _check_optional_str_or_null(row, "judge_model", label=label)
    _check_optional_str_or_null(row, "judge_config_model", label=label)
    if status != "success":
        if "style_likeness_1_to_10" in row and row["style_likeness_1_to_10"] is not None:
            raise SchemaValidationError(
                f"{label}: non-success style_likeness_1_to_10 must be null"
            )
        return

    score = row.get("style_likeness_1_to_10")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 1 <= float(score) <= 10
    ):
        raise SchemaValidationError(
            f"{label}: success style_likeness_1_to_10 must be numeric in [1, 10]"
        )
    wins = row.get("candidate_beats_negatives")
    hard_negative_count = row.get("hard_negative_count")
    if isinstance(wins, bool) or not isinstance(wins, int):
        raise SchemaValidationError(f"{label}: candidate_beats_negatives must be an int")
    if (
        isinstance(hard_negative_count, bool)
        or not isinstance(hard_negative_count, int)
        or hard_negative_count <= 0
    ):
        raise SchemaValidationError(f"{label}: hard_negative_count must be a positive int")
    if not 0 <= wins <= hard_negative_count:
        raise SchemaValidationError(
            f"{label}: candidate_beats_negatives must be within hard_negative_count"
        )
    generation = _require_key(row, "generation", label=label)
    if not isinstance(generation, dict):
        raise SchemaValidationError(f"{label}: generation must be an object")
    if not isinstance(generation.get("text"), str) or not generation["text"].strip():
        raise SchemaValidationError(f"{label}: generation.text must be non-empty")
    judge_report = _require_key(row, "judge_report", label=label)
    if not isinstance(judge_report, dict):
        raise SchemaValidationError(f"{label}: judge_report must be an object")
