"""Schema validation for loaded artifact rows."""

from __future__ import annotations

from typing import Any

from auto_skill.schemas.eval_row import validate_eval_row
from auto_skill.schemas.generated_output import validate_generated_output_row
from auto_skill.schemas.skill_row import validate_skill_row


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
