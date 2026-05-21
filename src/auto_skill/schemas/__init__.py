"""Shared schemas for auto-skill inputs and outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from auto_skill.schemas.artifacts import validate_artifact_rows
from auto_skill.schemas.eval_row import validate_eval_row
from auto_skill.schemas.generated_output import validate_generated_output_row
from auto_skill.schemas.records import (
    EVAL_SCHEMA_VERSIONS,
    GENERATED_OUTPUT_SCHEMA_VERSION,
    SKILL_INDUCTION_SCHEMA_VERSION,
    SchemaValidationError,
)
from auto_skill.schemas.skill_row import validate_skill_row
from auto_skill.schemas.user_example import SkillPackage, UserExample

__all__ = [
    "Any",
    "EVAL_SCHEMA_VERSIONS",
    "GENERATED_OUTPUT_SCHEMA_VERSION",
    "SKILL_INDUCTION_SCHEMA_VERSION",
    "SchemaValidationError",
    "SkillPackage",
    "UserExample",
    "dataclass",
    "field",
    "math",
    "validate_artifact_rows",
    "validate_eval_row",
    "validate_generated_output_row",
    "validate_skill_row",
]

for _module_name in (
    "artifacts",
    "eval_row",
    "generated_output",
    "records",
    "skill_row",
    "user_example",
):
    globals().pop(_module_name, None)
del _module_name
