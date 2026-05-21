"""Leave-one-out validation prompt builders."""

from __future__ import annotations

import json
from typing import Any

from auto_skill.baselines._shared import format_user_example
from auto_skill.schemas import UserExample


def build_leave_one_out_validation_prompt(
    *,
    candidate_skill_md: str,
    validation_example: UserExample,
) -> str:
    return f"""Validate this candidate skill against a held-out user example.

The validation example was not used to create the candidate skill.
Use only the visible example. Do not use hidden rubrics or benchmark metadata.

Return JSON with:
- supported_rules: rules visibly supported by the validation example
- contradicted_rules: rules that conflict with the validation example
- optional_or_contextual_rules: rules that might be useful only in some cases
- outlier_notes: whether this example should narrow the skill scope

Candidate SKILL.md:
{candidate_skill_md}

Validation example:
{format_user_example(validation_example)}
"""


def build_validation_aware_skill_merge_prompt(
    *,
    candidate_skills: list[dict[str, Any]],
    leave_one_out_reports: list[dict[str, Any]],
) -> str:
    return f"""Merge leave-one-out candidate skills into the final SKILL.md.

Each candidate skill was generated from n-1 user-visible examples. Its validation
report comes from the one example held out from that induction round.

Merge policy:
- Keep rules broadly supported by validation reports as required rules.
- Move partially supported or context-specific rules to optional rules.
- Remove rules contradicted by multiple validation examples.
- Narrow the skill scope when validation reports identify outliers.
- Do not add hidden benchmark rubrics, judge prompts, or heldout task feedback.

Return the final validation-aware SKILL.md only.

Leave-one-out candidate skills:
{json.dumps(candidate_skills, ensure_ascii=False, indent=2)}

Leave-one-out validation reports:
{json.dumps(leave_one_out_reports, ensure_ascii=False, indent=2)}
"""
