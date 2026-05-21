"""One-shot skill baseline prompt builder."""

from __future__ import annotations

from auto_skill.baselines._shared import format_user_example
from auto_skill.schemas import UserExample


def build_one_shot_skill_prompt(examples: list[UserExample]) -> str:
    joined = "\n\n".join(format_user_example(example) for example in examples)
    return f"""You are creating a reusable SKILL.md from user-provided examples.

Use only the visible task inputs, outputs, and materials in the examples.
Do not infer hidden rubrics, benchmark metadata, teacher traces, or heldout feedback.

Write one concise, operational SKILL.md. Include:
- When to use this skill
- Inputs expected from the user
- Procedure
- Required rules
- Optional rules
- Do not generalize
- Self-check before final answer

Examples:
{joined}
"""
