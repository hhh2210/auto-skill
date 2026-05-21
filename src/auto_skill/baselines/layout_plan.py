"""PresentBench layout-plan baseline prompt builder."""

from __future__ import annotations

from typing import Any

from auto_skill.baselines._shared import format_user_example
from auto_skill.cleaning.packs import material_context
from auto_skill.schemas import UserExample


def build_presentbench_layout_plan_prompt(
    *,
    task: dict[str, Any],
    examples: list[UserExample],
    skill_md: str | None = None,
    max_material_chars: int = 4000,
) -> str:
    task_id = task.get("task_id") or task.get("example_id") or task.get("source_task_id")
    materials = material_context(task.get("materials", []), max_chars=max_material_chars)
    examples_text = "\n\n".join(format_user_example(example) for example in examples)
    skill_text = skill_md or "(none)"
    return f"""Create a current-task slide layout plan before generation.

Use the heldout task input and material excerpts as the source of truth.
Use examples and skill only to notice reusable style/coverage patterns; do not
copy their slide counts, figure placement, section placement, or task-specific
wording.

Return a concise plan with these headings:
- Hard constraints: slide count, required figures, exact required wording,
  citation/source placement, bullet limits, section ordering.
- Per-slide allocation: one line per slide or slide range.
- Do-not-copy-from-examples: concrete example-specific patterns to ignore.
- Final generation checklist: constraints to verify before answering.

Task ID: {task_id}

User examples:
{examples_text}

Reusable skill:
{skill_text}

Heldout task input:
{task["task_input"]}

Material excerpts:
{materials}
"""
