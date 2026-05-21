"""Shared helpers for baseline prompt construction."""

from __future__ import annotations

from typing import Any

from auto_skill.baselines.feature_driven import build_task_first_feature_signature_prompt
from auto_skill.cleaning.packs import material_context
from auto_skill.schemas import UserExample

HELDOUT_GENERATION_PROMPT_VERSION = "deliverable-priority-v2"


def user_examples_from_pack(pack: dict[str, Any]) -> list[UserExample]:
    """Convert a clean example pack into the canonical user-example input."""

    examples = []
    for row in pack.get("train_examples", []):
        desired = row.get("desired_output")
        if not isinstance(desired, dict) or desired.get("status") != "generated":
            continue
        text = desired.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        materials = tuple(
            str(material.get("path"))
            for material in row.get("materials", [])
            if isinstance(material, dict) and material.get("path")
        )
        examples.append(
            UserExample(
                example_id=str(row["example_id"]),
                task_input=str(row["task_input"]),
                output=text,
                materials=materials,
                metadata={
                    "source": row.get("source"),
                    "source_task_id": row.get("source_task_id"),
                    "domain": row.get("domain", {}),
                },
            )
        )
    return examples


def format_user_example(example: UserExample) -> str:
    materials = "\n".join(f"- {item}" for item in example.materials) or "- None"
    output = example.output or "(missing output)"
    return f"""### Example {example.example_id}

Task input:
{example.task_input}

Desired output:
{output}

Materials:
{materials}
"""


def build_heldout_generation_prompt(
    *,
    task: dict[str, Any],
    mode: str,
    examples: list[UserExample],
    skill_md: str | None = None,
    max_material_chars: int = 4000,
) -> str:
    task_id = task.get("task_id") or task.get("example_id") or task.get("source_task_id")
    materials = material_context(task.get("materials", []), max_chars=max_material_chars)
    if mode in {
        "task_first_feature_signatures",
        "task_first_operational_anchors",
        "task_first_evidence_anchored_operational_anchors",
        "task_first_two_level_operational_anchors",
    }:
        return build_task_first_feature_signature_prompt(
            task=task,
            feature_signatures=skill_md,
            mode=mode,
            max_material_chars=max_material_chars,
        )
    examples_text = ""
    if mode in {
        "few_shot_examples_only",
        "few_shot_examples_bundle_preserve",
        "examples_plus_one_shot_skill",
        "examples_plus_feature_skill",
        "examples_plus_operational_skill",
        "examples_plus_feature_signatures",
        "slide_constrained_examples_plus_feature_skill",
        "layout_plan_examples_plus_feature_skill",
    }:
        examples_text = "\n\nUser examples:\n" + "\n\n".join(
            format_user_example(example) for example in examples
        )
    skill_text = ""
    if skill_md:
        skill_text = f"\n\nReusable skill:\n{skill_md}"
    mode_guidance = ""
    deliverable_guidance = """
Current-task deliverable priority:
- Produce the completed artifact requested by the heldout task input.
- Do not answer with a plan, outline, checklist, analysis, rubric mapping, or
  explanation of how to solve the task unless the heldout task explicitly asks
  for that artifact type.
- Treat the heldout task input and material excerpts as higher priority than
  user-example patterns or reusable-skill rules.
- Use examples and skills to infer style, structure, and constraints, but do
  not import example-specific facts, domains, entities, or section topics into
  the current answer unless they appear in the heldout task.
"""
    if mode == "slide_constrained_examples_plus_feature_skill":
        mode_guidance = """

Slide-task constraint priority:
- Treat the heldout task input and material excerpts as the source of truth for
  slide count, required figures, exact wording, citation placement, bullet
  limits, and section ordering.
- Use user examples only as abstract style and coverage references.
- Do not copy example slide counts, figure allocation, section placement,
  visual placeholders, or task-specific wording unless the current task asks for
  the same thing.
- Before writing the final answer, internally plan the current task's hard slide
  constraints and ensure every required figure or exact phrase has its own
  requested placement.
"""
    if mode == "few_shot_examples_bundle_preserve":
        mode_guidance = """

Author-style bundle preservation:
- Treat the user examples as examples of low-level writing operations, not as
  polished essays to summarize.
- Preserve the observed shape of the author's public examples when it is
  visible: multi-comment bundles, abrupt topic jumps, quote/reply fragments,
  short reactive turns, uneven paragraphing, hedges, slang, punctuation habits,
  spelling/nonstandard orthography, laughter markers, and casual register.
- Do not smooth the answer into a coherent essay, article, memo, or balanced
  analysis unless that polished form is clearly present in the examples.
- Keep the heldout task's topic and requested content current; do not copy
  example-specific topics, people, facts, or stories.
"""
    if mode == "layout_plan_examples_plus_feature_skill":
        mode_guidance = """

Slide-task constraint priority:
- Follow the provided current-task layout plan over any user-example pattern.
- Treat the heldout task input and material excerpts as the source of truth for
  slide count, required figures, exact wording, citation placement, bullet
  limits, and section ordering.
- Use user examples only as abstract style and coverage references.
- Do not copy example slide counts, figure allocation, section placement,
  visual placeholders, or task-specific wording unless the current task asks for
  the same thing.
"""
    return f"""Complete the heldout task below.

Mode: {mode}
Use only the provided user-visible examples, reusable skill, task input, and material excerpts.
Do not mention benchmark construction, hidden rubrics, or evaluation metadata.
Return only the final answer.
{deliverable_guidance}
{mode_guidance}

Task ID: {task_id}
{examples_text}
{skill_text}

Heldout task input:
{task["task_input"]}

Material excerpts:
{materials}
"""
