"""Evidence inventory and scaffolding prompts."""

from __future__ import annotations

import re
from typing import Any

from auto_skill.cleaning.packs import material_context
from auto_skill.prompts._format import _append_bullets, _dedupe_preserve_order


def build_current_evidence_inventory(
    task: dict[str, Any],
    *,
    max_items: int = 40,
    max_text_chars: int = 20000,
    strict_whitelist: bool = True,
) -> str:
    """Build a deterministic whitelist-style inventory from current task evidence."""

    chunks = [str(task.get("task_input") or "")]
    for material in task.get("materials") or []:
        if not isinstance(material, dict):
            continue
        chunks.append(str(material.get("path") or ""))
        chunks.append(str(material.get("text") or ""))
    text = "\n".join(chunks)[:max_text_chars]
    numbers = _dedupe_preserve_order(re.findall(r"\b\d+(?:[.,:/-]\d+)*(?:%|[A-Za-z]+)?\b", text))
    quoted = _dedupe_preserve_order(
        match.strip()
        for match in re.findall(r"[\"'“”‘’《》](.*?)[\"'“”‘’《》]", text)
        if match.strip()
    )
    capitalized = _dedupe_preserve_order(
        match.strip()
        for match in re.findall(
            r"\b(?:[A-Z][A-Za-z0-9&+.-]*)(?:\s+[A-Z][A-Za-z0-9&+.-]*){0,5}\b",
            text,
        )
        if len(match.strip()) > 1
    )
    cjk_terms = _dedupe_preserve_order(
        match.strip()
        for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9·-]{2,20}", text)
        if any("\u4e00" <= char <= "\u9fff" for char in match)
    )

    if strict_whitelist:
        sections = [
            "Use this inventory as a whitelist for concrete facts. If a concrete fact is "
            "not present here or in the current evidence above, keep it generic.",
        ]
    else:
        sections = [
            "Use this inventory as the specific-fact bank. Concrete task-specific "
            "claims should come from this bank or the current evidence above. Generic "
            "scaffolding is allowed only when it stays generic.",
        ]
    _append_bullets(
        sections,
        "Numbers and dated values found in current evidence",
        numbers[:max_items],
    )
    _append_bullets(sections, "Quoted or explicitly named phrases", quoted[:max_items])
    _append_bullets(sections, "Capitalized entities and technical terms", capitalized[:max_items])
    _append_bullets(sections, "Chinese terms found in current evidence", cjk_terms[:max_items])
    if len(sections) == 1:
        sections.append("- No concrete facts were extracted deterministically.")
    return "\n".join(sections).strip()


def build_evidence_scaffolding_plan_prompt(
    *,
    task: dict[str, Any],
    operational_anchors: str | None,
    max_material_chars: int = 4000,
) -> str:
    """Build a planner prompt that separates grounded facts from generic scaffolding."""

    task_id = task.get("task_id") or task.get("example_id") or task.get("source_task_id")
    materials = material_context(task.get("materials", []), max_chars=max_material_chars)
    anchors = operational_anchors or "(missing operational anchors)"
    inventory = build_current_evidence_inventory(task, strict_whitelist=False)
    return f"""Plan the evidence use for a heldout task before final generation.

Return strict JSON only. Do not write the final answer.

Task ID: {task_id}

Heldout task input:
{task["task_input"]}

Material excerpts:
{materials}

Current evidence inventory:
{inventory}

Reusable operational anchors:
{anchors}

Rules:
- grounded_facts must contain only concrete facts visible in the heldout task
  input, material excerpts, or evidence inventory.
- generic_scaffolding may contain common structure, generic advice, or
  non-factual connective framing, but no task-specific names, numbers, outcomes,
  citations, routes, costs, case details, compatibility claims, or dates.
- missing_specifics should list requested concrete details that are not present
  in current evidence and therefore must not be fabricated.
- generation_constraints should state how the final answer should use facts and
  generic scaffolding without inventing unsupported specifics.

JSON schema:
{{
  "grounded_facts": ["specific fact from current evidence"],
  "generic_scaffolding": ["generic non-factual structure or advice"],
  "missing_specifics": ["specific detail type missing from current evidence"],
  "generation_constraints": ["constraint for final generation"]
}}
"""


EVIDENCE_PLAN_FIELDS = (
    "grounded_facts",
    "generic_scaffolding",
    "missing_specifics",
    "generation_constraints",
)


def validate_evidence_plan(plan: dict[str, Any]) -> list[str]:
    """Return schema errors for a staged evidence/scaffolding plan."""

    errors: list[str] = []
    for field_name in EVIDENCE_PLAN_FIELDS:
        value = plan.get(field_name)
        if not isinstance(value, list):
            errors.append(f"{field_name}_not_list")
            continue
        if any(not isinstance(item, str) or not item.strip() for item in value):
            errors.append(f"{field_name}_contains_non_string")
    return errors
