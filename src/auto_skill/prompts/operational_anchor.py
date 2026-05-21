"""Operational-anchor prompt builders."""

from __future__ import annotations

import json
from typing import Any

from auto_skill.cleaning.packs import material_context
from auto_skill.prompts._format import (
    _append_bullets,
    _candidate_rule_lines,
    _dedupe_preserve_order,
    _described_item_lines,
    _humanize_key,
    _string_list,
)


def build_planned_operational_anchor_prompt(
    *,
    task: dict[str, Any],
    operational_anchors: str | None,
    evidence_plan: dict[str, Any],
    max_material_chars: int = 4000,
) -> str:
    """Build the final generation prompt for the staged planner ablation."""

    task_id = task.get("task_id") or task.get("example_id") or task.get("source_task_id")
    materials = material_context(task.get("materials", []), max_chars=max_material_chars)
    anchors = operational_anchors or "(missing operational anchors)"
    return f"""Complete the heldout task below.

Mode: task_first_planned_operational_anchors
Use the heldout task input and material excerpts as the source of truth.
Return only the final answer.

Current-task deliverable priority:
- Produce the completed artifact requested by the heldout task input.
- Do not answer with a plan, outline, checklist, analysis, rubric mapping, or
  explanation of how to solve the task unless the heldout task explicitly asks
  for that artifact type.

Evidence/scaffolding plan:
{json.dumps(evidence_plan, ensure_ascii=False, indent=2)}

Final-generation rules:
- You may use grounded_facts as task-specific facts.
- You may use generic_scaffolding only as generic structure or common-sense
  framing; do not turn it into task-specific facts.
- Do not invent missing_specifics.
- If a requested concrete value is missing, keep the treatment generic or state
  that the current materials do not specify it.
- Use reusable operational anchors only to decide what detail types to consider,
  not to import training-example facts.

Task ID: {task_id}

Heldout task input:
{task["task_input"]}

Material excerpts:
{materials}

Reusable operational anchors:
{anchors}
"""


def build_feature_signature_context(skill_row: dict[str, Any], *, max_items: int = 8) -> str | None:
    """Build a compact example-derived context from a feature-driven skill row.

    This context is an ablation surface: it uses only structured fields produced
    from user-visible train examples and intentionally omits the full SKILL.md.
    """

    report = skill_row.get("cross_example_report")
    if not isinstance(report, dict):
        return None

    stable_features = _string_list(report.get("stable_features"), limit=max_items)
    candidate_rules = _candidate_rule_lines(report.get("candidate_rules"), limit=max_items)
    optional_features = _string_list(report.get("optional_features"), limit=max_items)
    conflicts = _described_item_lines(report.get("conflicts"), label_key="feature", limit=4)
    outliers = _described_item_lines(report.get("outliers"), label_key="example_id", limit=4)

    if not any([stable_features, candidate_rules, optional_features, conflicts, outliers]):
        return None

    sections = [
        "# Abstract Feature Signatures",
        "",
        "Use these as compact patterns abstracted from the visible training examples.",
        "They are not private rubrics, heldout feedback, or official scores.",
    ]
    _append_bullets(sections, "Stable features", stable_features)
    _append_bullets(sections, "Candidate rules", candidate_rules)
    _append_bullets(sections, "Optional or contextual features", optional_features)
    _append_bullets(sections, "Known variation/conflicts", conflicts)
    _append_bullets(sections, "Outlier notes", outliers)
    return "\n".join(sections).strip()


def build_operational_anchor_context(
    skill_row: dict[str, Any],
    *,
    max_items: int = 12,
) -> str | None:
    """Build current-task detail anchors from user-example feature reports."""

    feature_reports = skill_row.get("feature_reports")
    if not isinstance(feature_reports, list):
        return None
    slot_names: list[str] = []
    style_slots: list[str] = []

    for report in feature_reports:
        if not isinstance(report, dict):
            continue
        for key in ("basic_attributes", "content_features", "structure_features"):
            value = report.get(key)
            if isinstance(value, dict):
                slot_names.extend(_humanize_key(name) for name in value)
        style = report.get("style_features")
        if isinstance(style, dict):
            style_slots.extend(_humanize_key(name) for name in style)
        preserve = report.get("must_not_generalize")
        if isinstance(preserve, list) and preserve:
            slot_names.append("example-specific preservation constraints")

    slots = _dedupe_preserve_order(slot_names)[:max_items]
    styles = _dedupe_preserve_order(style_slots)[:6]
    if not any([slots, styles]):
        return None

    output = [
        "# Task-Grounded Operational Anchors",
        "",
        "Use these anchors to instantiate the current task, not to copy training-example facts.",
        "The heldout task and its materials remain the source of truth.",
    ]
    _append_bullets(
        output,
        "Detail slots to fill from the current task",
        slots,
    )
    _append_bullets(
        output,
        "Depth and coverage anchors",
        [
            "Turn concrete current-task entities, variables, methods, standards, datasets, "
            "dates, and case-study details into substantive sections or bullets.",
            "Only fill a detail slot with a concrete fact when that fact appears in the "
            "current task input or current materials.",
            "When the current evidence lacks a concrete value, keep the treatment generic "
            "or state that the available materials do not specify it.",
            "Prefer completed content over placeholders; use placeholders only when the "
            "current task explicitly asks for an outline or future-populated section.",
            "For every major current-task requirement, include both the high-level section "
            "and task-specific supporting detail.",
        ],
    )
    _append_bullets(output, "Style slots to infer from the current task", styles)
    _append_bullets(
        output,
        "Anti-leakage checks",
        [
            "Do not copy names, numbers, institutions, algorithms, journals, datasets, "
            "locations, or case details from training examples unless they appear in "
            "the current task or materials.",
            "If an anchor slot cannot be grounded in the current task, omit it or keep "
            "it generic instead of inventing details.",
        ],
    )
    return "\n".join(output).strip()
