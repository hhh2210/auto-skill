"""MVP helpers for example-driven auto-skill induction and evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from auto_skill.example_packs import material_context
from auto_skill.schemas import UserExample

HELDOUT_GENERATION_PROMPT_VERSION = "deliverable-priority-v2"


@dataclass(frozen=True)
class PromptRunResult:
    """Result from one model call used by the MVP scripts."""

    text: str
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    duration_seconds: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillInductionResult:
    """A generated skill package plus the evidence used to build it."""

    pack_id: str
    mode: str
    skill_md: str
    feature_reports: list[dict[str, Any]] = field(default_factory=list)
    cross_example_report: dict[str, Any] | None = None
    leave_one_out: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = "skill-induction/v1"
        row["status"] = "success"
        row["solver_model"] = _first_call_model(self.model_calls)
        return row


def _first_call_model(model_calls: list[dict[str, Any]]) -> str | None:
    """Return the first non-empty ``model`` string from a model_calls list."""

    for call in model_calls:
        if not isinstance(call, dict):
            continue
        model = call.get("model")
        if isinstance(model, str) and model.strip():
            return model
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in an LLM response."""

    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match is not None:
        candidates.insert(0, fence_match.group(1).strip())

    last_error = "no_json_object_found"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if match is None:
                continue
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as nested_exc:
                last_error = str(nested_exc)
                continue
        if isinstance(parsed, dict):
            return parsed
        return {"raw_text": text, "parse_error": "json_root_is_not_object"}
    return {"raw_text": text, "parse_error": last_error}


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
        "examples_plus_one_shot_skill",
        "examples_plus_feature_skill",
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


def build_task_first_feature_signature_prompt(
    *,
    task: dict[str, Any],
    feature_signatures: str | None,
    mode: str = "task_first_feature_signatures",
    max_material_chars: int = 4000,
) -> str:
    """Build a task-grounded prompt that uses signatures only after the task."""

    task_id = task.get("task_id") or task.get("example_id") or task.get("source_task_id")
    materials = material_context(task.get("materials", []), max_chars=max_material_chars)
    signatures = feature_signatures or "(missing feature signatures)"
    anchor_modes = {
        "task_first_operational_anchors",
        "task_first_evidence_anchored_operational_anchors",
        "task_first_two_level_operational_anchors",
    }
    signature_label = "Reusable feature signatures"
    if mode in anchor_modes:
        signature_label = "Reusable operational anchors"
    evidence_policy = ""
    evidence_inventory = ""
    if mode in anchor_modes:
        evidence_policy = """
Evidence policy for operational anchors:
- Treat anchors as requests for detail types, not permission to invent facts.
- Before using any concrete name, number, citation, route, cost, method,
  dataset, tool, case study, outcome, date, standard, or compatibility claim,
  verify that it appears in the current task input or material excerpts.
- If the current evidence does not provide a concrete value, write a generic
  but useful treatment instead of fabricating a specific detail.
- It is acceptable to say that the current materials do not specify a concrete
  value when the task asks for one and no evidence is available.
- Do not create fictional examples, named cases, statistics, references, or
  vendor/product capabilities to satisfy an anchor slot.
"""
    if mode == "task_first_evidence_anchored_operational_anchors":
        evidence_inventory = (
            "\nCurrent evidence inventory for concrete facts:\n"
            f"{build_current_evidence_inventory(task)}\n"
        )
    if mode == "task_first_two_level_operational_anchors":
        evidence_inventory = (
            "\nCurrent evidence inventory for specific factual claims:\n"
            f"{build_current_evidence_inventory(task, strict_whitelist=False)}\n"
        )
        evidence_policy += """
Two-level evidence policy:
- Level 1 specific facts: names, numbers, citations, routes, costs, methods,
  datasets, tools, cases, outcomes, dates, standards, and compatibility claims
  must be grounded in the current evidence or the evidence inventory.
- Level 2 generic scaffolding: common advice, generic section framing, and
  broadly known background can be used when useful, but it must stay generic and
  must not be presented as a task-specific fact.
- If the task asks for a specific value that current evidence lacks, provide a
  generic decision rule or state that the materials do not specify the value.
"""
    return f"""Complete the heldout task below.

Mode: {mode}
Use the heldout task input and material excerpts as the source of truth.
Return only the final answer.

Current-task deliverable priority:
- Produce the completed artifact requested by the heldout task input.
- Do not answer with a plan, outline, checklist, analysis, rubric mapping, or
  explanation of how to solve the task unless the heldout task explicitly asks
  for that artifact type.
- Treat the current task and current materials as higher priority than all
  reusable feature signatures.
- Use feature signatures only as secondary guidance for transferable structure,
  tone, and self-checks.
- Do not import task-specific facts, domains, entities, section topics, or
  numeric details from training examples unless they appear in the current task.
{evidence_policy}

Task ID: {task_id}

Heldout task input:
{task["task_input"]}

Material excerpts:
{materials}
{evidence_inventory}

{signature_label}:
{signatures}
"""


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


def _append_bullets(sections: list[str], title: str, items: list[str]) -> None:
    if not items:
        return
    sections.extend(["", f"## {title}"])
    sections.extend(f"- {item}" for item in items)


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(_single_line(item))
        elif isinstance(item, dict):
            text = item.get("feature") or item.get("rule") or item.get("description")
            if isinstance(text, str) and text.strip():
                items.append(_single_line(text))
        if len(items) >= limit:
            break
    return items


def _candidate_rule_lines(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    lines = []
    for item in value:
        if isinstance(item, str):
            rule = item
            support = None
        elif isinstance(item, dict):
            rule = item.get("rule")
            support = item.get("support_count")
            if not isinstance(support, int):
                supporting = item.get("supporting_examples")
                support = len(supporting) if isinstance(supporting, list) else None
        else:
            continue
        if not isinstance(rule, str) or not rule.strip():
            continue
        suffix = f" (support={support})" if isinstance(support, int) else ""
        lines.append(_single_line(rule) + suffix)
        if len(lines) >= limit:
            break
    return lines


def _described_item_lines(value: Any, *, label_key: str, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    lines = []
    for item in value:
        if isinstance(item, str) and item.strip():
            lines.append(_single_line(item))
        elif isinstance(item, dict):
            label = item.get(label_key)
            description = item.get("description") or item.get("reason")
            if isinstance(description, str) and description.strip():
                prefix = f"{label}: " if isinstance(label, str) and label.strip() else ""
                lines.append(prefix + _single_line(description))
        if len(lines) >= limit:
            break
    return lines


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _humanize_key(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        normalized = value.casefold()
        if not value or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value)
    return output


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


def build_judge_prompt(
    *,
    task: dict[str, Any],
    candidate_output: str,
    private_eval: dict[str, Any] | None,
) -> str:
    if private_eval is None:
        raise ValueError("private_eval is required for benchmark scoring")
    judge = (private_eval or {}).get("judge") or {}
    judge_reference = {
        key: value for key, value in judge.items() if key != "domain_common_prompt"
    }
    common_prompt = judge.get("domain_common_prompt")
    if isinstance(common_prompt, dict):
        judge_reference["domain_common_prompt_keys"] = sorted(common_prompt)
    criteria = evaluation_criteria(private_eval)
    if not criteria:
        raise ValueError("private_eval has no rubric/checklist criteria")
    criteria_text = json.dumps(criteria, ensure_ascii=False, indent=2)
    return f"""Score the candidate output for the task.

Use the benchmark rubric/checklist below only for evaluation. It is not user-visible input.
This is an LLM rubric/checklist surrogate judge, not an official visual/PPT evaluator.
Follow the JSON schema below even if the original benchmark prompt used a different output format.
Return strict JSON with:
- overall_score: number from 1 to 10
- criterion_scores: object mapping criterion names to 1-10 scores
- rationale: concise evidence-based explanation
- major_failures: list of important issues

Judge metadata:
{json.dumps(judge_reference, ensure_ascii=False, indent=2)}

Rubric or checklist:
{criteria_text}

Task input:
{task["task_input"]}

Candidate output:
{candidate_output}
"""


def evaluation_criteria(private_eval: dict[str, Any] | None) -> list[Any]:
    """Extract actual rubric/checklist criteria from private eval metadata."""

    if private_eval is None:
        return []
    supervision = private_eval.get("supervision") or {}
    items = supervision.get("items") or supervision.get("criteria")
    if isinstance(items, list) and items:
        return items
    checklists = supervision.get("checklists")
    flattened = []
    if isinstance(checklists, dict):
        for group_name, group_items in sorted(checklists.items()):
            if not isinstance(group_items, list):
                continue
            for index, item in enumerate(group_items, start=1):
                flattened.append(
                    {
                        "group": group_name,
                        "index": index,
                        "requirement": item,
                    }
                )
    return flattened


def extract_overall_score(judge_report: dict[str, Any]) -> float | None:
    value = judge_report.get("overall_score")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
