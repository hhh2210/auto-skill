"""Prompt builders and deterministic helpers for example-driven skill induction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from auto_skill.cleaning.packs import material_context
from auto_skill.prompts.evidence import build_current_evidence_inventory
from auto_skill.schemas import UserExample

RuleStatus = Literal["required", "optional", "reject"]


@dataclass(frozen=True)
class RuleSupport:
    """Support evidence for one candidate skill rule."""

    rule: str
    supported_examples: tuple[str, ...] = ()
    contradicted_examples: tuple[str, ...] = ()


def classify_rule_support(
    support_count: int,
    total_examples: int,
    *,
    required_threshold: float = 0.8,
    optional_threshold: float = 0.5,
) -> RuleStatus:
    """Classify a rule based on how many examples support it."""

    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if support_count < 0 or support_count > total_examples:
        raise ValueError("support_count must be between 0 and total_examples")
    ratio = support_count / total_examples
    if ratio >= required_threshold:
        return "required"
    if ratio >= optional_threshold:
        return "optional"
    return "reject"


def merge_rule_supports(
    rules: list[RuleSupport],
    *,
    total_examples: int,
    required_threshold: float = 0.8,
    optional_threshold: float = 0.5,
) -> dict[RuleStatus, list[RuleSupport]]:
    """Group candidate rules into required, optional, and rejected buckets."""

    merged: dict[RuleStatus, list[RuleSupport]] = {
        "required": [],
        "optional": [],
        "reject": [],
    }
    for rule in rules:
        effective_support = max(
            0,
            len(set(rule.supported_examples)) - len(set(rule.contradicted_examples)),
        )
        status = classify_rule_support(
            effective_support,
            total_examples,
            required_threshold=required_threshold,
            optional_threshold=optional_threshold,
        )
        merged[status].append(rule)
    return merged


def build_feature_extraction_prompt(
    example: UserExample,
    *,
    emphasized_features: list[str] | None = None,
) -> str:
    """Build the first-stage prompt for extracting features from one user example."""

    emphasis = "\n".join(f"- {item}" for item in emphasized_features or [])
    if not emphasis:
        emphasis = "- No additional user-emphasized features."
    output = example.output or "(No explicit output text; inspect attached artifacts/materials.)"
    notes = example.user_notes or "(No user notes.)"
    materials = "\n".join(f"- {material}" for material in example.materials) or "- None"
    return f"""Extract reusable features from this user example.

Only use information visible in the example. Do not infer hidden rubrics or evaluation feedback.

Return only one strict JSON object. Do not wrap it in Markdown fences.
Escape any quotation marks inside string values. Use these top-level keys:
- basic_attributes
- content_features
- structure_features
- style_features
- must_not_generalize
- uncertainty

User-emphasized features:
{emphasis}

Example ID: {example.example_id}

Task input:
{example.task_input}

Output or artifact summary:
{output}

Materials:
{materials}

User notes:
{notes}
"""


def build_cross_example_analysis_prompt(feature_reports: list[str]) -> str:
    """Build the second-stage prompt for stable feature aggregation."""

    joined_reports = "\n\n---\n\n".join(feature_reports)
    return f"""Compare the feature reports across examples.

Return only one strict JSON object. Do not wrap it in Markdown fences.
Use these top-level keys:
- stable_features: features supported by most or all examples
- optional_features: features supported by a meaningful subset
- conflicts: features that contradict each other
- outliers: examples that should narrow the skill scope
- candidate_rules: concise rules with supporting example IDs

Do not promote a feature into a rule unless the reports provide evidence.

Feature reports:
{joined_reports}
"""


def build_skill_compilation_prompt(
    cross_example_report: str,
    examples: list[UserExample],
) -> str:
    """Build the final prompt for compiling a SKILL.md from aggregated evidence."""

    example_ids = ", ".join(example.example_id for example in examples)
    return f"""Create a reusable SKILL.md from the cross-example report.

The skill must be operational, scoped, and auditable.

Required sections:
- When to use this skill
- Inputs expected from the user
- Procedure
- Required rules
- Optional rules
- Do not generalize
- Self-check before final answer

Evidence examples: {example_ids}

Cross-example report:
{cross_example_report}
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
