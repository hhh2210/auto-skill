"""Prompt builders and deterministic helpers for example-driven skill induction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
