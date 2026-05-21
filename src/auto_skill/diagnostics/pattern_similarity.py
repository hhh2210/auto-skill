"""Example-pattern similarity evaluation for heldout outputs."""

from __future__ import annotations

import math
from typing import Any

from auto_skill.metrics.eval_summary import summarize_score_rows
from auto_skill.mvp import format_user_example, parse_json_object
from auto_skill.schemas import UserExample

SCHEMA_VERSION = "pattern-similarity-eval/v1"
EVALUATOR_KIND = "qwen_example_pattern_similarity"
SKILL_AWARE_EVALUATOR_KIND = "qwen_example_pattern_similarity_skill_aware_debug"


def build_pattern_similarity_prompt(
    *,
    examples: list[UserExample],
    heldout_task: dict[str, Any],
    mode: str,
    candidate_output: str,
    skill_md: str | None = None,
) -> str:
    examples_text = "\n\n".join(format_user_example(example) for example in examples)
    skill_text = f"\n\nCandidate skill debug context:\n{skill_md}" if skill_md else ""
    return f"""Evaluate whether the candidate output follows the reusable
patterns visible in the user examples.

Use only the user-visible examples, heldout task input, and candidate output.
If candidate skill debug context is present, this is a skill-aware debug run and
must not be used as a blind output-pattern metric.
Do not use hidden benchmark rubrics, private checklists, heldout feedback, or official scores.

Focus on "does this look like the examples in the reusable ways?" not whether
it is factually perfect.
Penalize:
- missing structural patterns that appear across examples;
- wrong tone/style for the example family;
- failing explicit count/length/format constraints visible in examples;
- copying example-specific facts, names, numbers, or one-off details into the heldout answer;
- hallucinating a pattern that is not supported by the examples.

Return strict JSON with:
- pattern_similarity_score: number from 1 to 10
- structural_similarity_score: number from 1 to 10
- style_similarity_score: number from 1 to 10
- constraint_transfer_score: number from 1 to 10
- unsupported_pattern_risk: one of "low", "medium", "high"
- supported_patterns: list of concise strings
- missed_patterns: list of concise strings
- unsupported_patterns: list of concise strings
- rationale: concise explanation

Mode: {mode}

User examples:
{examples_text}
{skill_text}

Heldout task input:
{heldout_task["task_input"]}

Candidate output:
{candidate_output}
"""


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score) or score < 1 or score > 10:
        return None
    return score


def parse_pattern_similarity_report(text: str) -> dict[str, Any]:
    parsed = parse_json_object(text)
    if "parse_error" in parsed:
        return parsed
    report = dict(parsed)
    for key in (
        "pattern_similarity_score",
        "structural_similarity_score",
        "style_similarity_score",
        "constraint_transfer_score",
    ):
        score = _finite_score(report.get(key))
        if score is None:
            report["parse_error"] = f"{key}_must_be_number_1_to_10"
            break
        report[key] = score
    risk = report.get("unsupported_pattern_risk")
    if "parse_error" not in report and risk not in {"low", "medium", "high"}:
        report["parse_error"] = "unsupported_pattern_risk_must_be_low_medium_or_high"
    for key in ("supported_patterns", "missed_patterns", "unsupported_patterns"):
        if "parse_error" not in report and not isinstance(report.get(key), list):
            report["parse_error"] = f"{key}_must_be_list"
    return report


def pattern_similarity_status(
    *,
    finish_reason: str | None,
    report: dict[str, Any],
) -> str:
    if finish_reason != "stop":
        return "judge_incomplete"
    if "parse_error" in report:
        return "judge_parse_error"
    return "success"


def summarize_pattern_similarity_rows(
    rows: list[dict[str, Any]],
    *,
    expected_cells: list[tuple[str, str, str]] | None = None,
    evaluator_kind: str = EVALUATOR_KIND,
) -> dict[str, Any]:
    return summarize_score_rows(
        rows,
        evaluator_kind=evaluator_kind,
        score_summary_key="mean_pattern_similarity_score",
        expected_cells=expected_cells,
    )
