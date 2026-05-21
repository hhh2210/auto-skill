"""Self-consistency diagnostic metric helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from auto_skill.llm.parse import parse_json_object
from auto_skill.metrics.numeric import numeric_summary
from auto_skill.mvp import user_examples_from_pack


def _example_signature_for_prompt(example: Any) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "task_input": example.task_input,
        "output": example.output,
        "materials": list(example.materials),
    }


def build_abstract_signatures_prompt(
    *,
    skill_md: str,
    n: int,
) -> str:
    return f"""Generate {n} abstract example signatures from this SKILL.md.

This is an eval-only skill encoding diagnostic. Use only the skill text.
Do not use benchmark rubrics, checklists, heldout feedback, official scores,
teacher traces, or any private metadata.

Do not reconstruct literal sample-specific details. Avoid source IDs, entity names,
document titles, paper titles, contact details, exact numeric strings, dates, paths,
payment ratios, or other identifiers that could have come from one train example.

Return strict JSON with this shape:
{{
  "abstract_example_signatures": [
    {{
      "task_family": "...",
      "abstract_task_input": "...",
      "expected_output_signature": "...",
      "stable_constraints": ["..."],
      "do_not_generalize": ["..."]
    }}
  ]
}}

SKILL.md:
{skill_md}
"""


def build_signature_consistency_judge_prompt(
    *,
    train_pack: dict[str, Any],
    abstract_signatures: list[dict[str, Any]],
    literal_leakage: dict[str, Any] | None = None,
) -> str:
    examples = [
        _example_signature_for_prompt(example)
        for example in user_examples_from_pack(train_pack)
    ]
    leakage_text = json.dumps(literal_leakage or {}, ensure_ascii=False, indent=2)
    return f"""Compare abstract example signatures against real user-visible train examples.

This is eval-only. Use only the real train examples and synthetic examples below.
Do not use private rubrics, checklists, benchmark judge traces, heldout feedback,
or official benchmark scores.

Score from 0 to 10 where higher is better, except penalties where higher is worse.
Return strict JSON with:
- stable_feature_recall: number 0-10
- constraint_recall: number 0-10
- structure_recall: number 0-10
- style_signature: number 0-10
- leakage_penalty: number 0-10
- unsupported_specificity_penalty: number 0-10
- overall_self_consistency: number 0-10
- matched_constraints: list of strings
- missing_or_distorted_constraints: list of strings
- rationale: concise explanation

Real train examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}

Abstract signatures:
{json.dumps(abstract_signatures, ensure_ascii=False, indent=2)}

Local literal leakage detector report:
{leakage_text}
"""


def parse_abstract_signatures(text: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    parsed = parse_json_object(text)
    signatures = parsed.get("abstract_example_signatures")
    if not isinstance(signatures, list):
        return []
    clean = [item for item in signatures if isinstance(item, dict)]
    return clean[:limit] if limit is not None else clean


SELF_CONSISTENCY_SCORE_FIELDS = (
    "stable_feature_recall",
    "constraint_recall",
    "structure_recall",
    "style_signature",
    "leakage_penalty",
    "unsupported_specificity_penalty",
    "overall_self_consistency",
)


def _finite_score_0_to_10(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if score < 0 or score > 10:
        return None
    return score


def parse_self_consistency_report(text: str) -> dict[str, Any]:
    """Parse and validate a self-consistency judge report."""

    parsed = parse_json_object(text)
    if "parse_error" in parsed:
        return parsed
    report = dict(parsed)
    for field in SELF_CONSISTENCY_SCORE_FIELDS:
        score = _finite_score_0_to_10(report.get(field))
        if score is None:
            report["parse_error"] = f"{field}_must_be_number_0_to_10"
            return report
        report[field] = score
    for field in ("matched_constraints", "missing_or_distorted_constraints"):
        if not isinstance(report.get(field), list):
            report["parse_error"] = f"{field}_must_be_list"
            return report
    if not isinstance(report.get("rationale"), str):
        report["parse_error"] = "rationale_must_be_string"
    return report


def self_consistency_report_errors(report: Any) -> list[str]:
    if not isinstance(report, dict):
        return ["judge_report_must_be_object"]
    errors = []
    if "parse_error" in report:
        errors.append(str(report["parse_error"]))
    for field in SELF_CONSISTENCY_SCORE_FIELDS:
        if _finite_score_0_to_10(report.get(field)) is None:
            errors.append(f"{field}_must_be_number_0_to_10")
    for field in ("matched_constraints", "missing_or_distorted_constraints"):
        if not isinstance(report.get(field), list):
            errors.append(f"{field}_must_be_list")
    if not isinstance(report.get("rationale"), str):
        errors.append("rationale_must_be_string")
    return sorted(set(errors))


def self_consistency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the eval-only skill encoding self-consistency diagnostic."""

    by_mode: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    leakage_by_mode: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "successful_rows": 0,
            "literal_leakage_row_count": 0,
            "matched_literal_count": 0,
        }
    )
    status_counts: Counter[str] = Counter()
    successful_rows = []
    invalid_success_rows = []
    diagnostic_warnings = []
    for row in rows:
        status_counts[str(row.get("status") or "unknown")] += 1
        if row.get("status") != "success":
            continue
        report = row.get("judge_report")
        errors = self_consistency_report_errors(report)
        if errors:
            invalid_success_rows.append(
                {
                    "pack_id": row.get("pack_id"),
                    "mode": row.get("mode"),
                    "errors": errors,
                }
            )
            continue
        assert isinstance(report, dict)
        values = {}
        mode = str(row.get("mode") or "unknown")
        leakage = row.get("literal_leakage")
        leakage_bucket = leakage_by_mode[mode]
        leakage_bucket["successful_rows"] += 1
        if isinstance(leakage, dict):
            matched = leakage.get("matched_literal_count")
            if isinstance(matched, int) and not isinstance(matched, bool):
                leakage_bucket["matched_literal_count"] += matched
            if leakage.get("has_literal_leakage"):
                leakage_bucket["literal_leakage_row_count"] += 1
        for field in SELF_CONSISTENCY_SCORE_FIELDS:
            value = float(report[field])
            values[field] = value
            by_mode[mode][field].append(value)
        if values:
            if (
                isinstance(leakage, dict)
                and leakage.get("has_literal_leakage")
                and values.get("overall_self_consistency", 0) >= 8
            ):
                diagnostic_warnings.append(
                    {
                        "pack_id": row.get("pack_id"),
                        "mode": row.get("mode"),
                        "warning": "high_self_consistency_with_literal_leakage",
                        "overall_self_consistency": values.get("overall_self_consistency"),
                        "matched_literal_count": leakage.get("matched_literal_count"),
                    }
                )
            successful_rows.append(
                {
                    "pack_id": row.get("pack_id"),
                    "mode": row.get("mode"),
                    "scores": values,
                    "literal_leakage": row.get("literal_leakage"),
                }
            )
    return {
        "diagnostic_kind": "skill_encoding_self_consistency",
        "main_performance_metric": False,
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "successful_rows": len(successful_rows),
        "invalid_success_rows": invalid_success_rows,
        "by_mode": {
            mode: {
                field: numeric_summary(values)
                for field, values in sorted(field_values.items())
            }
            for mode, field_values in sorted(by_mode.items())
        },
        "literal_leakage_by_mode": {
            mode: {
                **values,
                "literal_leakage_rate": (
                    values["literal_leakage_row_count"] / values["successful_rows"]
                    if values["successful_rows"]
                    else None
                ),
            }
            for mode, values in sorted(leakage_by_mode.items())
        },
        "diagnostic_warnings": diagnostic_warnings,
        "rows": successful_rows,
    }
