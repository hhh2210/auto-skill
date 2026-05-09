"""Experiment metrics for the auto-skill MVP artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from auto_skill.eval_summary import ScoreCell, summarize_score_rows
from auto_skill.mvp import parse_json_object, user_examples_from_pack

DEFAULT_BASELINE_MODE = "prompt_only"
DEFAULT_ARTIFACT_COMPARE_MODES = (
    "one_shot_skill_from_examples",
    "auto_skill_feature_driven_no_validation",
)


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "mean": _safe_mean(clean),
        "min": min(clean) if clean else None,
        "p50": _percentile(clean, 0.5),
        "max": max(clean) if clean else None,
    }


def score_and_negative_transfer_summary(
    rows: list[dict[str, Any]],
    *,
    evaluator_kind: str,
    score_summary_key: str = "mean_overall_score",
    baseline_mode: str = DEFAULT_BASELINE_MODE,
    expected_cells: Iterable[ScoreCell] | None = None,
) -> dict[str, Any]:
    """Summarize benchmark scores, paired deltas, and negative transfer rates."""

    summary = summarize_score_rows(
        rows,
        evaluator_kind=evaluator_kind,
        score_summary_key=score_summary_key,
        baseline_mode=baseline_mode,
        expected_cells=expected_cells,
    )
    negative_transfer: dict[str, Any] = {}
    for mode, delta_summary in summary.get("paired_deltas", {}).items():
        pair_count = int(delta_summary.get("count") or 0)
        expected_pairs = int(delta_summary.get("expected_pairs") or pair_count)
        missing_pairs = delta_summary.get("missing_pairs") or []
        losses = int(delta_summary.get("losses") or 0)
        negative_examples = delta_summary.get("negative_transfer_examples") or []
        severities = [
            abs(float(item["delta"]))
            for item in negative_examples
            if isinstance(item, dict) and isinstance(item.get("delta"), (int, float))
        ]
        negative_transfer[mode] = {
            "baseline_mode": baseline_mode,
            "expected_pair_count": expected_pairs,
            "paired_count": pair_count,
            "missing_pair_count": len(missing_pairs) if isinstance(missing_pairs, list) else 0,
            "missing_pair_rate": len(missing_pairs) / expected_pairs if expected_pairs else None,
            "paired_success_rate": pair_count / expected_pairs if expected_pairs else None,
            "negative_count": losses,
            "negative_transfer_rate_denominator": "paired_count",
            "negative_transfer_rate": losses / pair_count if pair_count else None,
            "negative_transfer_rate_on_successful_pairs": losses / pair_count
            if pair_count
            else None,
            "negative_transfer_rate_on_expected_pairs": losses / expected_pairs
            if expected_pairs
            else None,
            "mean_negative_delta_magnitude": _safe_mean(severities),
            "negative_transfer_examples": negative_examples,
        }
    summary["negative_transfer"] = negative_transfer
    return summary


def _normalize_heading(line: str) -> str:
    heading = re.sub(r"^#+\s*", "", line.strip()).strip()
    heading = re.sub(r"[:：]+$", "", heading).strip().lower()
    return heading


def _section_key(heading: str) -> str | None:
    if "required" in heading and "rule" in heading:
        return "required_rules"
    if "optional" in heading and "rule" in heading:
        return "optional_rules"
    if "do not generalize" in heading or "do-not-generalize" in heading:
        return "do_not_generalize_rules"
    return None


def _is_rule_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^[-*+]\s+\[[ xX]\]\s+", stripped):
        return True
    if re.match(r"^[-*+]\s+", stripped):
        return True
    return bool(re.match(r"^\d+[.)]\s+", stripped))


def count_skill_section_rules(skill_md: str) -> dict[str, int]:
    """Count list-style rules in standard SKILL.md sections."""

    counts = {
        "required_rules": 0,
        "optional_rules": 0,
        "do_not_generalize_rules": 0,
    }
    current: str | None = None
    for raw_line in skill_md.splitlines():
        line = raw_line.rstrip()
        if re.match(r"^#{1,6}\s+", line):
            current = _section_key(_normalize_heading(line))
            continue
        if current and _is_rule_line(line):
            counts[current] += 1
    return counts


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _candidate_support_count(rule: Any) -> int | None:
    if not isinstance(rule, dict):
        return None
    for key in ("support_count", "supported_count"):
        value = rule.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key in ("supporting_examples", "supported_examples", "evidence_examples"):
        value = rule.get(key)
        if isinstance(value, list):
            return len({str(item) for item in value})
    return None


def skill_artifact_metric(row: dict[str, Any]) -> dict[str, Any]:
    """Return one skill artifact metric row from a skill-induction JSON row."""

    skill_md = row.get("skill_md") if isinstance(row.get("skill_md"), str) else ""
    cross_report = row.get("cross_example_report")
    if not isinstance(cross_report, dict):
        cross_report = {}
    candidate_rules = cross_report.get("candidate_rules")
    if not isinstance(candidate_rules, list):
        candidate_rules = []
    support_counts = [
        count
        for count in (_candidate_support_count(rule) for rule in candidate_rules)
        if count is not None
    ]
    section_counts = count_skill_section_rules(skill_md)
    return {
        "pack_id": row.get("pack_id"),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "skill_char_length": len(skill_md),
        **section_counts,
        "conflicts_count": _list_len(cross_report.get("conflicts")),
        "outliers_count": _list_len(cross_report.get("outliers")),
        "candidate_rules_count": len(candidate_rules),
        "candidate_rule_support_counts": support_counts,
        "candidate_rule_support_distribution": dict(sorted(Counter(support_counts).items())),
        "has_cross_example_report": bool(cross_report),
    }


def skill_artifact_summary(
    rows: list[dict[str, Any]],
    *,
    compare_modes: tuple[str, str] = DEFAULT_ARTIFACT_COMPARE_MODES,
) -> dict[str, Any]:
    metrics = [skill_artifact_metric(row) for row in rows if row.get("status") == "success"]
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        mode = str(metric.get("mode") or "")
        if mode:
            by_mode[mode].append(metric)

    fields = [
        "skill_char_length",
        "required_rules",
        "optional_rules",
        "do_not_generalize_rules",
        "conflicts_count",
        "outliers_count",
        "candidate_rules_count",
    ]
    mode_summaries = {
        mode: {field: numeric_summary(item[field] for item in items) for field in fields}
        for mode, items in sorted(by_mode.items())
    }
    support_summaries = {
        mode: numeric_summary(
            count
            for item in items
            for count in item.get("candidate_rule_support_counts", [])
        )
        for mode, items in sorted(by_mode.items())
    }

    first_mode, second_mode = compare_modes
    by_pack_mode = {
        (str(item.get("pack_id")), str(item.get("mode"))): item
        for item in metrics
        if item.get("pack_id") and item.get("mode")
    }
    comparisons = []
    pack_ids = sorted({pack_id for pack_id, mode in by_pack_mode if mode in compare_modes})
    for pack_id in pack_ids:
        left = by_pack_mode.get((pack_id, first_mode))
        right = by_pack_mode.get((pack_id, second_mode))
        if left is None or right is None:
            continue
        comparisons.append(
            {
                "pack_id": pack_id,
                "left_mode": first_mode,
                "right_mode": second_mode,
                "deltas": {
                    field: right[field] - left[field]
                    for field in fields
                    if isinstance(left.get(field), (int, float))
                    and isinstance(right.get(field), (int, float))
                },
            }
        )

    return {
        "rows": metrics,
        "by_mode": mode_summaries,
        "candidate_rule_support_by_mode": support_summaries,
        "comparison": {
            "left_mode": first_mode,
            "right_mode": second_mode,
            "paired_pack_count": len(comparisons),
            "rows": comparisons,
            "mean_deltas": {
                field: _safe_mean(
                    [
                        float(item["deltas"][field])
                        for item in comparisons
                        if field in item["deltas"]
                    ]
                )
                for field in fields
            },
        },
    }


def _usage_tokens(usage: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def _empty_usage_tokens() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _add_usage(accumulator: dict[str, int], usage: dict[str, Any] | None) -> None:
    tokens = _usage_tokens(usage)
    for key, value in tokens.items():
        accumulator[key] += value


def _sum_usage_tokens(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        key: left[key] + right[key]
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def token_usage_summary(
    rows: list[dict[str, Any]],
    *,
    examples_only_mode: str = "few_shot_examples_only",
) -> dict[str, Any]:
    """Summarize heldout inference token usage from eval rows."""

    by_mode: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        item = by_mode.setdefault(
            mode,
            {
                "attempted_rows": 0,
                "successful_rows": 0,
                "status_counts": {},
                "layout_plan": _empty_usage_tokens(),
                "generation": _empty_usage_tokens(),
                "judge": _empty_usage_tokens(),
            },
        )
        item["attempted_rows"] += 1
        status = str(row.get("status") or "unknown")
        item["status_counts"][status] = item["status_counts"].get(status, 0) + 1
        generation = row.get("generation")
        if isinstance(generation, dict):
            _add_usage(item["generation"], generation.get("usage"))
        layout_plan = row.get("layout_plan")
        if isinstance(layout_plan, dict):
            _add_usage(item["layout_plan"], layout_plan.get("usage"))
        judge = row.get("judge")
        if isinstance(judge, dict):
            _add_usage(item["judge"], judge.get("usage"))
        for judge_call in row.get("judge_calls") or []:
            if isinstance(judge_call, dict):
                _add_usage(item["judge"], judge_call.get("usage"))
        if row.get("status") == "success":
            item["successful_rows"] += 1

    for item in by_mode.values():
        count = item["successful_rows"]
        attempted = item["attempted_rows"]
        item["combined_model_calls"] = _sum_usage_tokens(
            _sum_usage_tokens(item["layout_plan"], item["generation"]),
            item["judge"],
        )
        item["status_counts"] = dict(sorted(item["status_counts"].items()))
        for bucket in ("layout_plan", "generation", "judge"):
            totals = item[bucket]
            item[f"{bucket}_avg_per_success"] = {
                key: value / count if count else None for key, value in totals.items()
            }
            item[f"{bucket}_avg_per_attempt"] = {
                key: value / attempted if attempted else None for key, value in totals.items()
            }
        item["combined_model_calls_avg_per_success"] = {
            key: value / count if count else None
            for key, value in item["combined_model_calls"].items()
        }
        item["combined_model_calls_avg_per_attempt"] = {
            key: value / attempted if attempted else None
            for key, value in item["combined_model_calls"].items()
        }

    comparisons: dict[str, Any] = {}
    baseline = by_mode.get(examples_only_mode)
    if baseline:
        baseline_avg = baseline["generation_avg_per_success"]
        for mode, item in sorted(by_mode.items()):
            if mode == examples_only_mode:
                continue
            mode_avg = item["generation_avg_per_success"]
            comparisons[mode] = {
                key: {
                    "examples_only_avg": baseline_avg[key],
                    "mode_avg": mode_avg[key],
                    "delta": mode_avg[key] - baseline_avg[key]
                    if mode_avg[key] is not None and baseline_avg[key] is not None
                    else None,
                    "ratio": mode_avg[key] / baseline_avg[key]
                    if mode_avg[key] is not None and baseline_avg[key]
                    else None,
                }
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
    return {
        "by_mode": dict(sorted(by_mode.items())),
        "comparison_to_examples_only_generation": {
            "examples_only_mode": examples_only_mode,
            "modes": comparisons,
        },
    }


def skill_induction_token_usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize upfront skill-induction token usage from skill rows."""

    by_mode: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        item = by_mode.setdefault(
            mode,
            {
                "attempted_rows": 0,
                "successful_rows": 0,
                "status_counts": {},
                "model_calls": _empty_usage_tokens(),
                "by_stage": {},
            },
        )
        item["attempted_rows"] += 1
        status = str(row.get("status") or "unknown")
        item["status_counts"][status] = item["status_counts"].get(status, 0) + 1
        for call in row.get("model_calls") or []:
            if not isinstance(call, dict):
                continue
            _add_usage(item["model_calls"], call.get("usage"))
            stage = str(call.get("stage") or "unknown")
            stage_usage = item["by_stage"].setdefault(stage, _empty_usage_tokens())
            _add_usage(stage_usage, call.get("usage"))
        if row.get("status") == "success":
            item["successful_rows"] += 1

    for item in by_mode.values():
        count = item["successful_rows"]
        attempted = item["attempted_rows"]
        item["status_counts"] = dict(sorted(item["status_counts"].items()))
        item["model_calls_avg_per_success"] = {
            key: value / count if count else None
            for key, value in item["model_calls"].items()
        }
        item["model_calls_avg_per_attempt"] = {
            key: value / attempted if attempted else None
            for key, value in item["model_calls"].items()
        }
        item["by_stage"] = dict(sorted(item["by_stage"].items()))

    return {"by_mode": dict(sorted(by_mode.items()))}


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


def _json_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _literal_candidates_from_text(text: str) -> set[str]:
    candidates: set[str] = set()
    if not text:
        return candidates
    patterns = [
        r"\b\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d+(?:\.\d+)?%\b",
        r"\b\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?){1,4}\b",
        r"\b\d+(?:\.\d+)?\s*(?:元|万元|亿元|USD|RMB|dollars?|slides?|pages?)\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"https?://[^\s)]+",
        r"(?:^|[\s\"'`])(?:data|artifacts|runs|notes|docs|src|scripts|/Users)/[^\s\"'`)]+",
        r"\b(?:[A-Z][A-Za-z0-9&.-]+(?:\s+|[-:])){2,}[A-Z][A-Za-z0-9&.-]+\b",
        r"[\u4e00-\u9fffA-Za-z0-9]+(?:报告|论文|白皮书|合同|招标书|演示文稿|讲义|教材|章节|公司|大学|学院)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            literal = match.group(0).strip(" \t\n\"'`.,;:()[]{}")
            if len(literal) >= 3:
                candidates.add(literal)
    return candidates


def train_literal_candidates(pack: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for example in user_examples_from_pack(pack):
        candidates.add(example.example_id)
        source_task_id = example.metadata.get("source_task_id")
        if source_task_id:
            candidates.add(str(source_task_id))
        candidates.update(_literal_candidates_from_text(example.task_input))
        if example.output:
            candidates.update(_literal_candidates_from_text(example.output))
        candidates.update(str(material) for material in example.materials if str(material))
    return {item for item in candidates if len(item) >= 3}


def literal_leakage_report(
    *,
    pack: dict[str, Any],
    generated_signatures: Any,
) -> dict[str, Any]:
    """Detect sample-specific literals copied into abstract signatures."""

    haystack = _json_blob(generated_signatures)
    matches = sorted(
        literal for literal in train_literal_candidates(pack) if literal and literal in haystack
    )
    return {
        "candidate_literal_count": len(train_literal_candidates(pack)),
        "matched_literal_count": len(matches),
        "matched_literals": matches[:50],
        "has_literal_leakage": bool(matches),
    }


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
