"""Skill artifact metric helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from auto_skill.eval.modes import DEFAULT_ARTIFACT_COMPARE_MODES, DEFAULT_BASELINE_MODE
from auto_skill.metrics.eval_summary import ScoreCell, summarize_score_rows
from auto_skill.metrics.numeric import _safe_mean, numeric_summary


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
