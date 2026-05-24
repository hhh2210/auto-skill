"""Report rendering for the author-style reference-retrieval metric."""

from __future__ import annotations

import json
from typing import Any

PUBLIC_PARSE_ERROR_CODES = {
    "json_parse_error",
    "json_root_is_not_object",
    "most_similar_candidate_id_must_match_candidate_id",
    "ranked_candidate_ids_must_be_nonempty_list",
    "ranked_candidate_ids_must_match_candidate_ids",
    "ranked_candidate_ids_must_not_repeat",
    "ranked_candidate_ids_first_must_equal_selected",
    "confidence_must_be_low_medium_or_high",
    "rationale_must_be_string",
}


def public_reference_retrieval_report(report: dict[str, Any]) -> dict[str, Any]:
    if "parse_error" in report:
        return {"parse_error": _public_parse_error(report["parse_error"])}
    public_keys = (
        "most_similar_candidate_id",
        "ranked_candidate_ids",
        "confidence",
        "parse_error",
    )
    return {key: report[key] for key in public_keys if key in report}


def public_reference_retrieval_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_attempts = []
    for attempt in attempts:
        public_attempt = {
            "attempt": attempt.get("attempt"),
            "status": attempt.get("status"),
            "model": attempt.get("model"),
            "usage": attempt.get("usage"),
            "finish_reason": attempt.get("finish_reason"),
            "request_id": attempt.get("request_id"),
            "parse_error": _public_parse_error(attempt.get("parse_error")),
        }
        if attempt.get("status") == "success":
            public_attempt["selected_candidate_id"] = attempt.get("selected_candidate_id")
        public_attempts.append(public_attempt)
    return public_attempts


def reference_retrieval_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Author-Style Reference Retrieval Oracle",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Success rows | {summary.get('success', 0)} |",
        f"| Oracle correct | {summary.get('correct', 0)} |",
        f"| Accuracy | {_pct(summary.get('accuracy'))} |",
        f"| Mean expected rank | {_fmt(summary.get('mean_expected_rank'))} |",
        f"| Median expected rank | {_fmt(summary.get('median_expected_rank'))} |",
        "",
        "## By Source",
        "",
        "| Source | Success | Correct | Accuracy | Mean rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, row in sorted(summary.get("by_source", {}).items()):
        lines.append(
            f"| {source} | {row.get('success', 0)} | {row.get('correct', 0)} | "
            f"{_pct(row.get('accuracy'))} | {_fmt(row.get('mean_expected_rank'))} |"
        )
    for title, key in (
        ("By Hard Negative Variant", "by_hard_neg_variant"),
        ("By Source Hard Negative Variant", "by_source_hard_neg_variant"),
        ("By Length Bucket", "by_length_bucket"),
    ):
        rows = summary.get(key) or {}
        if not rows:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Bucket | Success | Correct | Accuracy | Mean rank |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for bucket, row in sorted(rows.items()):
            lines.append(
                f"| {bucket} | {row.get('success', 0)} | {row.get('correct', 0)} | "
                f"{_pct(row.get('accuracy'))} | {_fmt(row.get('mean_expected_rank'))} |"
            )
    lines.extend(
        [
            "",
            "## Status Counts",
            "",
            "```json",
            json.dumps(
                summary.get("status_counts", {}),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _public_parse_error(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value in PUBLIC_PARSE_ERROR_CODES:
        return value
    return "json_parse_error"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)
