"""Report rendering for the author-style reference-retrieval metric."""

from __future__ import annotations

import json
from typing import Any


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
