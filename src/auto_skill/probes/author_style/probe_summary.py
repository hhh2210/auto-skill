"""Summary helpers for author-style T2 probe outputs."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from auto_skill.io.jsonl import load_jsonl
from auto_skill.metrics.author_style_reference import summarize_reference_retrieval_rows
from auto_skill.probes.author_style.probe_report import render_markdown


@dataclass(frozen=True)
class ProbeSummaryConfig:
    main_oracle: Path
    summary_out: Path
    report_md: Path
    qwen_crosscheck: Path | None = None
    mimo_crosscheck: Path | None = None
    pairwise_summary: Path | None = None
    legacy_strict: list[Path] = field(default_factory=list)


def annotate(row: dict[str, Any]) -> dict[str, Any]:
    pack_id = str(row.get("pack_id") or "")
    length_match = re.match(r"probe_blog_length_(.+)_\d{4}$", pack_id)
    row = dict(row)
    if length_match:
        row["probe_kind"] = "length_bucket_oracle"
        row["corpus"] = "blog"
        row["hard_neg_variant"] = "length_bucket"
        row["length_bucket"] = length_match.group(1)
        return row
    match = re.match(r"probe_(blog|reddit)_[^_]+_\d{2}_(.+)$", pack_id)
    row["probe_kind"] = "author_oracle"
    row["corpus"] = match.group(1) if match else str(row.get("source") or "unknown")
    row["hard_neg_variant"] = match.group(2) if match else "unknown"
    return row


def accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("status") == "success"]
    correct = [row for row in success if row.get("oracle_correct") is True]
    return {
        "success": len(success),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(success), 6) if success else None,
    }


def grouped_accuracy(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[" | ".join(str(row.get(key) or "none") for key in keys)].append(row)
    return {key: accuracy(value) for key, value in sorted(groups.items())}


def crosscheck(main_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    main = {(row.get("pack_id"), row.get("task_id")): row for row in main_rows}
    success = [row for row in rows if row.get("status") == "success"]
    comparable = [row for row in success if (row.get("pack_id"), row.get("task_id")) in main]
    selected_agree = sum(
        1
        for row in comparable
        if row.get("selected_candidate_id")
        == main[(row.get("pack_id"), row.get("task_id"))].get("selected_candidate_id")
    )
    correct_agree = sum(
        1
        for row in comparable
        if row.get("oracle_correct")
        == main[(row.get("pack_id"), row.get("task_id"))].get("oracle_correct")
    )
    overall = accuracy(success)
    return {
        "rows": len(rows),
        "success": len(success),
        "correct": overall.get("correct"),
        "accuracy": overall.get("accuracy"),
        "by_source": grouped_accuracy(success, ("source",)),
        "selected_agreement_with_main": round(selected_agree / len(comparable), 6)
        if comparable
        else None,
        "correctness_agreement_with_main": round(correct_agree / len(comparable), 6)
        if comparable
        else None,
        "status_counts": dict(Counter(str(row.get("status") or "unknown") for row in rows)),
    }


def summarize_probe(config: ProbeSummaryConfig) -> dict[str, Any]:
    main_rows = [annotate(row) for row in load_jsonl(config.main_oracle)]
    qwen_rows = (
        [annotate(row) for row in load_jsonl(config.qwen_crosscheck)]
        if config.qwen_crosscheck
        else []
    )
    mimo_rows = (
        [annotate(row) for row in load_jsonl(config.mimo_crosscheck)]
        if config.mimo_crosscheck
        else []
    )
    pairwise = json.loads(config.pairwise_summary.read_text()) if config.pairwise_summary else {}
    return {
        "schema_version": "author-style-t2-probe-summary/v1",
        "main_oracle": {
            "overall": accuracy(main_rows),
            "status_counts": dict(
                Counter(str(row.get("status") or "unknown") for row in main_rows)
            ),
            "by_probe_corpus_variant": grouped_accuracy(
                [row for row in main_rows if row.get("probe_kind") == "author_oracle"],
                ("corpus", "hard_neg_variant"),
            ),
            "by_length_bucket": grouped_accuracy(
                [row for row in main_rows if row.get("probe_kind") == "length_bucket_oracle"],
                ("length_bucket",),
            ),
        },
        "crosscheck": {
            "qwen": crosscheck(main_rows, qwen_rows),
            "mimo": crosscheck(main_rows, mimo_rows),
        },
        "pairwise": pairwise,
        "legacy_strict": {
            path.stem: summarize_reference_retrieval_rows(load_jsonl(path))
            for path in config.legacy_strict
            if path.exists()
        },
    }


def write_probe_summary(config: ProbeSummaryConfig) -> dict[str, Any]:
    summary = summarize_probe(config)
    config.summary_out.parent.mkdir(parents=True, exist_ok=True)
    config.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    config.report_md.parent.mkdir(parents=True, exist_ok=True)
    config.report_md.write_text(render_markdown(summary), encoding="utf-8")
    return summary
