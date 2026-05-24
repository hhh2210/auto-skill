"""Markdown rendering for author-style T2 probe summaries."""

from __future__ import annotations

import json
from typing import Any


def render_markdown(summary: dict[str, Any]) -> str:
    lines = setup_lines()
    add_oracle_accuracy(lines, summary)
    add_length_bucket_table(lines, summary)
    add_pairwise_table(lines, summary)
    add_crosscheck_table(lines, summary)
    add_caveats(lines)
    if summary.get("legacy_strict"):
        lines.extend(["", "## Legacy Strict Control", "", "```json"])
        lines.append(
            json.dumps(summary["legacy_strict"], ensure_ascii=False, indent=2, sort_keys=True)
        )
        lines.append("```")
    return "\n".join(lines[:98]) + "\n"


def setup_lines() -> list[str]:
    return [
        "# Author-Style T2 Probe Results",
        "",
        "## Setup",
        "",
        "- Fresh minimal ingestion: normalized text length filter plus exact dedupe, "
        "no GPT author audit.",
        "- Main judge: gpt-5.5-high through Codex OAuth for all oracle triples.",
        "- Cross-check: Qwen3.5-Plus and MIMO on the same deterministic 10% subset.",
        "- Interpretation: these are judge-conditioned probes, not external truth labels.",
        "",
        "## Oracle Accuracy",
        "",
        "| Corpus / variant | Success | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]


def add_oracle_accuracy(lines: list[str], summary: dict[str, Any]) -> None:
    for key, row in summary["main_oracle"]["by_probe_corpus_variant"].items():
        lines.append(
            f"| {key.replace(' | ', ' / ')} | {row['success']} | {row['correct']} | "
            f"{pct(row['accuracy'])} |"
        )
    overall = summary["main_oracle"]["overall"]
    lines.extend(
        [
            "",
            f"Main oracle overall: {pct(overall['accuracy'])} "
            f"({overall['correct']}/{overall['success']}).",
            "Variant accuracies are close; this does not prove topic/length shortcuts are gone, "
            "because the negative pools are not orthogonal.",
        ]
    )


def add_length_bucket_table(lines: list[str], summary: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## Min-Words Buckets",
            "",
            "| Bucket | Success | Correct | Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ["0_50", "50_100", "100_200", "200_500", "500_plus"]:
        row = summary["main_oracle"]["by_length_bucket"].get(key)
        if row is not None:
            lines.append(
                f"| {key} | {row['success']} | {row['correct']} | {pct(row['accuracy'])} |"
            )


def add_pairwise_table(lines: list[str], summary: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## Pairwise Separation",
            "",
            "| Corpus | Within mean | Cross mean | KS D | AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for corpus, row in sorted(summary.get("pairwise", {}).get("by_corpus", {}).items()):
        lines.append(
            f"| {corpus} | {num(row.get('within_mean'))} | {num(row.get('cross_mean'))} | "
            f"{num(row.get('ks_d'))} | {num(row.get('auc_within_gt_cross'))} |"
        )
    lines.extend(
        [
            "",
            "Pairwise supports same-author as a useful noisy cluster, "
            "not as proof that each author has one stable style.",
        ]
    )


def add_crosscheck_table(lines: list[str], summary: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## Judge Cross-Check",
            "",
            "| Judge | Success | Correct | Accuracy | Blog | Reddit |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for judge in ("qwen", "mimo"):
        row = summary["crosscheck"].get(judge, {})
        blog = row.get("by_source", {}).get("personal_blog_history", {})
        reddit = row.get("by_source", {}).get("cross_topic_online_comment_history", {})
        lines.append(
            f"| {judge} | {row.get('success', 0)} | {row.get('correct', 0)} | "
            f"{pct(row.get('accuracy'))} | {pct(blog.get('accuracy'))} | "
            f"{pct(reddit.get('accuracy'))} |"
        )


def add_caveats(lines: list[str]) -> None:
    lines.extend(["", "## Adversarial Caveats", ""])
    lines.append(
        "- GPT-5.5 oracle accuracy should be read as GPT-5.5 retrieval signal, "
        "not an independent proof that minimal cleaning is correct."
    )
    lines.append(
        "- Qwen and MIMO are materially lower, especially on Reddit, "
        "so model-family robustness is not established."
    )
    lines.append(
        "- Length buckets do not justify a hard `min_words` cutoff yet; even `<50w` "
        "is above chance in this setup, but each bucket has only 80 triples."
    )
    lines.append(
        "- Minimal ingestion remains the right main baseline because it avoids GPT audit "
        "selection bias; report it as weakly filtered, not fully cleaned."
    )
    lines.extend(["", "## Case Analysis And Recommendation", ""])
    lines.append("- Read `_private` debug samples before treating any aggregate as final.")
    lines.append("- Keep minimal ingestion as the main baseline, not as a quality guarantee.")
    lines.append("- Keep `author id = style cluster`, but describe it as noisy.")
    lines.append("- Do not set a `min_words` threshold; use length stratification in reports.")
    lines.append(
        "- Redesign hard-negative ablations if we need a clean shortcut test; "
        "current pools are confounded."
    )


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def num(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3g}"
