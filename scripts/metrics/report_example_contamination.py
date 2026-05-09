#!/usr/bin/env python3
"""Report train-example phrase overlap that leaks into heldout outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def ngrams(tokens: list[str], n: int) -> set[str]:
    if n <= 0:
        raise ValueError("n must be positive")
    return {
        " ".join(tokens[index : index + n])
        for index in range(0, max(0, len(tokens) - n + 1))
        if any(token not in STOPWORDS for token in tokens[index : index + n])
    }


def text_ngrams(text: str, *, n: int) -> set[str]:
    return ngrams(tokenize(text), n)


def train_example_text(pack: dict[str, Any]) -> str:
    chunks: list[str] = []
    for example in pack.get("train_examples") or []:
        if not isinstance(example, dict):
            continue
        chunks.append(str(example.get("task_input") or ""))
        for material in example.get("materials") or []:
            if isinstance(material, dict):
                chunks.append(str(material.get("path") or ""))
        desired = example.get("desired_output")
        if isinstance(desired, dict):
            chunks.append(str(desired.get("text") or ""))
    return "\n".join(chunks)


def heldout_task_text(pack: dict[str, Any], task_id: str) -> str:
    for task in pack.get("heldout_tasks") or []:
        if not isinstance(task, dict) or str(task.get("task_id") or "") != task_id:
            continue
        chunks = [str(task.get("task_input") or "")]
        for material in task.get("materials") or []:
            if isinstance(material, dict):
                chunks.append(str(material.get("path") or ""))
                chunks.append(str(material.get("text") or ""))
        return "\n".join(chunks)
    return ""


def generation_text(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if not isinstance(generation, dict):
        return ""
    text = generation.get("text")
    return text if isinstance(text, str) else ""


def pack_index(packs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(pack.get("pack_id") or ""): pack
        for pack in packs
        if isinstance(pack.get("pack_id"), str)
    }


def contamination_row(
    eval_row: dict[str, Any],
    pack: dict[str, Any],
    *,
    n: int,
    top_k: int,
) -> dict[str, Any]:
    pack_id = str(eval_row.get("pack_id") or "")
    task_id = str(eval_row.get("task_id") or "")
    mode = str(eval_row.get("mode") or "")
    train_phrases = text_ngrams(train_example_text(pack), n=n)
    heldout_phrases = text_ngrams(heldout_task_text(pack, task_id), n=n)
    candidate_phrases = text_ngrams(generation_text(eval_row), n=n)
    suspicious = sorted((train_phrases - heldout_phrases) & candidate_phrases)
    candidate_tokens = tokenize(generation_text(eval_row))
    candidate_counts = Counter(
        " ".join(candidate_tokens[index : index + n])
        for index in range(0, max(0, len(candidate_tokens) - n + 1))
    )
    top_matches = [
        {"phrase": phrase, "candidate_count": candidate_counts.get(phrase, 0)}
        for phrase in sorted(
            suspicious,
            key=lambda phrase: (-candidate_counts.get(phrase, 0), phrase),
        )[:top_k]
    ]
    return {
        "schema_version": "example-contamination-report/v1",
        "pack_id": pack_id,
        "task_id": task_id,
        "mode": mode,
        "status": eval_row.get("status"),
        "ngram_n": n,
        "train_unique_ngrams": len(train_phrases),
        "heldout_unique_ngrams": len(heldout_phrases),
        "candidate_unique_ngrams": len(candidate_phrases),
        "suspicious_train_only_candidate_ngrams": len(suspicious),
        "suspicious_rate_vs_candidate": (
            len(suspicious) / len(candidate_phrases) if candidate_phrases else 0.0
        ),
        "top_matches": top_matches,
    }


def build_contamination_report(
    *,
    packs: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    n: int = 3,
    top_k: int = 20,
    baseline_mode: str = "prompt_only",
) -> list[dict[str, Any]]:
    packs_by_id = pack_index(packs)
    rows = []
    for row in eval_rows:
        if row.get("status") != "success":
            continue
        pack = packs_by_id.get(str(row.get("pack_id") or ""))
        if pack is None:
            continue
        rows.append(contamination_row(row, pack, n=n, top_k=top_k))
    baseline_counts = {
        (row["pack_id"], row["task_id"]): row["suspicious_train_only_candidate_ngrams"]
        for row in rows
        if row.get("mode") == baseline_mode
    }
    for row in rows:
        baseline_count = baseline_counts.get((row["pack_id"], row["task_id"]))
        row["baseline_mode"] = baseline_mode
        row["baseline_suspicious_train_only_candidate_ngrams"] = baseline_count
        row["excess_suspicious_ngrams_vs_baseline"] = (
            row["suspicious_train_only_candidate_ngrams"] - baseline_count
            if baseline_count is not None
            else None
        )
    return rows


def summarize_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode") or ""), []).append(row)
    return {
        "schema_version": "example-contamination-summary/v1",
        "rows": len(rows),
        "modes": {
            mode: {
                "count": len(items),
                "mean_suspicious_ngrams": sum(
                    item["suspicious_train_only_candidate_ngrams"] for item in items
                )
                / len(items),
                "mean_suspicious_rate_vs_candidate": sum(
                    item["suspicious_rate_vs_candidate"] for item in items
                )
                / len(items),
                "mean_excess_suspicious_ngrams_vs_baseline": (
                    sum(
                        item["excess_suspicious_ngrams_vs_baseline"]
                        for item in items
                        if item["excess_suspicious_ngrams_vs_baseline"] is not None
                    )
                    / len(
                        [
                            item
                            for item in items
                            if item["excess_suspicious_ngrams_vs_baseline"] is not None
                        ]
                    )
                    if any(
                        item["excess_suspicious_ngrams_vs_baseline"] is not None
                        for item in items
                    )
                    else None
                ),
            }
            for mode, items in sorted(by_mode.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--ngram-n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--baseline-mode", default="prompt_only")
    args = parser.parse_args()

    rows = build_contamination_report(
        packs=load_jsonl(args.packs),
        eval_rows=load_jsonl(args.eval),
        n=args.ngram_n,
        top_k=args.top_k,
        baseline_mode=args.baseline_mode,
    )
    write_jsonl(args.out, rows)
    summary = summarize_report(rows)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
