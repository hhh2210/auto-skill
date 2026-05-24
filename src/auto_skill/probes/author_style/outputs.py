"""Output helpers for author-style probe artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from auto_skill.io.jsonl import write_jsonl
from auto_skill.probes.author_style.sampling import bucket, stable_key


def manifest_row(
    selected_posts: dict[str, dict[str, list[dict[str, Any]]]],
    packs: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    pairwise_jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    style_ids: dict[str, dict[str, str]] | None = None,
    include_author_hash: bool = False,
) -> dict[str, Any]:
    style_rows = []
    for corpus, grouped in selected_posts.items():
        for author, posts in grouped.items():
            style_row = {
                "corpus": corpus,
                "style_id": (style_ids or {}).get(corpus, {}).get(author),
                "selected_word_counts": [post["word_count"] for post in posts],
                "selected_length_buckets": [bucket(int(post["word_count"])) for post in posts],
                "topic_count": len({post.get("private_topic") for post in posts}),
            }
            if include_author_hash:
                style_row["author_hash"] = author
            style_rows.append(style_row)
    return {
        "schema_version": "author-style-t2-probe-manifest/v1",
        "seed": args.seed,
        "styles": len(style_rows),
        "style_counts": dict(Counter(row["corpus"] for row in style_rows)),
        "oracle_packs": len(packs),
        "hard_negatives": len(negatives),
        "pairwise_jobs": len(pairwise_jobs),
        "crosscheck_packs": max(1, round(len(packs) * args.crosscheck_rate)),
        "selected_styles": style_rows,
    }


def write_crosscheck_subset(
    out_dir: Path,
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    seed: int,
    rate: float,
) -> None:
    pack_ids = [
        pack["pack_id"]
        for pack in sorted(packs, key=lambda p: stable_key(seed, "xcheck", p["pack_id"]))
    ]
    selected = set(pack_ids[: max(1, round(len(pack_ids) * rate))])
    write_jsonl(out_dir / "packs.jsonl", [row for row in packs if row["pack_id"] in selected])
    write_jsonl(
        out_dir / "private_eval.jsonl",
        [row for row in private_eval if row["pack_id"] in selected],
    )
    task_refs = {
        task["task_id"]
        for pack in packs
        if pack["pack_id"] in selected
        for task in pack.get("heldout_tasks", [])
    }
    write_jsonl(
        out_dir / "hard_negatives.jsonl",
        [row for row in negatives if row["target_task_ref"] in task_refs],
    )
