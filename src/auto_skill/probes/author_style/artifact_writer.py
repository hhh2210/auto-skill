"""Filesystem writer for author-style T2 probe artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from auto_skill.io.jsonl import load_jsonl, write_jsonl
from auto_skill.probes.author_style.artifacts import build_probe
from auto_skill.probes.author_style.outputs import write_crosscheck_subset


@dataclass(frozen=True)
class ProbeArtifactConfig:
    blog_clean_posts: Path
    reddit_clean_posts: Path
    out_dir: Path
    seed: int = 20260523
    blog_authors: int = 50
    reddit_authors: int = 50
    posts_per_author: int = 5
    blog_min_buckets: int = 3
    triples_per_author: int = 4
    triples_per_bucket: int = 80
    negatives_per_triple: int = 4
    cross_pairs_per_corpus: int = 500
    crosscheck_rate: float = 0.10


def write_probe_artifacts(config: ProbeArtifactConfig) -> dict[str, object]:
    packs, private_eval, negatives, pairwise_jobs, manifest, private_manifest = build_probe(
        blog_posts=load_jsonl(config.blog_clean_posts),
        reddit_posts=load_jsonl(config.reddit_clean_posts),
        args=config,
    )
    oracle_dir = config.out_dir / "oracle"
    write_jsonl(oracle_dir / "packs.jsonl", packs)
    write_jsonl(oracle_dir / "private_eval.jsonl", private_eval)
    write_jsonl(oracle_dir / "hard_negatives.jsonl", negatives)
    write_crosscheck_subset(
        config.out_dir / "oracle_crosscheck",
        packs,
        private_eval,
        negatives,
        seed=config.seed,
        rate=config.crosscheck_rate,
    )
    private_dir = config.out_dir / "_private"
    write_jsonl(private_dir / "pairwise_jobs.jsonl", pairwise_jobs)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / "sample_manifest_private.json").write_text(
        json.dumps(private_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def manifest_summary(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "styles",
            "style_counts",
            "oracle_packs",
            "pairwise_jobs",
            "crosscheck_packs",
        )
    }
