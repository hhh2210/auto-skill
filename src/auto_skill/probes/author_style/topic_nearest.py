"""Target-nearest semantic hard negatives for author-style probes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_skill.io.jsonl import load_jsonl, write_jsonl
from auto_skill.probes.author_style.embedding_cache import embed_with_cache
from auto_skill.probes.author_style.negatives import match_features

SOURCE_TO_CORPUS = {
    "BlogAuthorship": "blog",
    "MendeleyRedditCrossTopic": "reddit",
}


@dataclass(frozen=True)
class TopicNearestConfig:
    source_probe_dir: Path
    blog_clean_posts: Path
    reddit_clean_posts: Path
    out_dir: Path
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_backend: str = "sentence-transformers"
    embedding_base_url: str | None = None
    embedding_api_key: str = "EMPTY"
    embedding_expected_dim: int = 1024
    embedding_max_retries: int = 3
    embedding_workers: int = 1
    device: str = "mps"
    batch_size: int = 8
    max_seq_length: int = 1024
    max_candidate_chars: int = 3000
    negatives_per_target: int = 4
    cache_root: Path = Path("runs/cache/embeddings")
    limit_targets: int | None = None
    limit_corpus_rows: int | None = None
    max_scan: int = 200


def write_topic_nearest_artifacts(config: TopicNearestConfig) -> dict[str, Any]:
    packs = load_jsonl(config.source_probe_dir / "oracle" / "packs.jsonl")
    private_eval = load_jsonl(config.source_probe_dir / "oracle" / "private_eval.jsonl")
    source_pools = {"blog": load_jsonl(config.blog_clean_posts), "reddit": load_jsonl(config.reddit_clean_posts)}
    target_posts = resolve_target_posts(private_eval, source_pools)
    if config.limit_targets is not None:
        target_posts = dict(list(target_posts.items())[: config.limit_targets])
        packs, private_eval = filter_artifacts(packs, private_eval, set(target_posts))
    pools = {corpus: load_pool(posts, limit=config.limit_corpus_rows) for corpus, posts in source_pools.items()}
    target_embeddings, target_summary = embed_posts(list(target_posts.values()), config)
    pool_embeddings = {}
    pool_embedding_summaries = {}
    for corpus, posts in pools.items():
        pool_embeddings[corpus], pool_embedding_summaries[corpus] = embed_posts(posts, config)
    embedding_summary = {**target_summary, "target_cache": target_summary, "pool_cache": pool_embedding_summaries}
    nearest = select_nearest_for_targets(
        target_posts=target_posts,
        target_embeddings=target_embeddings,
        pools=pools,
        pool_embeddings=pool_embeddings,
        config=config,
    )
    hard_negatives = build_negative_rows(private_eval, target_posts, nearest)
    write_oracle_dir(config.out_dir / "oracle", [rewrite_pack_metadata(pack) for pack in packs], private_eval, hard_negatives)
    manifest = build_manifest(config, packs, pools, hard_negatives, embedding_summary)
    summary = build_summary(config, manifest, packs, pools, hard_negatives, embedding_summary)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_pool(rows: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    return rows[:limit] if limit is not None else rows


def resolve_target_posts(
    private_eval: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    by_source_id = {corpus: {str(post["source_id"]): post for post in posts} for corpus, posts in pools.items()}
    targets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in private_eval:
        corpus = SOURCE_TO_CORPUS[str(row["source"])]
        for task in row.get("heldout_private", []):
            source_id = str(task["source_task_id"])
            try:
                targets[(str(row["pack_id"]), str(task["task_ref"]))] = by_source_id[corpus][
                    source_id
                ]
            except KeyError as exc:
                raise KeyError(f"target source_id missing from {corpus} pool: {source_id}") from exc
    return targets


def filter_artifacts(
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    keep: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keep_packs = {pack_id for pack_id, _ in keep}
    filtered_private_eval = []
    for row in private_eval:
        pack_id = str(row.get("pack_id"))
        if pack_id not in keep_packs:
            continue
        heldout = [
            task
            for task in row.get("heldout_private", [])
            if (pack_id, str(task.get("task_ref"))) in keep
        ]
        if heldout:
            filtered_private_eval.append({**row, "heldout_private": heldout})
    return [pack for pack in packs if pack.get("pack_id") in keep_packs], filtered_private_eval


def embed_posts(
    posts: list[dict[str, Any]],
    config: TopicNearestConfig,
) -> tuple[Any, dict[str, Any]]:
    return embed_with_cache(
        [post["text"] for post in posts],
        cache_root=config.cache_root,
        model_id=config.model,
        backend=config.embedding_backend,
        device=config.device,
        batch_size=config.batch_size,
        max_seq_length=config.max_seq_length,
        max_chars=config.max_candidate_chars,
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        expected_dim=config.embedding_expected_dim,
        max_retries=config.embedding_max_retries,
        workers=config.embedding_workers,
    )


def select_nearest_for_targets(
    *,
    target_posts: dict[tuple[str, str], dict[str, Any]],
    target_embeddings: Any,
    pools: dict[str, list[dict[str, Any]]],
    pool_embeddings: dict[str, Any],
    config: TopicNearestConfig,
) -> dict[tuple[str, str], list[tuple[dict[str, Any], float]]]:
    output = {}
    for index, (target_key, target) in enumerate(target_posts.items()):
        corpus = "blog" if target["source"] == "BlogAuthorship" else "reddit"
        selected = select_distinct_author_nearest(
            target=target,
            target_embedding=target_embeddings[index],
            candidates=pools[corpus],
            candidate_embeddings=pool_embeddings[corpus],
            count=config.negatives_per_target,
            max_scan=config.max_scan,
        )
        output[target_key] = selected
    return output


def select_distinct_author_nearest(
    *,
    target: dict[str, Any],
    target_embedding: Any,
    candidates: list[dict[str, Any]],
    candidate_embeddings: Any,
    count: int,
    max_scan: int,
) -> list[tuple[dict[str, Any], float]]:
    import numpy as np

    scores = candidate_embeddings @ target_embedding
    take = min(max_scan, len(scores))
    top_indexes = np.argpartition(scores, -take)[-take:]
    ranked = sorted(top_indexes, key=lambda index: (-float(scores[int(index)]), int(index)))
    selected = []
    used_authors = set()
    for index in ranked:
        post = candidates[int(index)]
        if post["author_hash"] == target["author_hash"] or post["source_id"] == target["source_id"]:
            continue
        if post["author_hash"] in used_authors:
            continue
        selected.append((post, float(scores[int(index)])))
        used_authors.add(post["author_hash"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"topic-nearest found {len(selected)} distinct authors in top {max_scan}; "
            f"need {count}"
        )
    return selected


def rewrite_pack_metadata(pack: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(pack.get("probe_metadata") or {})
    metadata["source_hard_neg_variant"] = metadata.get("hard_neg_variant")
    metadata["hard_neg_variant"] = "topic_nearest_qwen3"
    return {**pack, "probe_metadata": metadata}


def build_negative_rows(
    private_eval: list[dict[str, Any]],
    targets: dict[tuple[str, str], dict[str, Any]],
    nearest: dict[tuple[str, str], list[tuple[dict[str, Any], float]]],
) -> list[dict[str, Any]]:
    rows = []
    for row in private_eval:
        pack_id = str(row["pack_id"])
        for task in row.get("heldout_private", []):
            task_id = str(task["task_ref"])
            target = targets[(pack_id, task_id)]
            for rank, (negative, score) in enumerate(nearest[(pack_id, task_id)], start=1):
                rows.append(negative_row(pack_id, task_id, rank, target, negative, score))
    return rows


def negative_row(
    pack_id: str,
    task_id: str,
    rank: int,
    target: dict[str, Any],
    negative: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    features = match_features(target, negative)
    features["embedding_cosine"] = round(score, 6)
    return {
        "schema_version": "author-style-hard-negative/v1",
        "negative_id": f"{pack_id}::negative::{rank}",
        "target_task_ref": task_id,
        "negative_type": "topic_nearest_qwen3_other_author",
        "public_negative_text": negative["text"],
        "match_features": features,
        "private_label": "different_author",
        "must_not_use_for_induction": True,
    }


def write_oracle_dir(
    oracle_dir: Path,
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    hard_negatives: list[dict[str, Any]],
) -> None:
    write_jsonl(oracle_dir / "packs.jsonl", packs)
    write_jsonl(oracle_dir / "private_eval.jsonl", private_eval)
    write_jsonl(oracle_dir / "hard_negatives.jsonl", hard_negatives)


def build_manifest(
    config: TopicNearestConfig,
    packs: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    hard_negatives: list[dict[str, Any]],
    embedding_summary: dict[str, Any],
) -> dict[str, Any]:
    base_manifest = config.source_probe_dir / "sample_manifest.json"
    return {
        "schema_version": "author-style-topic-nearest-manifest/v1",
        "base_sample_manifest": str(base_manifest),
        "base_sample_manifest_sha256": hashlib.sha256(base_manifest.read_bytes()).hexdigest() if base_manifest.exists() else None,
        "resampled_styles": False,
        "source_probe_dir": str(config.source_probe_dir),
        "pool_source_paths": {
            "blog": str(config.blog_clean_posts),
            "reddit": str(config.reddit_clean_posts),
        },
        "pool_sizes": {corpus: len(posts) for corpus, posts in pools.items()},
        "model": config.model,
        "embedding_backend": config.embedding_backend,
        "embedding_base_url": config.embedding_base_url,
        "embedding_expected_dim": config.embedding_expected_dim,
        "embedding_max_retries": config.embedding_max_retries,
        "embedding_workers": config.embedding_workers,
        "device": config.device,
        "batch_size": config.batch_size,
        "max_seq_length": config.max_seq_length,
        "max_candidate_chars": config.max_candidate_chars,
        "max_scan": config.max_scan,
        "embedding": embedding_summary,
        "oracle_packs": len(packs),
        "hard_negatives": len(hard_negatives),
        "negatives_per_target": config.negatives_per_target,
        "limit_targets": config.limit_targets,
        "limit_corpus_rows": config.limit_corpus_rows,
    }


def build_summary(
    config: TopicNearestConfig,
    manifest: dict[str, Any],
    packs: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    hard_negatives: list[dict[str, Any]],
    embedding_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "author-style-topic-nearest-summary/v1",
        "sample_manifest": str(config.out_dir / "sample_manifest.json"),
        "base_sample_manifest": manifest.get("base_sample_manifest"),
        "base_sample_manifest_sha256": manifest.get("base_sample_manifest_sha256"),
        "resampled_styles": False,
        "source_probe_dir": str(config.source_probe_dir),
        "pool_source_paths": manifest.get("pool_source_paths"),
        "pool_sizes": {corpus: len(posts) for corpus, posts in pools.items()},
        "oracle_packs": len(packs),
        "hard_negatives": len(hard_negatives),
        "negatives_per_target": config.negatives_per_target,
        "embedding_backend": config.embedding_backend,
        "embedding_model": config.model,
        "embedding_expected_dim": config.embedding_expected_dim,
        "embedding_dim": embedding_summary.get("embedding_dim"),
        "embedding_workers": config.embedding_workers,
        "embedding_batch_size": config.batch_size,
        "max_candidate_chars": config.max_candidate_chars,
        "max_scan": config.max_scan,
        "cache_namespace_policy": embedding_summary.get("cache_namespace_policy"),
        "embedding_cache_slug": embedding_summary.get("model_slug"),
        "package_versions": embedding_summary.get("package_versions"),
    }
