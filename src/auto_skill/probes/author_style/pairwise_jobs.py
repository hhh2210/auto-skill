"""Pairwise job builders for author-style separation probes."""

from __future__ import annotations

from typing import Any

from auto_skill.metrics.author_style_reference import text_sha256
from auto_skill.probes.author_style.sampling import stable_key


def add_within_pairwise(
    jobs: list[dict[str, Any]],
    *,
    corpus: str,
    author: str,
    posts: list[dict[str, Any]],
) -> None:
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            jobs.append(pairwise_job("within_author", corpus, author, author, posts[i], posts[j]))


def add_cross_pairwise(
    jobs: list[dict[str, Any]],
    selected_posts: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    seed: int,
    cross_pairs_per_corpus: int,
) -> None:
    for corpus, grouped in selected_posts.items():
        add_cross_pairwise_for_corpus(
            jobs,
            corpus=corpus,
            grouped=grouped,
            seed=seed,
            target=cross_pairs_per_corpus,
        )


def add_cross_pairwise_for_corpus(
    jobs: list[dict[str, Any]],
    *,
    corpus: str,
    grouped: dict[str, list[dict[str, Any]]],
    seed: int,
    target: int,
) -> None:
    authors = sorted(grouped)
    candidates = []
    for left_index, left_author in enumerate(authors):
        for right_author in authors[left_index + 1 :]:
            for left in grouped[left_author]:
                for right in grouped[right_author]:
                    candidates.append((left_author, right_author, left, right))
    candidates.sort(
        key=lambda item: stable_key(
            seed,
            corpus,
            "cross",
            item[2]["source_id"],
            item[3]["source_id"],
        )
    )
    for left_author, right_author, left, right in candidates[:target]:
        jobs.append(pairwise_job("cross_author", corpus, left_author, right_author, left, right))


def pairwise_job(
    kind: str,
    corpus: str,
    left_author: str,
    right_author: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    key = stable_key(0, kind, corpus, left["source_id"], right["source_id"])[:12]
    return {
        "schema_version": "author-style-pairwise-probe-job/v1",
        "job_id": f"pairwise_{kind}_{corpus}_{key}",
        "pair_kind": kind,
        "corpus": corpus,
        "left_author_hash": left_author,
        "right_author_hash": right_author,
        "left_text": left["text"],
        "right_text": right["text"],
        "left_sha256": text_sha256(left["text"]),
        "right_sha256": text_sha256(right["text"]),
        "left_word_count": left["word_count"],
        "right_word_count": right["word_count"],
    }
