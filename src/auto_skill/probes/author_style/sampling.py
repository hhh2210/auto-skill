"""Sampling helpers for author-style probe construction."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

BUCKET_EDGES = (50, 100, 200, 500)
VARIANTS = ("random", "topic", "length", "style_surface")


def bucket(words: int, edges: tuple[int, ...] = BUCKET_EDGES) -> str:
    start = 0
    for edge in edges:
        if words < edge:
            return f"{start}_{edge}"
        start = edge
    return f"{edges[-1]}_plus"


def bucket_names(edges: tuple[int, ...] = BUCKET_EDGES) -> list[str]:
    return [bucket(edge - 1, edges) for edge in edges] + [f"{edges[-1]}_plus"]


def stable_key(seed: int, *parts: Any) -> str:
    return hashlib.sha256("::".join(map(str, (seed, *parts))).encode()).hexdigest()


def stable_order(rows: list[dict[str, Any]], seed: int, *parts: Any) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (stable_key(seed, *parts, row["source_id"]), row["source_id"]),
    )


def by_author(posts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in posts:
        grouped[str(post["author_hash"])].append(post)
    return grouped


def select_authors(
    posts: list[dict[str, Any]],
    *,
    corpus: str,
    count: int,
    posts_per_author: int,
    seed: int,
    blog_min_buckets: int,
) -> dict[str, list[dict[str, Any]]]:
    eligible = eligible_authors(
        posts,
        corpus=corpus,
        posts_per_author=posts_per_author,
        blog_min_buckets=blog_min_buckets,
    )
    if corpus != "blog":
        ordered = sorted(eligible, key=lambda item: stable_key(seed, corpus, item[0]))
        return dict(ordered[:count])
    selected = select_blog_by_post_count_strata(eligible, count=count, seed=seed)
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} eligible {corpus} authors for requested {count}")
    return selected


def eligible_authors(
    posts: list[dict[str, Any]],
    *,
    corpus: str,
    posts_per_author: int,
    blog_min_buckets: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    eligible = []
    for author, author_posts in by_author(posts).items():
        if len(author_posts) < posts_per_author:
            continue
        buckets = {bucket(int(post["word_count"])) for post in author_posts}
        if corpus == "blog" and len(buckets) < blog_min_buckets:
            continue
        eligible.append(
            (
                author,
                sorted(author_posts, key=lambda p: (p["private_date"], p["source_id"])),
            )
        )
    return eligible


def select_blog_by_post_count_strata(
    eligible: list[tuple[str, list[dict[str, Any]]]],
    *,
    count: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    quotas = [(5, 10, 0.20), (11, 30, 0.32), (31, 100, 0.28), (101, math.inf, 0.20)]
    selected: dict[str, list[dict[str, Any]]] = {}
    for low, high, fraction in quotas:
        need = round(count * fraction)
        pool = [
            item for item in eligible if low <= len(item[1]) <= high and item[0] not in selected
        ]
        pool.sort(key=lambda item: stable_key(seed, "blog", low, high, item[0]))
        selected.update(dict(pool[:need]))
    if len(selected) < count:
        fill = [item for item in eligible if item[0] not in selected]
        fill.sort(key=lambda item: stable_key(seed, "blog", "fill", item[0]))
        selected.update(dict(fill[: count - len(selected)]))
    return selected


def choose_posts(
    author: str,
    posts: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen = set()
    for name in bucket_names():
        in_bucket = [post for post in posts if bucket(int(post["word_count"])) == name]
        for post in stable_order(in_bucket, seed, author, name)[:1]:
            picked.append(post)
            seen.add(post["source_id"])
            break
        if len(picked) >= n:
            return picked
    for post in stable_order(posts, seed, author, "fill"):
        if post["source_id"] in seen:
            continue
        picked.append(post)
        if len(picked) >= n:
            return picked
    return picked
