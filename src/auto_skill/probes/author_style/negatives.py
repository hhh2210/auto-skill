"""Hard-negative selection for author-style probe artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auto_skill.cleaning.author_style.pack_negatives import (
    days_apart,
    jaccard,
    style_similarity_features,
    word_count_length_ratio,
)
from auto_skill.probes.author_style.sampling import bucket, stable_key, stable_order


@dataclass(frozen=True)
class NegativePool:
    posts_by_author: dict[str, list[dict[str, Any]]]
    all_reps: list[dict[str, Any]]
    topic_reps: dict[str, list[dict[str, Any]]]
    length_reps: dict[str, list[dict[str, Any]]]


def build_negative_pool(posts: list[dict[str, Any]], *, seed: int) -> NegativePool:
    return NegativePool(
        posts_by_author=group_by_author(posts),
        all_reps=one_post_per_author(posts, seed=seed, salt="all"),
        topic_reps=group_representatives(posts, seed=seed, salt="topic", key=topic_key),
        length_reps=group_representatives(posts, seed=seed, salt="length", key=length_key),
    )


def select_negatives(
    target: dict[str, Any],
    pool: NegativePool | list[dict[str, Any]],
    *,
    variant: str,
    count: int,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    if variant == "topic":
        candidates = fill_preferred(*candidate_pair(pool, target, "topic", seed=seed), count)
    elif variant in {"length", "length_bucket"}:
        candidates = fill_preferred(*candidate_pair(pool, target, "length", seed=seed), count)
    elif variant == "style_surface":
        candidates = style_surface_candidates(pool, target, seed=seed, salt=f"{salt}:style")
        candidates = sorted(
            candidates,
            key=lambda post: (-surface_score(target, post), post["source_id"]),
        )
    else:
        candidates = candidate_reps(pool, seed=seed, salt=f"{salt}:random")
        candidates = exclude_author(candidates, target)
        candidates = stable_order(candidates, seed, salt)
    return first_distinct_author_posts(candidates, count=count, salt=salt)


def candidate_pair(
    pool: NegativePool | list[dict[str, Any]],
    target: dict[str, Any],
    kind: str,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(pool, NegativePool):
        preferred = (
            pool.topic_reps.get(topic_key(target), [])
            if kind == "topic"
            else pool.length_reps.get(length_key(target), [])
        )
        return exclude_author(preferred, target), exclude_author(pool.all_reps, target)
    candidates = exclude_author(pool, target)
    if kind == "topic":
        preferred = same_topic_posts(target, candidates)
    else:
        preferred = same_length_bucket_posts(target, candidates)
    return (
        one_post_per_author(preferred, seed=seed, salt=kind),
        one_post_per_author(candidates, seed=seed, salt="fallback"),
    )


def candidate_reps(
    pool: NegativePool | list[dict[str, Any]],
    *,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    if isinstance(pool, NegativePool):
        return pool.all_reps
    return one_post_per_author(pool, seed=seed, salt=salt)


def style_surface_candidates(
    pool: NegativePool | list[dict[str, Any]],
    target: dict[str, Any],
    *,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    if not isinstance(pool, NegativePool):
        return exclude_author(one_post_per_author(pool, seed=seed, salt=salt), target)
    candidates = []
    target_author = target["author_hash"]
    for author, posts in pool.posts_by_author.items():
        if author == target_author:
            continue
        candidates.append(max(posts, key=lambda post: surface_score(target, post)))
    return candidates


def exclude_author(
    candidates: list[dict[str, Any]],
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    return [post for post in candidates if post["author_hash"] != target["author_hash"]]


def same_topic_posts(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [post for post in candidates if topic_key(post) == topic_key(target)]


def same_length_bucket_posts(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_bucket = length_key(target)
    return [post for post in candidates if length_key(post) == target_bucket]


def fill_preferred(
    preferred: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    authors = {post["author_hash"] for post in preferred}
    if len(authors) >= count:
        return preferred
    return preferred + [post for post in fallback if post["author_hash"] not in authors]


def first_distinct_author_posts(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    salt: str,
) -> list[dict[str, Any]]:
    selected = []
    used_authors = set()
    for post in candidates:
        if post["author_hash"] in used_authors:
            continue
        selected.append(post)
        used_authors.add(post["author_hash"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(f"negative pool too small for {salt}: {len(selected)} < {count}")
    return selected


def one_post_per_author(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    by_author: dict[str, dict[str, Any]] = {}
    by_key: dict[str, str] = {}
    for post in candidates:
        author = post["author_hash"]
        key = stable_order_key(post, seed=seed, salt=salt)
        if author not in by_author or key < by_key[author]:
            by_author[author] = post
            by_key[author] = key
    return list(by_author.values())


def group_by_author(posts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for post in posts:
        grouped.setdefault(post["author_hash"], []).append(post)
    return grouped


def group_representatives(
    posts: list[dict[str, Any]],
    *,
    seed: int,
    salt: str,
    key: Any,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for post in posts:
        grouped.setdefault(key(post), []).append(post)
    return {
        group_key: one_post_per_author(rows, seed=seed, salt=f"{salt}:{group_key}")
        for group_key, rows in grouped.items()
    }


def topic_key(post: dict[str, Any]) -> str:
    return str(post.get("private_topic") or "")


def length_key(post: dict[str, Any]) -> str:
    return bucket(int(post["word_count"]))


def stable_order_key(post: dict[str, Any], *, seed: int, salt: str) -> str:
    return f"{stable_key(seed, salt, post['source_id'])}:{post['source_id']}"


def surface_score(target: dict[str, Any], candidate: dict[str, Any]) -> float:
    same_topic = 2.0 if target.get("private_topic") == candidate.get("private_topic") else 0.0
    length = word_count_length_ratio(int(target["word_count"]), int(candidate["word_count"]))
    lexical = tag_jaccard(target, candidate)
    style = style_similarity_features(
        target.get("style_features", {}),
        candidate.get("style_features", {}),
    )
    time_score = max(
        0.0,
        1.0 - days_apart(str(target.get("private_date")), str(candidate.get("private_date"))) / 365,
    )
    return same_topic + length + lexical + time_score + style * 2.0


def match_features(target: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "same_topic": target.get("private_topic") == candidate.get("private_topic"),
        "length_ratio": round(
            word_count_length_ratio(int(target["word_count"]), int(candidate["word_count"])),
            3,
        ),
        "content_tag_jaccard": round(tag_jaccard(target, candidate), 3),
        "style_surface_score": round(surface_score(target, candidate), 3),
        "target_word_count": target.get("word_count"),
        "negative_word_count": candidate.get("word_count"),
    }


def tag_jaccard(target: dict[str, Any], candidate: dict[str, Any]) -> float:
    return jaccard(tuple(target.get("content_tags", [])), tuple(candidate.get("content_tags", [])))
