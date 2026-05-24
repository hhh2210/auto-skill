"""Hard-negative builders for author-style packs."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from auto_skill.cleaning.author_style.common import (
    NATIVE_IMPOSTOR_NEGATIVE_TYPE,
    CleanPost,
    stable_hash,
)


def jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def days_apart(left: str, right: str) -> int:
    return abs((datetime.fromisoformat(left) - datetime.fromisoformat(right)).days)


def style_similarity(left: CleanPost, right: CleanPost) -> float:
    """Return a rough 0-1 similarity over cheap surface-style features."""

    return style_similarity_features(left.style_features, right.style_features)


def style_similarity_features(left: dict[str, float], right: dict[str, float]) -> float:
    """Return a rough 0-1 similarity over cheap surface-style feature dicts."""

    normalizers = {
        "avg_sentence_words": 30.0,
        "question_per_100w": 3.0,
        "exclamation_per_100w": 3.0,
        "ellipsis_per_100w": 2.0,
        "paren_per_100w": 3.0,
        "first_person_per_100w": 8.0,
        "contraction_per_100w": 4.0,
        "hedge_marker": 1.0,
    }
    distances = []
    for key, normalizer in normalizers.items():
        if key not in left or key not in right:
            continue
        distances.append(abs(left[key] - right[key]) / normalizer)
    if not distances:
        return 0.0
    return round(max(0.0, 1.0 - min(1.0, sum(distances) / len(distances))), 3)


def word_count_length_ratio(left_word_count: int, right_word_count: int) -> float:
    """Return a 0-1 length match score for two word counts."""

    return min(left_word_count, right_word_count) / max(left_word_count, right_word_count, 1)


def find_hard_negatives(
    selected: list[tuple[str, list[CleanPost], list[CleanPost]]],
    all_posts: list[CleanPost],
    *,
    negatives_per_heldout: int,
    negative_author_cap: int = 2,
    pack_ids_by_author: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    pack_ids_by_author = pack_ids_by_author or {}
    rows: list[dict[str, Any]] = []
    for author_hash, _, heldout_posts in selected:
        pack_id = pack_ids_by_author.get(author_hash, stable_hash(author_hash, prefix="pack"))
        pool = [post for post in all_posts if post.author_hash != author_hash]
        for heldout_index, heldout in enumerate(heldout_posts):
            scored = []
            for candidate in pool:
                topic_match = int(candidate.private_topic == heldout.private_topic)
                length_ratio = word_count_length_ratio(candidate.word_count, heldout.word_count)
                time_score = max(
                    0.0, 1.0 - days_apart(candidate.private_date, heldout.private_date) / 365
                )
                lexical = jaccard(candidate.content_tags, heldout.content_tags)
                style_score = style_similarity(candidate, heldout)
                score = topic_match * 2.0 + length_ratio + time_score + lexical + style_score * 2.0
                scored.append(
                    (score, topic_match, length_ratio, time_score, lexical, style_score, candidate)
                )
            scored.sort(key=lambda item: (-item[0], item[6].source_id))
            author_counts: dict[str, int] = defaultdict(int)
            selected_scored = []
            for item in scored:
                candidate = item[6]
                if author_counts[candidate.author_hash] >= negative_author_cap:
                    continue
                author_counts[candidate.author_hash] += 1
                selected_scored.append(item)
                if len(selected_scored) >= negatives_per_heldout:
                    break
            for rank, (
                score,
                topic_match,
                length_ratio,
                time_score,
                lexical,
                style_score,
                candidate,
            ) in enumerate(selected_scored, start=1):
                rows.append(
                    {
                        "schema_version": "author-style-hard-negative/v1",
                        "negative_id": f"{pack_id}::heldout::{heldout_index}::negative::{rank}",
                        "target_author_hash": author_hash,
                        "negative_author_hash": candidate.author_hash,
                        "target_task_ref": f"{pack_id}::heldout::{heldout_index}",
                        "negative_type": "topic_time_length_style_matched_impostor",
                        "public_negative_text": candidate.text,
                        "public_negative_content_tags": list(candidate.content_tags[:5]),
                        "match_features": {
                            "same_topic": bool(topic_match),
                            "target_topic_private": heldout.private_topic,
                            "negative_topic_private": candidate.private_topic,
                            "target_date_private": heldout.private_date,
                            "negative_date_private": candidate.private_date,
                            "length_ratio": round(length_ratio, 3),
                            "time_score": round(time_score, 3),
                            "content_tag_jaccard": round(lexical, 3),
                            "style_similarity": style_score,
                            "score": round(score, 3),
                        },
                        "private_label": "different_author",
                        "must_not_use_for_induction": True,
                    }
                )
    return rows


def find_mendeley_native_impostor_negatives(
    selected: list[tuple[str, list[CleanPost], list[CleanPost]]],
    all_posts: list[CleanPost],
    *,
    pack_ids_by_author: dict[str, str],
) -> list[dict[str, Any]]:
    """Use original PAN ``N`` unknowns as native impostors for the claimed author."""

    selected_by_author = {author_hash: heldout for author_hash, _, heldout in selected}
    problem_author_hash: dict[str, str] = {}
    for post in all_posts:
        metadata = post.private_metadata
        if post.source != "MendeleyRedditCrossTopic":
            continue
        if metadata.get("doc_role_private") != "known":
            continue
        problem_id = str(metadata.get("mendeley_problem_id_private") or "")
        if problem_id:
            problem_author_hash[problem_id] = post.author_hash

    rows: list[dict[str, Any]] = []
    for candidate in all_posts:
        metadata = candidate.private_metadata
        if candidate.source != "MendeleyRedditCrossTopic":
            continue
        if metadata.get("doc_role_private") != "unknown":
            continue
        if metadata.get("truth_label_private") != "N":
            continue
        problem_id = str(metadata.get("mendeley_problem_id_private") or "")
        target_author_hash = problem_author_hash.get(problem_id)
        if not target_author_hash or target_author_hash == candidate.author_hash:
            continue
        heldout_posts = selected_by_author.get(target_author_hash)
        if not heldout_posts:
            continue
        pack_id = pack_ids_by_author[target_author_hash]
        for heldout_index, heldout in enumerate(heldout_posts):
            topic_match = int(candidate.private_topic == heldout.private_topic)
            length_ratio = word_count_length_ratio(candidate.word_count, heldout.word_count)
            time_score = max(
                0.0, 1.0 - days_apart(candidate.private_date, heldout.private_date) / 365
            )
            lexical = jaccard(candidate.content_tags, heldout.content_tags)
            style_score = style_similarity(candidate, heldout)
            score = topic_match * 2.0 + length_ratio + time_score + lexical + style_score * 2.0
            rows.append(
                {
                    "schema_version": "author-style-hard-negative/v1",
                    "negative_id": (
                        f"{pack_id}::heldout::{heldout_index}::negative::original_av_impostor"
                    ),
                    "target_author_hash": target_author_hash,
                    "negative_author_hash": candidate.author_hash,
                    "target_task_ref": f"{pack_id}::heldout::{heldout_index}",
                    "negative_type": NATIVE_IMPOSTOR_NEGATIVE_TYPE,
                    "public_negative_text": candidate.text,
                    "public_negative_content_tags": list(candidate.content_tags[:5]),
                    "match_features": {
                        "same_topic": bool(topic_match),
                        "target_topic_private": heldout.private_topic,
                        "negative_topic_private": candidate.private_topic,
                        "target_date_private": heldout.private_date,
                        "negative_date_private": candidate.private_date,
                        "length_ratio": round(length_ratio, 3),
                        "time_score": round(time_score, 3),
                        "content_tag_jaccard": round(lexical, 3),
                        "style_similarity": style_score,
                        "score": round(score, 3),
                        "source_problem_private": problem_id,
                    },
                    "private_label": "different_author",
                    "must_not_use_for_induction": True,
                }
            )
    return rows


def merge_negative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    key_order: list[tuple[str, str, str]] = []
    for row in rows:
        key = (
            str(row.get("target_task_ref") or ""),
            str(row.get("negative_author_hash") or ""),
            stable_hash(str(row.get("public_negative_text") or ""), prefix="text"),
        )
        existing = merged_by_key.get(key)
        if existing is None:
            key_order.append(key)
            merged_by_key[key] = row
            continue
        row_status = row.get("gpt_rerank", {}).get("status")
        existing_status = existing.get("gpt_rerank", {}).get("status")
        if row_status == "source_native_impostor" and existing_status != row_status:
            merged_by_key[key] = row
    return [merged_by_key[key] for key in key_order]


def source_native_impostor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    native_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("negative_type") != NATIVE_IMPOSTOR_NEGATIVE_TYPE:
            continue
        native = dict(row)
        native["gpt_rerank"] = {
            **native.get("gpt_rerank", {}),
            "status": "source_native_impostor",
            "reason": "original PAN truth=N unknown for the claimed problem author",
        }
        native_rows.append(native)
    return native_rows
