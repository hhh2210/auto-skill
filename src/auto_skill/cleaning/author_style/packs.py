"""Pack, split, private-eval, and hard-negative builders for author style."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

from auto_skill.cleaning.author_style.common import (
    SCHEMA_VERSION,
    CleanPost,
)
from auto_skill.cleaning.author_style.pack_negatives import (
    days_apart,
    find_hard_negatives,
    find_mendeley_native_impostor_negatives,
    jaccard,
    merge_negative_rows,
    source_native_impostor_rows,
    style_similarity,
)

__all__ = [
    "aggregate_style_features",
    "build_public_pack",
    "build_split_row",
    "choose_authors",
    "clean_post_row",
    "days_apart",
    "default_task_input",
    "derive_cluster_tags",
    "find_hard_negatives",
    "find_mendeley_native_impostor_negatives",
    "jaccard",
    "merge_negative_rows",
    "pack_prefix_for_source_choice",
    "post_public_row",
    "private_author_row",
    "private_task_row",
    "public_source_label",
    "source_domain",
    "source_inventory_row",
    "source_native_impostor_rows",
    "style_similarity",
]


def aggregate_style_features(posts: list[CleanPost]) -> dict[str, float]:
    keys = sorted({key for post in posts for key in post.style_features})
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [post.style_features[key] for post in posts if key in post.style_features]
        aggregated[f"mean_{key}"] = round(sum(values) / len(values), 3)
        aggregated[f"std_{key}"] = round(
            math.sqrt(
                sum((value - aggregated[f"mean_{key}"]) ** 2 for value in values) / len(values)
            ),
            3,
        )
    return aggregated


def derive_cluster_tags(features: dict[str, float]) -> tuple[str, ...]:
    tags: list[str] = []
    if features.get("mean_avg_sentence_words", 0) >= 24:
        tags.append("long_sentence")
    elif features.get("mean_avg_sentence_words", 0) <= 12:
        tags.append("short_sentence")
    if features.get("mean_first_person_per_100w", 0) >= 5:
        tags.append("first_person_heavy")
    if features.get("mean_question_per_100w", 0) >= 0.8:
        tags.append("question_heavy")
    if features.get("mean_exclamation_per_100w", 0) >= 0.8:
        tags.append("exclamation_heavy")
    if features.get("mean_ellipsis_per_100w", 0) >= 0.4:
        tags.append("ellipsis_heavy")
    if features.get("mean_paren_per_100w", 0) >= 0.6:
        tags.append("parenthetical")
    if features.get("mean_contraction_per_100w", 0) >= 1.2:
        tags.append("conversational_contractions")
    if not tags:
        tags.append("moderate_plain_prose")
    return tuple(tags)


def choose_authors(
    posts: list[CleanPost],
    *,
    authors: int,
    train_posts: int,
    heldout_posts: int,
) -> list[tuple[str, list[CleanPost], list[CleanPost]]]:
    grouped: dict[str, list[CleanPost]] = defaultdict(list)
    for post in posts:
        grouped[post.author_hash].append(post)
    required = train_posts + heldout_posts
    candidates = []
    for author_hash, author_posts in grouped.items():
        author_posts = sorted(author_posts, key=lambda item: (item.private_date, item.source_id))
        if len(author_posts) < required:
            continue
        feature_tags = derive_cluster_tags(aggregate_style_features(author_posts[:train_posts]))
        topic_count = len({post.private_topic for post in author_posts})
        candidates.append(
            (len(feature_tags), topic_count, len(author_posts), author_hash, author_posts)
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    selected = []
    for _, _, _, author_hash, author_posts in candidates[:authors]:
        selected.append(
            (
                author_hash,
                author_posts[:train_posts],
                author_posts[train_posts : train_posts + heldout_posts],
            )
        )
    return selected


def source_domain(source: str) -> dict[str, str]:
    if source == "BlogAuthorship":
        return {"kind": "personal_blog", "language": "en", "split_kind": "temporal"}
    if source == "LongLaMPTopicWritingTemporal":
        return {"kind": "personal_topic_writing", "language": "en", "split_kind": "temporal"}
    if source == "LongLaMPProductReviewTemporal":
        return {"kind": "product_review", "language": "en", "split_kind": "temporal"}
    if source == "MendeleyRedditCrossTopic":
        return {
            "kind": "cross_topic_online_comment",
            "language": "en",
            "split_kind": "known_unknown_cross_topic",
        }
    return {"kind": "personal_writing", "language": "en", "split_kind": "temporal"}


def pack_prefix_for_source_choice(source_choice: str) -> str:
    if source_choice == "mendeley-reddit":
        return "author_style_reddit_cross_topic"
    if source_choice == "longlamp-topic":
        return "author_style_topic_post"
    if source_choice == "longlamp-product":
        return "author_style_product_review"
    return "author_style_personal_blog"


def public_source_label(source: str) -> str:
    if source == "BlogAuthorship":
        return "personal_blog_history"
    if source == "LongLaMPTopicWritingTemporal":
        return "personal_topic_writing_history"
    if source == "LongLaMPProductReviewTemporal":
        return "personal_product_review_history"
    if source == "MendeleyRedditCrossTopic":
        return "cross_topic_online_comment_history"
    return "personal_author_history"


def default_task_input(post: CleanPost) -> str:
    if post.task_input:
        return post.task_input
    if post.source == "MendeleyRedditCrossTopic":
        return (
            "Continue the same anonymous online discussion user's authentic voice "
            "while writing a new comment bundle touching on: "
            + ", ".join(post.content_tags[:5])
        )
    if post.source == "LongLaMPProductReviewTemporal":
        return "Write a product review touching on: " + ", ".join(post.content_tags[:5])
    if post.source == "LongLaMPTopicWritingTemporal":
        return "Write a personal online post touching on: " + ", ".join(post.content_tags[:5])
    return "Write a personal blog-style post touching on: " + ", ".join(post.content_tags[:5])


def post_public_row(
    post: CleanPost,
    *,
    row_id_key: str,
    row_id: str,
    include_output: bool,
) -> dict[str, Any]:
    row = {
        "schema_version": SCHEMA_VERSION,
        row_id_key: row_id,
        "source": public_source_label(post.source),
        "source_task_id": post.source_id,
        "domain": source_domain(post.source),
        "task_input": default_task_input(post),
        "metadata": {
            "public_content_tags": list(post.content_tags[:5]),
            "word_count": post.word_count,
        },
    }
    if include_output:
        row["desired_output"] = {
            "status": "generated",
            "text": post.text,
            "source": "original_user_history",
            "provenance": "source_provided",
        }
    else:
        row["desired_output"] = None
    return row


def build_split_row(
    pack_id: str,
    train_posts: list[CleanPost],
    heldout_posts: list[CleanPost],
) -> dict[str, Any]:
    """Build a split row for inspection before converting to pack format."""

    return {
        "schema_version": "author-style-split/v1",
        "split_id": f"{pack_id}::temporal",
        "pack_id": pack_id,
        "source": train_posts[0].source,
        "learning_problem": "personal_author_style_induction",
        "domain": source_domain(train_posts[0].source),
        "train_source_task_ids": [post.source_id for post in train_posts],
        "heldout_source_task_ids": [post.source_id for post in heldout_posts],
        "private_boundary": {
            "private_dates_are_not_prompt_inputs": True,
            "private_topics_are_for_negative_matching_only": True,
        },
    }


def build_public_pack(
    pack_id: str,
    train_posts: list[CleanPost],
    heldout_posts: list[CleanPost],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": pack_id,
        "split_id": f"{pack_id}::temporal",
        "source": public_source_label(train_posts[0].source),
        "learning_problem": "personal_author_style_induction",
        "domain": source_domain(train_posts[0].source),
        "input_boundary": {
            "auto_skill_module_can_use": [
                "train_examples.task_input",
                "train_examples.desired_output.text",
            ],
            "must_not_use_for_induction": [
                "raw author ids",
                "author hashes",
                "private dates",
                "private demographic fields",
                "hard negative labels",
                "heldout outputs",
                "raw dataset inputs",
                "profile summaries",
                "source config names",
                "synthetic/private temporal indices",
            ],
        },
        "train_examples": [
            post_public_row(
                post,
                row_id_key="example_id",
                row_id=f"{pack_id}::train::{index}",
                include_output=True,
            )
            for index, post in enumerate(train_posts)
        ],
        "heldout_tasks": [
            post_public_row(
                post,
                row_id_key="task_id",
                row_id=f"{pack_id}::heldout::{index}",
                include_output=False,
            )
            for index, post in enumerate(heldout_posts)
        ],
    }


def private_task_row(post: CleanPost, *, task_ref: str, include_reference: bool) -> dict[str, Any]:
    row = {
        "task_ref": task_ref,
        "source": post.source,
        "source_task_id": post.source_id,
        "private_topic": post.private_topic,
        "private_date": post.private_date,
        "word_count": post.word_count,
        "content_tags": list(post.content_tags),
        "style_features": post.style_features,
        "public_task_input": default_task_input(post),
        "private_metadata": post.private_metadata,
        "supervision": {"type": "author_style_reference_likeness"},
        "judge": {"kind": "pairwise_style_likeness_with_reference"},
    }
    if include_reference:
        row["reference_output_private"] = post.text
    return row


def private_author_row(
    pack_id: str, author_hash: str, train: list[CleanPost], heldout: list[CleanPost]
) -> dict[str, Any]:
    posts = train + heldout
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": pack_id,
        "split_id": f"{pack_id}::temporal",
        "author_hash": author_hash,
        "source": train[0].source,
        "raw_author_id_private": train[0].raw_author_id,
        "date_is_synthetic": bool(train[0].private_metadata.get("date_is_synthetic")),
        "source_config": train[0].private_metadata.get("source_config"),
        "input_sanitizer_version": train[0].private_metadata.get("input_sanitizer_version"),
        "private_topics": sorted({post.private_topic for post in posts}),
        "private_date_range": [posts[0].private_date, posts[-1].private_date],
        "train_post_ids": [post.source_id for post in train],
        "heldout_post_ids": [post.source_id for post in heldout],
        "train_private": [
            private_task_row(post, task_ref=f"{pack_id}::train::{index}", include_reference=False)
            for index, post in enumerate(train)
        ],
        "heldout_private": [
            private_task_row(post, task_ref=f"{pack_id}::heldout::{index}", include_reference=True)
            for index, post in enumerate(heldout)
        ],
        "style_summary": aggregate_style_features(train),
        "cluster_tags": list(derive_cluster_tags(aggregate_style_features(train))),
    }


def clean_post_row(post: CleanPost) -> dict[str, Any]:
    return {
        "schema_version": "author-style-clean-post/v1",
        "source": post.source,
        "source_id": post.source_id,
        "author_hash": post.author_hash,
        "private_topic": post.private_topic,
        "private_date": post.private_date,
        "word_count": post.word_count,
        "content_tags": list(post.content_tags),
        "style_features": post.style_features,
        "text": post.text,
        "private_metadata": post.private_metadata,
    }


def source_inventory_row(args: argparse.Namespace, posts: list[CleanPost]) -> dict[str, Any]:
    by_author: dict[str, int] = defaultdict(int)
    by_topic: dict[str, int] = defaultdict(int)
    for post in posts:
        by_author[post.author_hash] += 1
        by_topic[post.private_topic] += 1
    return {
        "schema_version": "author-style-source-inventory/v1",
        "source": args.source,
        "hf_dataset": args.hf_dataset,
        "mendeley_path": str(args.mendeley_path) if getattr(args, "mendeley_path", None) else None,
        "source_label": posts[0].source if posts else None,
        "scanned_max_rows": args.max_rows,
        "clean_posts": len(posts),
        "authors_with_clean_posts": len(by_author),
        "topics_with_clean_posts": len(by_topic),
        "min_words": args.min_words,
        "max_words": args.max_words,
        "top_clean_post_counts_per_author": sorted(by_author.values(), reverse=True)[:20],
        "top_topics": [
            {"topic_private": topic, "clean_posts": count}
            for topic, count in sorted(by_topic.items(), key=lambda item: (-item[1], item[0]))[
                :20
            ]
        ],
    }
