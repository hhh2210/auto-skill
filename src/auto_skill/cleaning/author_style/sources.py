"""Source-specific loaders for public author-style corpora."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from typing import Any

from auto_skill.cleaning.author_style.common import (
    CleanPost,
    clean_blog_row,
    content_tags,
    normalize_text,
    reject_reasons,
    stable_hash,
    style_features,
    word_tokens,
)
from auto_skill.cleaning.author_style.source_mendeley import (
    MENDELEY_FILENAME_RE,
    clean_mendeley_reddit_doc,
    iter_mendeley_text_entries,
    load_mendeley_reddit_posts,
    mendeley_task_input,
    parse_mendeley_filename,
    parse_mendeley_truth,
)

__all__ = [
    "MENDELEY_FILENAME_RE",
    "add_unique_post",
    "clean_longlamp_product_output",
    "clean_longlamp_product_profile_item",
    "clean_longlamp_topic_output",
    "clean_longlamp_topic_profile_item",
    "clean_mendeley_reddit_doc",
    "clipped_task_input",
    "iter_mendeley_text_entries",
    "load_blog_posts",
    "load_hf_dataset_stream",
    "load_longlamp_product_posts",
    "load_longlamp_topic_posts",
    "load_mendeley_reddit_posts",
    "load_posts",
    "mendeley_task_input",
    "parse_mendeley_filename",
    "parse_mendeley_truth",
    "profile_items",
    "sanitize_product_task_input",
    "sanitize_topic_task_input",
    "synthetic_date",
]


def synthetic_date(index: int) -> str:
    return (date(2000, 1, 1) + timedelta(days=index)).isoformat()


def clipped_task_input(text: str, *, max_chars: int = 420) -> str:
    text = normalize_text(text)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


def sanitize_topic_task_input(value: str) -> str:
    text = clipped_task_input(value)
    text = re.sub(
        r"^generate the content for a reddit post",
        "Write a personal online post",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\breddit\b", "online forum", text, flags=re.IGNORECASE)
    text = re.sub(r"\bLongLaMP\b", "source", text, flags=re.IGNORECASE)
    return text or "Write a personal online post."


def sanitize_product_task_input(value: str) -> str:
    text = clipped_task_input(value)
    text = re.sub(
        r"^generate the review text written by a reviewer",
        "Write a product review",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\breviewerId\b|\bLongLaMP\b|\bprofile\b", "source", text, flags=re.IGNORECASE)
    return text or "Write a product review."


def profile_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def clean_longlamp_topic_output(
    row: dict[str, Any],
    *,
    row_index: int,
    min_words: int,
    max_words: int,
) -> CleanPost | None:
    raw_author_id = str(row.get("author") or "").strip()
    if not raw_author_id:
        return None
    text = normalize_text(str(row.get("output") or ""))
    tokens = word_tokens(text)
    if reject_reasons(text, tokens, min_words=min_words, max_words=max_words):
        return None
    task_input = sanitize_topic_task_input(str(row.get("input") or "Write a personal post."))
    author_hash = stable_hash(f"longlamp-topic:{raw_author_id}", prefix="author")
    post_hash = stable_hash(
        f"longlamp-topic:output:{raw_author_id}:{row_index}:{text[:160]}",
        prefix="post",
    )
    return CleanPost(
        source="LongLaMPTopicWritingTemporal",
        source_id=post_hash,
        author_hash=author_hash,
        raw_author_id=raw_author_id,
        private_topic="topic_writing_temporal",
        private_date=synthetic_date(row_index),
        text=text,
        word_count=len(tokens),
        content_tags=content_tags(tokens),
        style_features=style_features(text, tokens),
        task_input=task_input,
        private_metadata={
            "source_config": "topic_writing_temporal",
            "date_is_synthetic": True,
            "temporal_index_private": row_index,
            "source_input_private": str(row.get("input") or ""),
            "input_sanitizer_version": "longlamp-v1",
        },
    )


def clean_longlamp_topic_profile_item(
    item: dict[str, Any],
    *,
    row_index: int,
    item_index: int,
    min_words: int,
    max_words: int,
) -> CleanPost | None:
    raw_author_id = str(item.get("author") or "").strip()
    text = normalize_text(str(item.get("content") or ""))
    if not raw_author_id or not text:
        return None
    tokens = word_tokens(text)
    if reject_reasons(text, tokens, min_words=min_words, max_words=max_words):
        return None
    author_hash = stable_hash(f"longlamp-topic:{raw_author_id}", prefix="author")
    source_id_seed = str(item.get("id") or f"{row_index}:{item_index}:{text[:160]}")
    post_hash = stable_hash(
        f"longlamp-topic:profile:{raw_author_id}:{source_id_seed}", prefix="post"
    )
    return CleanPost(
        source="LongLaMPTopicWritingTemporal",
        source_id=post_hash,
        author_hash=author_hash,
        raw_author_id=raw_author_id,
        private_topic="topic_writing_temporal_profile",
        private_date=synthetic_date(max(0, row_index - 10_000 + item_index)),
        text=text,
        word_count=len(tokens),
        content_tags=content_tags(tokens),
        style_features=style_features(text, tokens),
        task_input="Write a personal online post touching on: "
        + ", ".join(content_tags(tokens)[:5]),
        private_metadata={
            "source_config": "topic_writing_temporal",
            "date_is_synthetic": True,
            "temporal_index_private": row_index - 10_000 + item_index,
            "source_profile_id_private": str(item.get("id") or ""),
            "source_profile_summary_private": str(item.get("summary") or ""),
            "input_sanitizer_version": "longlamp-v1",
        },
    )


def clean_longlamp_product_output(
    row: dict[str, Any],
    *,
    row_index: int,
    min_words: int,
    max_words: int,
) -> CleanPost | None:
    raw_author_id = str(row.get("reviewerId") or "").strip()
    if not raw_author_id:
        return None
    text = normalize_text(str(row.get("output") or ""))
    tokens = word_tokens(text)
    if reject_reasons(text, tokens, min_words=min_words, max_words=max_words):
        return None
    task_input = sanitize_product_task_input(str(row.get("input") or "Write a product review."))
    author_hash = stable_hash(f"longlamp-product:{raw_author_id}", prefix="author")
    post_hash = stable_hash(
        f"longlamp-product:output:{raw_author_id}:{row_index}:{text[:160]}",
        prefix="post",
    )
    return CleanPost(
        source="LongLaMPProductReviewTemporal",
        source_id=post_hash,
        author_hash=author_hash,
        raw_author_id=raw_author_id,
        private_topic="product_review_temporal",
        private_date=synthetic_date(row_index),
        text=text,
        word_count=len(tokens),
        content_tags=content_tags(tokens),
        style_features=style_features(text, tokens),
        task_input=task_input,
        private_metadata={
            "source_config": "product_review_temporal",
            "date_is_synthetic": True,
            "temporal_index_private": row_index,
            "source_input_private": str(row.get("input") or ""),
            "input_sanitizer_version": "longlamp-v1",
        },
    )


def clean_longlamp_product_profile_item(
    item: dict[str, Any],
    *,
    raw_author_id: str,
    row_index: int,
    item_index: int,
    min_words: int,
    max_words: int,
) -> CleanPost | None:
    text = normalize_text(str(item.get("reviewText") or ""))
    if not raw_author_id or not text:
        return None
    tokens = word_tokens(text)
    if reject_reasons(text, tokens, min_words=min_words, max_words=max_words):
        return None
    author_hash = stable_hash(f"longlamp-product:{raw_author_id}", prefix="author")
    description = str(item.get("description") or item.get("summary") or "a product")
    rating = str(item.get("overall") or "").strip()
    prompt = f'Write a product review for product context "{description}".'
    if rating:
        prompt = (
            f'Write a product review with rating "{rating}" for product context "{description}".'
        )
    post_hash = stable_hash(
        f"longlamp-product:profile:{raw_author_id}:{row_index}:{item_index}:{text[:160]}",
        prefix="post",
    )
    return CleanPost(
        source="LongLaMPProductReviewTemporal",
        source_id=post_hash,
        author_hash=author_hash,
        raw_author_id=raw_author_id,
        private_topic=f"product_review_temporal_rating_{rating or 'unknown'}",
        private_date=synthetic_date(max(0, row_index - 10_000 + item_index)),
        text=text,
        word_count=len(tokens),
        content_tags=content_tags(tokens),
        style_features=style_features(text, tokens),
        task_input=clipped_task_input(prompt),
        private_metadata={
            "source_config": "product_review_temporal",
            "date_is_synthetic": True,
            "temporal_index_private": row_index - 10_000 + item_index,
            "product_description_private": description,
            "overall_rating_private_or_public": rating,
            "source_profile_summary_private": str(item.get("summary") or ""),
            "input_sanitizer_version": "longlamp-v1",
        },
    )


def add_unique_post(posts: list[CleanPost], seen: set[str], post: CleanPost | None) -> None:
    if post is None or post.source_id in seen:
        return
    seen.add(post.source_id)
    posts.append(post)


def load_hf_dataset_stream(args: argparse.Namespace, *, dataset_name: str, config: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The Hugging Face datasets package is required. Run with: "
            "uv run --with datasets python -m auto_skill.author_style_cleaning ..."
        ) from exc

    if config:
        return load_dataset(
            dataset_name,
            config,
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
    return load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=False)


def load_blog_posts(args: argparse.Namespace) -> list[CleanPost]:
    dataset = load_hf_dataset_stream(args, dataset_name=args.hf_dataset, config=None)
    posts: list[CleanPost] = []
    scanned = 0
    for row in dataset:
        scanned += 1
        clean = clean_blog_row(row, min_words=args.min_words, max_words=args.max_words)
        if clean is not None:
            posts.append(clean)
        if scanned >= args.max_rows:
            break
    return posts


def load_longlamp_topic_posts(args: argparse.Namespace) -> list[CleanPost]:
    dataset = load_hf_dataset_stream(
        args,
        dataset_name=args.hf_dataset,
        config="topic_writing_temporal",
    )
    posts: list[CleanPost] = []
    seen: set[str] = set()
    for scanned, row in enumerate(dataset, start=1):
        add_unique_post(
            posts,
            seen,
            clean_longlamp_topic_output(
                row,
                row_index=scanned,
                min_words=args.min_words,
                max_words=args.max_words,
            ),
        )
        for item_index, item in enumerate(
            profile_items(row.get("profile"))[: args.longlamp_profile_items_per_row]
        ):
            add_unique_post(
                posts,
                seen,
                clean_longlamp_topic_profile_item(
                    item,
                    row_index=scanned,
                    item_index=item_index,
                    min_words=args.min_words,
                    max_words=args.max_words,
                ),
            )
        if scanned >= args.max_rows:
            break
    return posts


def load_longlamp_product_posts(args: argparse.Namespace) -> list[CleanPost]:
    dataset = load_hf_dataset_stream(
        args,
        dataset_name=args.hf_dataset,
        config="product_review_temporal",
    )
    posts: list[CleanPost] = []
    seen: set[str] = set()
    for scanned, row in enumerate(dataset, start=1):
        raw_author_id = str(row.get("reviewerId") or "").strip()
        add_unique_post(
            posts,
            seen,
            clean_longlamp_product_output(
                row,
                row_index=scanned,
                min_words=args.min_words,
                max_words=args.max_words,
            ),
        )
        for item_index, item in enumerate(
            profile_items(row.get("profile"))[: args.longlamp_profile_items_per_row]
        ):
            add_unique_post(
                posts,
                seen,
                clean_longlamp_product_profile_item(
                    item,
                    raw_author_id=raw_author_id,
                    row_index=scanned,
                    item_index=item_index,
                    min_words=args.min_words,
                    max_words=args.max_words,
                ),
            )
        if scanned >= args.max_rows:
            break
    return posts


def load_posts(args: argparse.Namespace) -> list[CleanPost]:
    if args.source == "blog":
        return load_blog_posts(args)
    if args.source == "longlamp-topic":
        return load_longlamp_topic_posts(args)
    if args.source == "longlamp-product":
        return load_longlamp_product_posts(args)
    if args.source == "mendeley-reddit":
        return load_mendeley_reddit_posts(args)
    raise ValueError(f"unsupported source: {args.source}")
