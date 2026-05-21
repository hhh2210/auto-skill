"""Source-specific loaders for public author-style corpora."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
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


MENDELEY_FILENAME_RE = re.compile(
    r"^(?P<role>known|unknown) - (?P<doc_author>.*?) - "
    r"\[(?P<subreddit>[^\]]+)\]\[(?P<year>\d{4})\]\.txt$"
)


def parse_mendeley_filename(path_name: str) -> dict[str, str] | None:
    path = Path(path_name)
    if len(path.parts) < 2 or path.suffix.lower() != ".txt":
        return None
    filename = path.name
    if filename in {"truth.txt"}:
        return None
    match = MENDELEY_FILENAME_RE.match(filename)
    if not match:
        return None
    return {
        "problem_id": path.parts[-2],
        "role": match.group("role"),
        "doc_author": match.group("doc_author"),
        "subreddit": match.group("subreddit"),
        "year": match.group("year"),
    }


def parse_mendeley_truth(text: str) -> dict[str, str]:
    truth: dict[str, str] = {}
    for line in text.replace("\ufeff", "").splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        problem_id, label = parts
        if label in {"Y", "N"}:
            truth[problem_id] = label
    return truth


def iter_mendeley_text_entries(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            truth_text = archive.read("truth.txt").decode("utf-8-sig", errors="replace")
            entries = [
                (name, archive.read(name).decode("utf-8-sig", errors="replace"))
                for name in archive.namelist()
                if parse_mendeley_filename(name) is not None
            ]
        return parse_mendeley_truth(truth_text), entries

    if path.is_dir():
        truth_path = path / "truth.txt"
        if not truth_path.exists():
            raise FileNotFoundError(f"Mendeley truth.txt not found under {path}")
        truth = parse_mendeley_truth(truth_path.read_text(encoding="utf-8-sig"))
        entries = [
            (str(file.relative_to(path)), file.read_text(encoding="utf-8-sig", errors="replace"))
            for file in path.rglob("*.txt")
            if parse_mendeley_filename(str(file.relative_to(path))) is not None
        ]
        return truth, entries

    raise FileNotFoundError(f"Mendeley path does not exist: {path}")


def mendeley_task_input(post: CleanPost) -> str:
    tags = ", ".join(post.content_tags[:5])
    stem = "Continue the same anonymous online discussion user's authentic voice"
    if tags:
        return stem + " while writing a new comment bundle touching on: " + tags
    return stem + " while writing a new comment bundle."


def clean_mendeley_reddit_doc(
    *,
    problem_id: str,
    truth_label: str,
    role: str,
    doc_author: str,
    subreddit: str,
    year: str,
    text: str,
    role_index: int,
    min_words: int,
    max_words: int,
) -> CleanPost | None:
    text = normalize_text(text)
    tokens = word_tokens(text)
    if reject_reasons(text, tokens, min_words=min_words, max_words=max_words):
        return None
    is_same_author_problem = truth_label == "Y"
    author_hash = stable_hash(f"mendeley-reddit:{doc_author}", prefix="author")
    source_id = stable_hash(
        f"mendeley-reddit:{problem_id}:{truth_label}:{role}:{role_index}:{doc_author}:{subreddit}:{year}",
        prefix="post",
    )
    synthetic_index = role_index if role == "known" else 10_000
    post = CleanPost(
        source="MendeleyRedditCrossTopic",
        source_id=source_id,
        author_hash=author_hash,
        raw_author_id=doc_author,
        private_topic=subreddit,
        private_date=synthetic_date(synthetic_index),
        text=text,
        word_count=len(tokens),
        content_tags=content_tags(tokens),
        style_features=style_features(text, tokens),
        private_metadata={
            "source_config": "reddit_cross_topic_pan",
            "date_is_synthetic": True,
            "mendeley_problem_id_private": problem_id,
            "truth_label_private": truth_label,
            "doc_role_private": role,
            "doc_role_index_private": role_index,
            "actual_doc_author_private": doc_author,
            "claimed_problem_author_private": problem_id,
            "actual_year_private": year,
            "subreddit_private": subreddit,
            "original_problem_same_author": is_same_author_problem,
            "input_sanitizer_version": "mendeley-reddit-v1",
        },
    )
    return CleanPost(
        source=post.source,
        source_id=post.source_id,
        author_hash=post.author_hash,
        raw_author_id=post.raw_author_id,
        private_topic=post.private_topic,
        private_date=post.private_date,
        text=post.text,
        word_count=post.word_count,
        content_tags=post.content_tags,
        style_features=post.style_features,
        task_input=mendeley_task_input(post),
        private_metadata=post.private_metadata,
        reject_flags=post.reject_flags,
    )


def load_mendeley_reddit_posts(args: argparse.Namespace) -> list[CleanPost]:
    if args.mendeley_path is None:
        raise SystemExit("--mendeley-path is required when --source=mendeley-reddit")
    truth, entries = iter_mendeley_text_entries(args.mendeley_path)
    known_counts: dict[str, int] = defaultdict(int)
    unknown_counts: dict[str, int] = defaultdict(int)
    posts: list[CleanPost] = []
    seen: set[str] = set()
    for path_name, text in entries:
        parsed = parse_mendeley_filename(path_name)
        if parsed is None:
            continue
        problem_id = parsed["problem_id"]
        truth_label = truth.get(problem_id)
        if truth_label not in {"Y", "N"}:
            continue
        role = parsed["role"]
        if role == "known":
            known_counts[problem_id] += 1
            role_index = known_counts[problem_id] - 1
        else:
            unknown_counts[problem_id] += 1
            role_index = unknown_counts[problem_id] - 1
        add_unique_post(
            posts,
            seen,
            clean_mendeley_reddit_doc(
                problem_id=problem_id,
                truth_label=truth_label,
                role=role,
                doc_author=parsed["doc_author"],
                subreddit=parsed["subreddit"],
                year=parsed["year"],
                text=text,
                role_index=role_index,
                min_words=args.min_words,
                max_words=args.max_words,
            ),
        )
    return posts


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
