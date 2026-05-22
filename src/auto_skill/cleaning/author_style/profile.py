"""Raw corpus profiling for author-style benchmark sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_skill.cleaning.author_style.common import normalize_text, parse_blog_date, word_tokens
from auto_skill.cleaning.author_style.source_mendeley import (
    iter_mendeley_text_entries,
    parse_mendeley_filename,
)
from auto_skill.cleaning.author_style.sources import load_hf_dataset_stream

GOPHER_STOPWORDS = ("the", "be", "to", "of", "and", "that", "have", "with")


@dataclass(frozen=True)
class ProfileRecord:
    source: str
    raw_author_id: str
    text: str
    topic: str = ""
    date: str = ""


def percentile(values: Iterable[int | float], q: float) -> float | None:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    index = (len(sorted_values) - 1) * q / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def stats(values: Iterable[int | float]) -> dict[str, float | int | None]:
    materialized = list(values)
    if not materialized:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(materialized),
        "min": min(materialized),
        "p10": round(percentile(materialized, 10) or 0, 3),
        "p50": round(percentile(materialized, 50) or 0, 3),
        "p90": round(percentile(materialized, 90) or 0, 3),
        "p99": round(percentile(materialized, 99) or 0, 3),
        "max": max(materialized),
        "mean": round(sum(materialized) / len(materialized), 3),
    }


def gini(values: Iterable[int | float]) -> float | None:
    sorted_values = sorted(float(value) for value in values if value >= 0)
    if not sorted_values:
        return None
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(sorted_values))
    n = len(sorted_values)
    return round((2 * weighted) / (n * total) - (n + 1) / n, 6)


def sentence_count(text: str) -> int:
    return max(1, sum(1 for char in text if char in ".!?"))


def record_from_blog_row(row: dict[str, Any]) -> ProfileRecord:
    return ProfileRecord(
        source="blog",
        raw_author_id=str(row.get("id") or "").strip(),
        text=str(row.get("text") or ""),
        topic=str(row.get("topic") or ""),
        date=parse_blog_date(str(row.get("date") or "")) or "",
    )


def iter_blog_records(args: argparse.Namespace) -> Iterable[ProfileRecord]:
    dataset = load_hf_dataset_stream(args, dataset_name=args.hf_dataset, config=None)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    for index, row in enumerate(dataset, start=1):
        yield record_from_blog_row(row)
        if max_rows is not None and index >= max_rows:
            break


def iter_mendeley_records(path: Path, *, max_rows: int | None = None) -> Iterable[ProfileRecord]:
    truth, entries = iter_mendeley_text_entries(path)
    for index, (path_name, text) in enumerate(entries, start=1):
        parsed = parse_mendeley_filename(path_name)
        if parsed is None:
            continue
        if truth.get(parsed["problem_id"]) not in {"Y", "N"}:
            continue
        yield ProfileRecord(
            source="mendeley-reddit",
            raw_author_id=parsed["doc_author"],
            text=text,
            topic=parsed["subreddit"],
            date=parsed["year"],
        )
        if max_rows is not None and index >= max_rows:
            break


def _text_hash(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _top_one_percent_share(counts: list[int]) -> float | None:
    if not counts:
        return None
    top_n = max(1, math.ceil(len(counts) * 0.01))
    return round(sum(sorted(counts, reverse=True)[:top_n]) / sum(counts), 6)


def profile_records(
    records: Iterable[ProfileRecord],
    *,
    source: str,
    train_posts: int = 1,
    heldout_posts: int = 1,
) -> dict[str, Any]:
    required_posts = train_posts + heldout_posts
    author_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    text_hash_counts: Counter[str] = Counter()
    words: list[int] = []
    chars: list[int] = []
    sentences: list[int] = []
    avg_sentence_words: list[float] = []
    unique_token_ratios: list[float] = []
    stopword_hits: list[int] = []
    mean_word_lengths: list[float] = []
    empty_text = 0
    empty_author = 0
    profiled_rows = 0

    for record in records:
        profiled_rows += 1
        author_id = record.raw_author_id.strip()
        text = normalize_text(record.text)
        if not author_id:
            empty_author += 1
        if not text:
            empty_text += 1
        if author_id:
            author_counts[author_id] += 1
        if record.topic:
            topic_counts[record.topic] += 1
        if text:
            text_hash_counts[_text_hash(text)] += 1

        tokens = word_tokens(text)
        word_count = len(tokens)
        sentence_total = sentence_count(text)
        words.append(word_count)
        chars.append(len(text))
        sentences.append(sentence_total)
        avg_sentence_words.append(round(word_count / sentence_total, 3))
        if tokens:
            unique_token_ratios.append(round(len(set(tokens)) / len(tokens), 6))
            mean_word_lengths.append(round(sum(len(token) for token in tokens) / len(tokens), 3))
        else:
            unique_token_ratios.append(0.0)
            mean_word_lengths.append(0.0)
        token_set = set(tokens)
        stopword_hits.append(sum(1 for stopword in GOPHER_STOPWORDS if stopword in token_set))

    duplicate_rows = sum(count - 1 for count in text_hash_counts.values() if count > 1)
    duplicate_groups = sum(1 for count in text_hash_counts.values() if count > 1)
    author_post_counts = list(author_counts.values())
    feasible_authors = sum(1 for count in author_post_counts if count >= required_posts)

    return {
        "schema_version": "author-style-corpus-profile/v1",
        "source": source,
        "profiled_rows": profiled_rows,
        "empty_text_rows": empty_text,
        "empty_author_rows": empty_author,
        "exact_text_duplicate_rows": duplicate_rows,
        "exact_text_duplicate_groups": duplicate_groups,
        "exact_text_duplicate_rate": round(duplicate_rows / profiled_rows, 6)
        if profiled_rows
        else None,
        "author_count": len(author_counts),
        "topic_count": len(topic_counts),
        "posts_per_author": stats(author_post_counts),
        "author_post_gini": gini(author_post_counts),
        "top_1pct_author_row_share": _top_one_percent_share(author_post_counts),
        "authors_with_at_least_2_posts": sum(1 for count in author_post_counts if count >= 2),
        "authors_with_at_least_5_posts": sum(1 for count in author_post_counts if count >= 5),
        "authors_with_at_least_10_posts": sum(1 for count in author_post_counts if count >= 10),
        "triple_feasibility": {
            "train_posts": train_posts,
            "heldout_posts": heldout_posts,
            "required_posts_per_author": required_posts,
            "author_count": len(author_counts),
            "feasible_author_count": feasible_authors,
            "feasible_author_rate": round(feasible_authors / len(author_counts), 6)
            if author_counts
            else None,
        },
        "word_count": stats(words),
        "char_count": stats(chars),
        "sentence_count": stats(sentences),
        "avg_sentence_words": stats(avg_sentence_words),
        "unique_token_ratio": stats(unique_token_ratios),
        "mean_word_length": stats(mean_word_lengths),
        "gopher_stopword_hits": stats(stopword_hits),
        "rows_with_stopword_hits_lt_2": sum(1 for value in stopword_hits if value < 2),
        "top_topics": topic_counts.most_common(20),
    }


def write_profile_json(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_profile_markdown(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"# Author-Style Corpus Profile: {profile['source']}",
        "",
        f"- Rows: {profile['profiled_rows']}",
        f"- Authors: {profile['author_count']}",
        f"- Exact duplicate rows: {profile['exact_text_duplicate_rows']} "
        f"({profile['exact_text_duplicate_rate']})",
        f"- Empty text rows: {profile['empty_text_rows']}",
        f"- Empty author rows: {profile['empty_author_rows']}",
        f"- Author post Gini: {profile['author_post_gini']}",
        f"- Top 1% author row share: {profile['top_1pct_author_row_share']}",
        "",
        "## Length",
        "",
        _markdown_stats("word_count", profile["word_count"]),
        _markdown_stats("char_count", profile["char_count"]),
        _markdown_stats("sentence_count", profile["sentence_count"]),
        _markdown_stats("avg_sentence_words", profile["avg_sentence_words"]),
        "",
        "## Authors",
        "",
        _markdown_stats("posts_per_author", profile["posts_per_author"]),
        f"- Authors with >=2 posts: {profile['authors_with_at_least_2_posts']}",
        f"- Authors with >=5 posts: {profile['authors_with_at_least_5_posts']}",
        f"- Authors with >=10 posts: {profile['authors_with_at_least_10_posts']}",
        f"- Triple-feasible authors: {profile['triple_feasibility']['feasible_author_count']} "
        f"/ {profile['triple_feasibility']['author_count']}",
        "",
        "## Text Quality Signals",
        "",
        _markdown_stats("unique_token_ratio", profile["unique_token_ratio"]),
        _markdown_stats("mean_word_length", profile["mean_word_length"]),
        _markdown_stats("gopher_stopword_hits", profile["gopher_stopword_hits"]),
        f"- Rows with <2 Gopher stopword hits: {profile['rows_with_stopword_hits_lt_2']}",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _markdown_stats(name: str, values: dict[str, Any]) -> str:
    keys = ("min", "p10", "p50", "p90", "p99", "max", "mean")
    body = ", ".join(f"{key}={values[key]}" for key in keys)
    return f"- `{name}`: {body}"
