"""Shared types and text utilities for author-style cleaning."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
SCHEMA_VERSION = "author-style-pack/v1"
NATIVE_IMPOSTOR_NEGATIVE_TYPE = "original_av_impostor_cross_topic"
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "with",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class CleanPost:
    source: str
    source_id: str
    author_hash: str
    raw_author_id: str
    private_topic: str
    private_date: str
    text: str
    word_count: int
    content_tags: tuple[str, ...]
    style_features: dict[str, float]
    task_input: str | None = None
    private_metadata: dict[str, Any] = field(default_factory=dict)
    reject_flags: tuple[str, ...] = ()


def stable_hash(value: str, *, prefix: str, length: int = 12) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"\burlLink\b", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']+", text.lower())


def parse_blog_date(value: str) -> str | None:
    value = value.strip()
    for fmt in ("%d,%B,%Y", "%d,%b,%Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def content_tags(tokens: list[str], *, limit: int = 8) -> tuple[str, ...]:
    counts = Counter(token for token in tokens if token not in STOPWORDS and len(token) > 3)
    return tuple(token for token, _ in counts.most_common(limit))


def sentence_count(text: str) -> int:
    count = len(re.findall(r"[.!?]+(?:\s|$)", text))
    return max(1, count)


def style_features(text: str, tokens: list[str]) -> dict[str, float]:
    words = max(1, len(tokens))
    sentences = sentence_count(text)
    lower_tokens = set(tokens)
    first_person = sum(1 for token in tokens if token in {"i", "me", "my", "mine"})
    contractions = sum(1 for token in tokens if "'" in token)
    return {
        "avg_sentence_words": round(words / sentences, 3),
        "question_per_100w": round(text.count("?") * 100 / words, 3),
        "exclamation_per_100w": round(text.count("!") * 100 / words, 3),
        "ellipsis_per_100w": round(text.count("...") * 100 / words, 3),
        "paren_per_100w": round((text.count("(") + text.count(")")) * 100 / words, 3),
        "first_person_per_100w": round(first_person * 100 / words, 3),
        "contraction_per_100w": round(contractions * 100 / words, 3),
        "hedge_marker": float(bool(lower_tokens & {"maybe", "probably", "somewhat", "perhaps"})),
    }


def reject_reasons(text: str, tokens: list[str], *, min_words: int, max_words: int) -> list[str]:
    reasons: list[str] = []
    word_count = len(tokens)
    if word_count < min_words:
        reasons.append("too_short")
    if word_count > max_words:
        reasons.append("too_long")
    if not re.search(r"[.!?]", text):
        reasons.append("no_sentence_punctuation")
    if len(set(tokens)) < max(20, int(word_count * 0.25)):
        reasons.append("low_lexical_diversity")
    copied_markers = ("newsgroups:", "organization:", "copyright", "all rights reserved")
    if any(marker in text.lower() for marker in copied_markers):
        reasons.append("copied_or_forwarded_marker")
    if text.lower().count("lyrics") or text.lower().count("chorus"):
        reasons.append("possible_lyrics")
    alpha = sum(1 for char in text if char.isalpha())
    if alpha and sum(1 for char in text if char.isupper()) / alpha > 0.35:
        reasons.append("upper_case_heavy")
    return reasons


def normalized_text_key(text: str) -> str:
    return normalize_text(text).casefold()


def minimal_reject_reasons(
    *,
    text: str,
    raw_author_id: str,
    min_normalized_chars: int,
) -> list[str]:
    reasons: list[str] = []
    if not raw_author_id:
        reasons.append("empty_author")
    if not text:
        reasons.append("empty_text")
    elif len(text) < min_normalized_chars:
        reasons.append("too_few_normalized_chars")
    return reasons


def dedupe_posts_by_text(posts: list[CleanPost]) -> tuple[list[CleanPost], dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[CleanPost] = []
    duplicate_rows = 0
    for post in posts:
        key = normalized_text_key(post.text)
        if key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)
        deduped.append(post)
    return deduped, {
        "enabled": True,
        "key": "normalized_text.casefold",
        "input_rows": len(posts),
        "output_rows": len(deduped),
        "duplicate_rows_dropped": duplicate_rows,
    }


def clean_blog_row(
    row: dict[str, Any],
    *,
    min_words: int,
    max_words: int,
    min_normalized_chars: int = 20,
    filter_policy: str = "strict",
) -> CleanPost | None:
    raw_author_id = str(row.get("id") or "").strip()
    raw_date = str(row.get("date") or "").strip()
    parsed_date = parse_blog_date(raw_date)
    if not raw_author_id or parsed_date is None:
        return None
    text = normalize_text(str(row.get("text") or ""))
    tokens = word_tokens(text)
    if filter_policy == "minimal":
        reasons = minimal_reject_reasons(
            text=text,
            raw_author_id=raw_author_id,
            min_normalized_chars=min_normalized_chars,
        )
    else:
        reasons = reject_reasons(text, tokens, min_words=min_words, max_words=max_words)
    if reasons:
        return None
    author_hash = stable_hash(f"blog-authorship:{raw_author_id}", prefix="author")
    post_hash = stable_hash(
        f"blog-authorship:{raw_author_id}:{raw_date}:{text[:160]}",
        prefix="post",
    )
    return CleanPost(
        source="BlogAuthorship",
        source_id=post_hash,
        author_hash=author_hash,
        raw_author_id=raw_author_id,
        private_topic=str(row.get("topic") or "unknown"),
        private_date=parsed_date,
        text=text,
        word_count=len(tokens),
        content_tags=content_tags(tokens),
        style_features=style_features(text, tokens),
        task_input=None,
    )
