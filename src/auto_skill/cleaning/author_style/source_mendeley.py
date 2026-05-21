"""Mendeley Reddit source loader for author-style cleaning."""

from __future__ import annotations

import argparse
import re
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from auto_skill.cleaning.author_style.common import (
    CleanPost,
    content_tags,
    normalize_text,
    reject_reasons,
    stable_hash,
    style_features,
    word_tokens,
)

MENDELEY_FILENAME_RE = re.compile(
    r"^(?P<role>known|unknown) - (?P<doc_author>.*?) - "
    r"\[(?P<subreddit>[^\]]+)\]\[(?P<year>\d{4})\]\.txt$"
)


def _synthetic_date(index: int) -> str:
    return (date(2000, 1, 1) + timedelta(days=index)).isoformat()


def _add_unique_post(posts: list[CleanPost], seen: set[str], post: CleanPost | None) -> None:
    if post is None or post.source_id in seen:
        return
    seen.add(post.source_id)
    posts.append(post)


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
        private_date=_synthetic_date(synthetic_index),
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
        _add_unique_post(
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
