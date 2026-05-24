"""Stylometric cosine baseline for author-style retrieval triples."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from auto_skill.io.jsonl import load_jsonl
from auto_skill.metrics.author_style_reference import (
    ReferenceRetrievalJob,
    build_reference_retrieval_jobs,
    text_sha256,
)

TOKEN_RE = re.compile(r"[A-Za-z']+")
SENTENCE_RE = re.compile(r"[.!?]+")
PUNCTUATION = (".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "\n")
FUNCTION_WORD_CANDIDATES = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into
    is it its itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with you your yours
    yourself yourselves
    i'm i've i'd i'll you're you've you'd you'll he's he'd he'll she's she'd she'll it's
    it'd it'll we're we've we'd we'll they're they've they'd they'll that's there's here's
    where's what's who's don't doesn't didn't can't couldn't won't wouldn't isn't aren't
    wasn't weren't hasn't haven't hadn't
    """.split()
)


def jobs_from_probe_dir(probe_dir: Path) -> list[ReferenceRetrievalJob]:
    manifest = probe_dir / "sample_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing sample_manifest.json under {probe_dir}")
    return jobs_from_paths(
        packs=probe_dir / "oracle" / "packs.jsonl",
        private_eval=probe_dir / "oracle" / "private_eval.jsonl",
        hard_negatives=probe_dir / "oracle" / "hard_negatives.jsonl",
    )


def jobs_from_paths(
    *,
    packs: Path,
    private_eval: Path,
    hard_negatives: Path,
) -> list[ReferenceRetrievalJob]:
    jobs, skipped = build_reference_retrieval_jobs(
        packs=load_jsonl(packs),
        private_eval=load_jsonl(private_eval),
        hard_negatives=load_jsonl(hard_negatives),
    )
    if skipped:
        raise ValueError(f"cannot build stylometric jobs, skipped={len(skipped)}")
    return jobs


def run_baseline(
    jobs: list[ReferenceRetrievalJob],
    *,
    pack_metadata: dict[str, dict[str, Any]] | None = None,
    function_words: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    vocabulary = function_words or function_words_for_jobs(jobs)
    vectors = {text: feature_vector(text, vocabulary) for text in texts_for_jobs(jobs)}
    rows = []
    for job in jobs:
        metadata = (pack_metadata or {}).get(job.pack_id, {})
        target = vectors[job.target_text]
        scored = [
            (
                cosine(target, vectors[candidate.text]),
                candidate.candidate_id,
                candidate.kind,
            )
            for candidate in job.candidates
        ]
        score, selected_id, selected_kind = min(scored, key=lambda item: (-item[0], item[1]))
        rows.append(
            {
                "schema_version": "author-style-stylometric-baseline/v1",
                "pack_id": job.pack_id,
                "task_id": job.task_id,
                "source": job.source,
                "target_sha256": text_sha256(job.target_text),
                "hard_neg_variant": metadata.get("hard_neg_variant") or "unknown",
                "length_bucket": metadata.get("length_bucket") or "unknown",
                "selected_candidate_id": selected_id,
                "selected_candidate_kind": selected_kind,
                "expected_candidate_id": job.expected_candidate_id,
                "correct": selected_id == job.expected_candidate_id,
                "score": round(score, 6),
                "candidate_count": len(job.candidates),
                "candidate_manifest": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "kind": candidate.kind,
                        "text_sha256": text_sha256(candidate.text),
                    }
                    for candidate in job.candidates
                ],
            }
        )
    return rows


def feature_vector(text: str, function_words: tuple[str, ...]) -> dict[str, float]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    token_count = max(len(tokens), 1)
    counts = Counter(tokens)
    features = {
        f"fw:{word}": counts[word] / token_count
        for word in function_words
    }
    features.update(punctuation_features(text, token_count))
    features.update(sentence_features(text, tokens))
    features.update(char_ngram_features(text))
    return {key: value for key, value in features.items() if value}


def function_vocabulary(texts: list[str], *, limit: int) -> tuple[str, ...]:
    counts = Counter(
        token.lower()
        for text in texts
        for token in TOKEN_RE.findall(text)
        if token.lower() in FUNCTION_WORD_CANDIDATES
    )
    return tuple(word for word, _ in counts.most_common(limit))


def function_words_for_jobs(
    jobs: list[ReferenceRetrievalJob],
    *,
    limit: int = 200,
) -> tuple[str, ...]:
    return function_vocabulary(texts_for_jobs(jobs), limit=limit)


def punctuation_features(text: str, token_count: int) -> dict[str, float]:
    return {
        f"punct:{mark}": text.count(mark) * 100 / token_count
        for mark in PUNCTUATION
    }


def sentence_features(text: str, tokens: list[str]) -> dict[str, float]:
    lengths = [len(TOKEN_RE.findall(part)) for part in SENTENCE_RE.split(text) if part.strip()]
    clean = sorted(length for length in lengths if length > 0) or [len(tokens) or 1]
    return {
        "sent:mean": statistics.mean(clean) / 100,
        "sent:std": (statistics.pstdev(clean) if len(clean) > 1 else 0.0) / 100,
        "sent:p10": percentile(clean, 0.10) / 100,
        "sent:p90": percentile(clean, 0.90) / 100,
    }


def char_ngram_features(text: str) -> dict[str, float]:
    normalized = re.sub(r"\s+", " ", text.lower())
    counts: Counter[str] = Counter()
    for n in (3, 4):
        counts.update(
            normalized[index : index + n]
            for index in range(max(len(normalized) - n + 1, 0))
        )
    total = sum(counts.values()) or 1
    return {f"char:{gram}": count / total for gram, count in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def summarize(
    rows: list[dict[str, Any]],
    *,
    function_words: tuple[str, ...] = (),
    judge_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "author-style-stylometric-baseline-summary/v1",
        "overall": accuracy(rows),
        "by_source": group_accuracy(rows, "source"),
        "by_variant": group_accuracy(rows, "hard_neg_variant"),
        "by_length_bucket": group_accuracy(rows, "length_bucket"),
        "agreement": agreement_summary(rows, judge_rows or {}),
        "feature_config": {
            "function_word_count": len(function_words),
            "function_words": list(function_words),
        },
        "random_chance_reference": {
            "requested_reference": 0.25,
            "uniform_5_way": 0.2,
            "note": (
                "This task has 5 candidates, so true uniform chance is 20%; "
                "25% is included per spec."
            ),
        },
    }


def agreement_summary(
    rows: list[dict[str, Any]],
    judge_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    mine = {(row["pack_id"], row["task_id"]): row for row in rows}
    output = {}
    for name, raw_rows in judge_rows.items():
        latest = {(row.get("pack_id"), row.get("task_id")): row for row in raw_rows}
        overlap = [key for key in mine if key in latest and latest[key].get("status") == "success"]
        agree = sum(
            mine[key]["selected_candidate_id"] == latest[key].get("selected_candidate_id")
            for key in overlap
        )
        output[name] = {
            "overlap": len(overlap),
            "selected_agreement": round(agree / len(overlap), 6) if overlap else None,
        }
    return output


def group_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        str(value): accuracy([row for row in rows if row.get(key) == value])
        for value in sorted({row.get(key) for row in rows}, key=str)
    }


def accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for row in rows if row.get("correct") is True)
    low, high = wilson(correct, len(rows))
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 6) if rows else None,
        "ci95_low": low,
        "ci95_high": high,
    }


def wilson(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return round((centre - margin) / denom, 6), round((centre + margin) / denom, 6)


def percentile(values: list[int], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(values[low])
    return values[low] * (high - position) + values[high] * (position - low)


def texts_for_jobs(jobs: list[ReferenceRetrievalJob]) -> list[str]:
    return sorted(
        {job.target_text for job in jobs}
        | {candidate.text for job in jobs for candidate in job.candidates}
    )


def pack_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        metadata = row.get("probe_metadata")
        if isinstance(metadata, dict):
            output[str(row.get("pack_id") or "")] = metadata
    return output
