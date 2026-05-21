"""Shared deterministic metric helpers for author-style quality reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

TOPIC_MARKERS = {
    "anime",
    "australian_context",
    "canadian",
    "conspiracy",
    "finance",
    "food",
    "gaming",
    "gamer",
    "goth",
    "law",
    "pc_vocab",
    "political",
    "pop_culture",
    "religious",
    "rights",
    "sports",
    "technical",
    "uk_colloquialisms",
}
STYLE_NORMALIZERS = {
    "avg_sentence_words": 30.0,
    "question_per_100w": 3.0,
    "exclamation_per_100w": 3.0,
    "ellipsis_per_100w": 2.0,
    "paren_per_100w": 3.0,
    "first_person_per_100w": 8.0,
    "contraction_per_100w": 4.0,
    "hedge_marker": 1.0,
}


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(float(value)):
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 3),
        "mean": mean(values),
        "max": round(max(values), 3),
    }


def distribution(values: list[float]) -> dict[str, int]:
    return dict(Counter(str(int(value)) for value in values))


def tag_counts(tag_lists: list[list[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for tags in tag_lists:
        counts.update(str(tag) for tag in tags if isinstance(tag, str) and tag)
    return dict(counts.most_common())


def top_items(counter: Counter[str], *, limit: int = 20) -> dict[str, int]:
    return dict(counter.most_common(limit))


def normalize_style_tag(tag: str) -> str | None:
    """Map sparse GPT audit tags into coarse style clusters.

    Topic/register tags are intentionally dropped from the canonical style view
    and reported separately so downstream clustering does not silently become
    subject-matter clustering.
    """

    raw = tag.strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return None
    if any(marker in raw for marker in TOPIC_MARKERS):
        return None
    if "ellipsis" in raw or "ellipses" in raw:
        return "ellipsis_heavy"
    if "exclamation" in raw or raw in {"heavy_exclamation", "high_exclamation"}:
        return "exclamation_heavy"
    if "question" in raw or "rhetorical" in raw:
        return "question_heavy"
    if any(word in raw for word in ("sarcas", "snark", "mock", "dry_humor", "absurdist")):
        return "sarcasm_snark"
    if "parenthetical" in raw or "paren" in raw or "aside" in raw:
        return "parenthetical_asides"
    if "contraction" in raw or "apostrophe" in raw:
        return "conversational_contractions"
    if "short" in raw and any(word in raw for word in ("sentence", "fragment", "punchy")):
        return "short_sentences"
    if "long" in raw or "run_on" in raw or "rambling" in raw or "comma_splice" in raw:
        return "long_run_on"
    if "profane" in raw or "profanity" in raw or "swear" in raw or "cuss" in raw:
        return "profanity"
    if any(word in raw for word in ("emoticon", "emoji", "lol", "lmfao", "haha", "hehe")):
        return "emoticon_laughter"
    if any(
        word in raw
        for word in ("misspell", "typo", "spelling", "grammar", "non_native", "esl")
    ):
        return "nonstandard_orthography"
    if "lowercase_i" in raw:
        return "lowercase_i"
    if "caps" in raw or "all_caps" in raw:
        return "caps_emphasis"
    if any(word in raw for word in ("first_person", "diary", "confessional", "anecdotal")):
        return "first_person_diary"
    if any(word in raw for word in ("direct_address", "second_person", "reader_address")):
        return "direct_address"
    if any(word in raw for word in ("hedg", "qualified", "speculative", "soft_hedging")):
        return "hedging"
    if any(word in raw for word in ("topic_shift", "abrupt", "fragmented_comment")):
        return "abrupt_topic_shifts"
    if any(word in raw for word in ("casual", "informal", "conversational", "forum", "reddit")):
        return "casual_conversational"
    if "quote" in raw:
        return "quote_response"
    if "edit_marker" in raw or "self_correction" in raw:
        return "edit_self_correction"
    if "supportive" in raw or "thanks" in raw or "polite" in raw:
        return "supportive_polite"
    return None


def split_canonical_tags(tags: list[str]) -> tuple[list[str], list[str]]:
    canonical = []
    topic_like = []
    for tag in tags:
        normalized = normalize_style_tag(tag)
        raw = tag.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized is not None:
            canonical.append(normalized)
        elif any(marker in raw for marker in TOPIC_MARKERS):
            topic_like.append(raw)
    return sorted(set(canonical)), sorted(set(topic_like))


def average_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    features = [
        row.get("style_features")
        for row in rows
        if isinstance(row.get("style_features"), dict)
    ]
    keys = sorted({key for row in features for key in row})
    averaged = {}
    for key in keys:
        values = [value for row in features if (value := numeric(row.get(key))) is not None]
        if values:
            averaged[key] = float(mean(values))
    return averaged


def style_similarity_from_features(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    distances = []
    for key, normalizer in STYLE_NORMALIZERS.items():
        left_value = numeric(left.get(key))
        right_value = numeric(right.get(key))
        if left_value is None or right_value is None:
            continue
        distances.append(abs(left_value - right_value) / normalizer)
    if not distances:
        return None
    return round(max(0.0, 1.0 - min(1.0, sum(distances) / len(distances))), 3)


def features_from_text(text: str) -> dict[str, float]:
    from auto_skill.cleaning.author_style.common import style_features, word_tokens

    tokens = word_tokens(text)
    return style_features(text, tokens)


def content_tag_jaccard(left: list[str], right: list[str]) -> float | None:
    left_set = {tag for tag in left if isinstance(tag, str) and tag}
    right_set = {tag for tag in right if isinstance(tag, str) and tag}
    if not left_set or not right_set:
        return None
    return round(len(left_set & right_set) / len(left_set | right_set), 3)


def row_by_pack(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        pack_id = row.get("pack_id")
        if isinstance(pack_id, str) and pack_id:
            indexed[pack_id] = row
    return indexed


def negatives_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_ref = row.get("target_task_ref")
        if isinstance(task_ref, str) and task_ref:
            grouped[task_ref].append(row)
    return grouped


def source_pool_heldout_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = []
    for row in rows:
        author_hash = row.get("author_hash")
        pack_id = row.get("pack_id")
        if not isinstance(author_hash, str) or not isinstance(pack_id, str):
            continue
        for heldout in row.get("heldout_private", []):
            if not isinstance(heldout, dict):
                continue
            pool_row = dict(heldout)
            pool_row["author_hash"] = author_hash
            pool_row["pack_id"] = pack_id
            pool.append(pool_row)
    return pool


def gpt_audit_payload(audit_row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(audit_row, dict) or audit_row.get("status") != "success":
        return {}
    audit = audit_row.get("audit")
    return audit if isinstance(audit, dict) else {}


def task_refs(pack: dict[str, Any]) -> list[str]:
    refs = []
    for task in pack.get("heldout_tasks", []):
        task_id = task.get("task_id")
        if isinstance(task_id, str) and task_id:
            refs.append(task_id)
    return refs
