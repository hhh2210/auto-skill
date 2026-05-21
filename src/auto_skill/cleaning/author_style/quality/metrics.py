"""Metric helpers for cleaned personal author-style artifacts."""

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


def hard_negative_metrics(negatives: list[dict[str, Any]]) -> dict[str, Any]:
    confusability = []
    topic_shortcut = []
    style_similarity = []
    length_ratio = []
    content_jaccard = []
    time_score = []
    same_topic_count = 0
    selected_count = 0
    rerank_status = Counter()
    negative_types = Counter()
    for row in negatives:
        negative_types[str(row.get("negative_type") or "")] += 1
        rerank = row.get("gpt_rerank")
        if isinstance(rerank, dict):
            status = str(rerank.get("status") or "")
            rerank_status[status] += 1
            if status == "selected":
                selected_count += 1
            value = numeric(rerank.get("style_confusability_1_to_5"))
            if value is not None:
                confusability.append(value)
            value = numeric(rerank.get("topic_shortcut_risk_1_to_5"))
            if value is not None:
                topic_shortcut.append(value)
        features = row.get("match_features")
        if isinstance(features, dict):
            if features.get("same_topic") is True:
                same_topic_count += 1
            for key, bucket in (
                ("style_similarity", style_similarity),
                ("length_ratio", length_ratio),
                ("content_tag_jaccard", content_jaccard),
                ("time_score", time_score),
            ):
                value = numeric(features.get(key))
                if value is not None:
                    bucket.append(value)
    total = len(negatives)
    return {
        "count": total,
        "selected_count": selected_count,
        "rerank_status_counts": dict(rerank_status),
        "negative_type_counts": dict(negative_types),
        "mean_style_confusability_1_to_5": mean(confusability),
        "style_confusability_distribution": distribution(confusability),
        "mean_topic_shortcut_risk_1_to_5": mean(topic_shortcut),
        "mean_prefilter_style_similarity": mean(style_similarity),
        "prefilter_style_similarity_stats": stats(style_similarity),
        "mean_length_ratio": mean(length_ratio),
        "length_ratio_stats": stats(length_ratio),
        "mean_content_tag_jaccard": mean(content_jaccard),
        "content_tag_jaccard_stats": stats(content_jaccard),
        "mean_time_score": mean(time_score),
        "time_score_stats": stats(time_score),
        "same_topic_rate": round(same_topic_count / total, 3) if total else None,
    }


def deterministic_pack_metrics(
    private: dict[str, Any],
    negatives_by_ref: dict[str, list[dict[str, Any]]],
    source_pool_heldout: list[dict[str, Any]],
) -> dict[str, Any]:
    train_rows = [
        row for row in private.get("train_private", []) if isinstance(row, dict)
    ]
    heldout_rows = [
        row for row in private.get("heldout_private", []) if isinstance(row, dict)
    ]
    train_mean = average_features(train_rows)
    heldout_style_similarity = []
    heldout_style_similarity_by_task = []
    train_negative_style_similarity = []
    train_source_pool_style_similarity = []
    source_pool_task_metrics = []
    source_pool_pairwise_win_rates = []
    source_pool_margins = []
    source_pool_counts = []
    source_pool_content_jaccards = []
    source_pool_length_ratios = []
    source_pool_same_topic_rates = []
    heldout_topic_seen = []
    train_heldout_content_jaccard = []
    train_word_counts = [
        float(value)
        for row in train_rows
        if (value := numeric(row.get("word_count"))) is not None
    ]
    heldout_word_counts = [
        float(value)
        for row in heldout_rows
        if (value := numeric(row.get("word_count"))) is not None
    ]
    negative_word_counts = []
    train_topics = {
        str(row.get("private_topic"))
        for row in train_rows
        if isinstance(row.get("private_topic"), str)
    }
    heldout_topics = {
        str(row.get("private_topic"))
        for row in heldout_rows
        if isinstance(row.get("private_topic"), str)
    }
    seen_heldout_topics = train_topics & heldout_topics
    train_content_tags = [
        str(tag)
        for row in train_rows
        for tag in row.get("content_tags", [])
        if isinstance(tag, str)
    ]
    author_hash = private.get("author_hash")
    for row in heldout_rows:
        same_author_similarity = None
        heldout_features = row.get("style_features")
        if isinstance(heldout_features, dict):
            similarity = style_similarity_from_features(train_mean, heldout_features)
            if similarity is not None:
                heldout_style_similarity.append(similarity)
                same_author_similarity = similarity
                heldout_style_similarity_by_task.append(
                    {"task_ref": row.get("task_ref"), "style_similarity": similarity}
                )
        topic = row.get("private_topic")
        if isinstance(topic, str) and train_topics:
            heldout_topic_seen.append(1.0 if topic in train_topics else 0.0)
        tags = [str(tag) for tag in row.get("content_tags", []) if isinstance(tag, str)]
        jaccard = content_tag_jaccard(train_content_tags, tags)
        if jaccard is not None:
            train_heldout_content_jaccard.append(jaccard)
        task_ref_for_pool = row.get("task_ref")
        heldout_word_count = numeric(row.get("word_count"))
        source_pool_similarities = []
        task_source_pool_content_jaccards = []
        task_source_pool_length_ratios = []
        task_source_pool_same_topic = []
        for candidate in source_pool_heldout:
            if candidate.get("author_hash") == author_hash:
                continue
            candidate_features = candidate.get("style_features")
            if not isinstance(candidate_features, dict):
                continue
            similarity = style_similarity_from_features(train_mean, candidate_features)
            if similarity is None:
                continue
            source_pool_similarities.append(similarity)
            train_source_pool_style_similarity.append(similarity)
            candidate_tags = [
                str(tag)
                for tag in candidate.get("content_tags", [])
                if isinstance(tag, str)
            ]
            candidate_jaccard = content_tag_jaccard(tags, candidate_tags)
            if candidate_jaccard is not None:
                task_source_pool_content_jaccards.append(candidate_jaccard)
                source_pool_content_jaccards.append(candidate_jaccard)
            candidate_word_count = numeric(candidate.get("word_count"))
            if (
                candidate_word_count is not None
                and heldout_word_count is not None
                and heldout_word_count > 0
            ):
                length_ratio = round(candidate_word_count / heldout_word_count, 3)
                task_source_pool_length_ratios.append(length_ratio)
                source_pool_length_ratios.append(length_ratio)
            candidate_topic = candidate.get("private_topic")
            if isinstance(topic, str) and isinstance(candidate_topic, str):
                same_topic = 1.0 if candidate_topic == topic else 0.0
                task_source_pool_same_topic.append(same_topic)
        if source_pool_similarities:
            pool_mean = mean(source_pool_similarities)
            margin = (
                round(same_author_similarity - pool_mean, 3)
                if same_author_similarity is not None and pool_mean is not None
                else None
            )
            win_rate = (
                round(
                    sum(1 for value in source_pool_similarities if same_author_similarity > value)
                    / len(source_pool_similarities),
                    3,
                )
                if same_author_similarity is not None
                else None
            )
            if margin is not None:
                source_pool_margins.append(margin)
            if win_rate is not None:
                source_pool_pairwise_win_rates.append(win_rate)
            source_pool_counts.append(float(len(source_pool_similarities)))
            source_pool_same_topic_rates.extend(task_source_pool_same_topic)
            source_pool_task_metrics.append(
                {
                    "task_ref": task_ref_for_pool,
                    "same_author_style_similarity": same_author_similarity,
                    "source_pool_impostor_count": len(source_pool_similarities),
                    "mean_source_pool_style_similarity": pool_mean,
                    "margin_vs_source_pool_impostors": margin,
                    "source_pool_pairwise_win_rate": win_rate,
                    "mean_source_pool_content_tag_jaccard": mean(
                        task_source_pool_content_jaccards
                    ),
                    "mean_source_pool_length_ratio": mean(task_source_pool_length_ratios),
                    "source_pool_same_topic_rate": mean(task_source_pool_same_topic),
                }
            )
        task_ref = row.get("task_ref")
        if not isinstance(task_ref, str):
            continue
        for negative in negatives_by_ref.get(task_ref, []):
            text = negative.get("public_negative_text")
            if isinstance(text, str) and text.strip():
                negative_features = features_from_text(text)
                similarity = style_similarity_from_features(train_mean, negative_features)
                if similarity is not None:
                    train_negative_style_similarity.append(similarity)
                negative_word_counts.append(float(max(1, len(text.split()))))
    same_author_mean = mean(heldout_style_similarity)
    negative_mean = mean(train_negative_style_similarity)
    margin = (
        round(same_author_mean - negative_mean, 3)
        if same_author_mean is not None and negative_mean is not None
        else None
    )
    return {
        "mean_train_heldout_style_similarity": same_author_mean,
        "train_heldout_style_similarity_by_task": heldout_style_similarity_by_task,
        "mean_train_negative_style_similarity": negative_mean,
        "train_negative_style_similarity_stats": stats(train_negative_style_similarity),
        "style_similarity_margin_vs_negatives": margin,
        "margin_vs_selected_hard_negatives": margin,
        "mean_train_source_pool_impostor_style_similarity": mean(
            train_source_pool_style_similarity
        ),
        "source_pool_impostor_style_similarity_stats": stats(
            train_source_pool_style_similarity
        ),
        "source_pool_impostors_per_task": stats(source_pool_counts),
        "margin_vs_source_pool_impostors": mean(source_pool_margins),
        "source_pool_pairwise_win_rate": mean(source_pool_pairwise_win_rates),
        "source_pool_impostor_content_tag_jaccard": mean(source_pool_content_jaccards),
        "source_pool_impostor_length_ratio": mean(source_pool_length_ratios),
        "source_pool_impostor_same_topic_rate": mean(source_pool_same_topic_rates),
        "source_pool_impostor_task_metrics": source_pool_task_metrics,
        "heldout_topic_seen_in_train_rate": mean(heldout_topic_seen),
        "train_topic_count": len(train_topics),
        "heldout_topic_count": len(heldout_topics),
        "heldout_topics_seen_in_train_count": len(seen_heldout_topics),
        "mean_train_heldout_content_tag_jaccard": mean(train_heldout_content_jaccard),
        "word_count": {
            "train": stats(train_word_counts),
            "heldout": stats(heldout_word_counts),
            "hard_negatives": stats(negative_word_counts),
        },
        "word_count_ratio_train_to_heldout": ratio(
            mean(train_word_counts),
            mean(heldout_word_counts),
        ),
        "word_count_ratio_hard_negatives_to_heldout": ratio(
            mean(negative_word_counts),
            mean(heldout_word_counts),
        ),
    }
