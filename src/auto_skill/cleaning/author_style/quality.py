"""Quality summaries for cleaned personal author-style artifacts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "author-style-quality-summary/v3"
PACK_SCHEMA_VERSION = "author-style-pack-quality/v3"

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
DIAGNOSTIC_ONLY_FIELDS = [
    "style_extractability_1_to_5",
    "topic_leakage_risk_1_to_5",
    "model_familiarity_risk_1_to_5",
    "negative_strength_1_to_5",
    "gpt_style_cluster_tags",
    "canonical_gpt_style_tags",
    "topic_like_gpt_tags",
    "mean_style_confusability_1_to_5",
    "mean_topic_shortcut_risk_1_to_5",
    "reject_reasons",
]
METRIC_PROVENANCE = {
    "deterministic_metrics": "private_eval_style_features",
    "hard_negative_metrics": "selected_hard_negative_match_features",
    "llm_diagnostics": "gpt55_author_audit_and_rerank",
}
SOURCE_POLICIES = {
    "cross_topic_online_comment_history": "mendeley_cross_topic_v1",
    "personal_blog_history": "blog_temporal_same_context_v1",
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


def source_policy(pack: dict[str, Any]) -> str:
    source = str(pack.get("source") or "")
    if source in SOURCE_POLICIES:
        return SOURCE_POLICIES[source]
    split_kind = str((pack.get("domain") or {}).get("split_kind") or "")
    domain_kind = str((pack.get("domain") or {}).get("kind") or "")
    if split_kind == "known_unknown_cross_topic":
        return "mendeley_cross_topic_v1"
    if domain_kind == "personal_blog":
        return "blog_temporal_same_context_v1"
    return "generic_author_style_v1"


def min_hard_negative_coverage(task_negative_counts: dict[str, int]) -> int | None:
    if not task_negative_counts:
        return None
    return min(task_negative_counts.values())


def all_hard_negatives_selected(metrics: dict[str, Any]) -> bool:
    count = metrics.get("count")
    selected_count = metrics.get("selected_count")
    if count is None or selected_count is None:
        return False
    statuses = metrics.get("rerank_status_counts")
    return (
        count == selected_count
        and isinstance(statuses, dict)
        and set(statuses) <= {"selected"}
    )


def source_thresholds(policy: str) -> dict[str, float]:
    if policy == "mendeley_cross_topic_v1":
        return {
            "min_hard_negatives_per_task": 4.0,
            "min_style_similarity": 0.5,
            "moderate_style_similarity": 0.7,
            "high_style_similarity": 0.8,
            "max_content_tag_jaccard": 0.2,
            "min_margin_vs_selected_hard_negatives": -0.05,
            "min_source_pool_impostors_per_task": 8.0,
            "min_margin_vs_source_pool_impostors": 0.0,
            "moderate_margin_vs_source_pool_impostors": 0.03,
            "high_margin_vs_source_pool_impostors": 0.08,
            "min_source_pool_pairwise_win_rate": 0.6,
            "min_word_count_ratio": 0.5,
            "max_word_count_ratio": 2.0,
        }
    if policy == "blog_temporal_same_context_v1":
        return {
            "min_hard_negatives_per_task": 4.0,
            "min_style_similarity": 0.45,
            "moderate_style_similarity": 0.45,
            "high_style_similarity": 0.55,
            "max_content_tag_jaccard": 0.2,
            "min_margin_vs_selected_hard_negatives": -0.05,
            "min_source_pool_impostors_per_task": 8.0,
            "min_margin_vs_source_pool_impostors": 0.0,
            "moderate_margin_vs_source_pool_impostors": 0.06,
            "high_margin_vs_source_pool_impostors": 0.12,
            "min_source_pool_pairwise_win_rate": 0.6,
            "min_word_count_ratio": 0.5,
            "max_word_count_ratio": 2.0,
        }
    return {
        "min_hard_negatives_per_task": 4.0,
        "min_style_similarity": 0.5,
        "moderate_style_similarity": 0.7,
        "high_style_similarity": 0.8,
        "max_content_tag_jaccard": 0.2,
        "min_margin_vs_selected_hard_negatives": -0.05,
        "min_source_pool_impostors_per_task": 8.0,
        "min_margin_vs_source_pool_impostors": 0.0,
        "moderate_margin_vs_source_pool_impostors": 0.03,
        "high_margin_vs_source_pool_impostors": 0.08,
        "min_source_pool_pairwise_win_rate": 0.6,
        "min_word_count_ratio": 0.5,
        "max_word_count_ratio": 2.0,
    }


def diagnostic_caveats(
    *,
    quality_flags: list[str],
    policy: str,
    topic_like_tags: list[str],
    deterministic_metrics: dict[str, Any],
) -> list[str]:
    caveats = [
        flag
        for flag in quality_flags
        if flag != "insufficient_hard_negative_coverage"
    ]
    if policy == "blog_temporal_same_context_v1":
        caveats.append("same_train_heldout_topic_by_design")
    if topic_like_tags:
        caveats.append("topic_like_llm_style_tags")

    thresholds = source_thresholds(policy)
    margin = numeric(deterministic_metrics.get("margin_vs_selected_hard_negatives"))
    if margin is not None and margin < thresholds["min_margin_vs_selected_hard_negatives"]:
        caveats.append("selected_negative_more_similar_than_heldout")
    source_pool_margin = numeric(
        deterministic_metrics.get("margin_vs_source_pool_impostors")
    )
    if (
        source_pool_margin is not None
        and source_pool_margin < thresholds["min_margin_vs_source_pool_impostors"]
    ):
        caveats.append("nonpositive_margin_vs_source_pool_impostors")
    elif (
        source_pool_margin is not None
        and source_pool_margin < thresholds["moderate_margin_vs_source_pool_impostors"]
    ):
        caveats.append("weak_margin_vs_source_pool_impostors")
    source_pool_win_rate = numeric(
        deterministic_metrics.get("source_pool_pairwise_win_rate")
    )
    if (
        source_pool_win_rate is not None
        and source_pool_win_rate < thresholds["min_source_pool_pairwise_win_rate"]
    ):
        caveats.append("low_source_pool_pairwise_win_rate")
    content_jaccard = numeric(
        deterministic_metrics.get("mean_train_heldout_content_tag_jaccard")
    )
    if content_jaccard is not None and content_jaccard > thresholds["max_content_tag_jaccard"]:
        caveats.append("high_train_heldout_content_overlap")
    word_ratio = numeric(deterministic_metrics.get("word_count_ratio_train_to_heldout"))
    if word_ratio is not None and (
        word_ratio < thresholds["min_word_count_ratio"]
        or word_ratio > thresholds["max_word_count_ratio"]
    ):
        caveats.append("train_heldout_length_ratio_outlier")
    return sorted(set(caveats))


def assign_quality_gate(
    *,
    pack: dict[str, Any],
    task_negative_counts: dict[str, int],
    hard_negative_metrics_payload: dict[str, Any],
    deterministic_metrics: dict[str, Any],
    caveats: list[str],
) -> dict[str, Any]:
    policy = source_policy(pack)
    thresholds = source_thresholds(policy)
    failures = []
    warnings = []

    coverage = min_hard_negative_coverage(task_negative_counts)
    if coverage is None:
        failures.append("missing_hard_negative_coverage")
    elif coverage < thresholds["min_hard_negatives_per_task"]:
        failures.append("insufficient_hard_negative_coverage")
    if not all_hard_negatives_selected(hard_negative_metrics_payload):
        failures.append("hard_negatives_not_selected_only")
    source_pool_counts = deterministic_metrics.get("source_pool_impostors_per_task")
    if not isinstance(source_pool_counts, dict):
        failures.append("missing_source_pool_impostor_baseline")
    else:
        min_source_pool = numeric(source_pool_counts.get("min"))
        if min_source_pool is None:
            failures.append("missing_source_pool_impostor_baseline")
        elif min_source_pool < thresholds["min_source_pool_impostors_per_task"]:
            failures.append("insufficient_source_pool_impostor_baseline")

    style_similarity = numeric(
        deterministic_metrics.get("mean_train_heldout_style_similarity")
    )
    if style_similarity is None:
        failures.append("missing_train_heldout_style_similarity")
    elif style_similarity < thresholds["min_style_similarity"]:
        failures.append("low_train_heldout_style_similarity")

    topic_seen = numeric(deterministic_metrics.get("heldout_topic_seen_in_train_rate"))
    if policy == "mendeley_cross_topic_v1":
        if topic_seen is None:
            failures.append("missing_cross_topic_split_check")
        elif topic_seen != 0.0:
            failures.append("cross_topic_split_violation")
    elif policy == "blog_temporal_same_context_v1":
        warnings.append("same_context_temporal_split")
    else:
        warnings.append("unknown_source_policy")

    if "high_train_heldout_content_overlap" in caveats:
        warnings.append("high_train_heldout_content_overlap")
    if "selected_negative_more_similar_than_heldout" in caveats:
        warnings.append("selected_negative_more_similar_than_heldout")
    if "train_heldout_length_ratio_outlier" in caveats:
        warnings.append("train_heldout_length_ratio_outlier")
    if "nonpositive_margin_vs_source_pool_impostors" in caveats:
        failures.append("nonpositive_margin_vs_source_pool_impostors")
    if "weak_margin_vs_source_pool_impostors" in caveats:
        warnings.append("weak_margin_vs_source_pool_impostors")
    if "low_source_pool_pairwise_win_rate" in caveats:
        warnings.append("low_source_pool_pairwise_win_rate")

    source_pool_margin = numeric(
        deterministic_metrics.get("margin_vs_source_pool_impostors")
    )
    if policy == "mendeley_cross_topic_v1":
        if (
            style_similarity is not None
            and style_similarity >= thresholds["high_style_similarity"]
            and source_pool_margin is not None
            and source_pool_margin >= thresholds["high_margin_vs_source_pool_impostors"]
        ):
            tier = "cross_topic_high_signal"
        elif (
            style_similarity is not None
            and style_similarity >= thresholds["moderate_style_similarity"]
            and source_pool_margin is not None
            and source_pool_margin >= thresholds["moderate_margin_vs_source_pool_impostors"]
        ):
            tier = "cross_topic_moderate_signal"
        else:
            tier = "cross_topic_low_signal"
        paper_plot_role = (
            "main_cross_topic" if tier != "cross_topic_low_signal" else "debug_only"
        )
        paper_main_candidate = not failures and tier != "cross_topic_low_signal"
    elif policy == "blog_temporal_same_context_v1":
        if (
            style_similarity is not None
            and style_similarity >= thresholds["high_style_similarity"]
            and source_pool_margin is not None
            and source_pool_margin >= thresholds["high_margin_vs_source_pool_impostors"]
        ):
            tier = "temporal_same_context_high_signal"
        elif (
            style_similarity is not None
            and style_similarity >= thresholds["moderate_style_similarity"]
            and source_pool_margin is not None
            and source_pool_margin >= thresholds["moderate_margin_vs_source_pool_impostors"]
        ):
            tier = "temporal_same_context_moderate_signal"
        else:
            tier = "temporal_context_dominated"
        paper_plot_role = (
            "temporal_same_context_analysis"
            if tier != "temporal_context_dominated"
            else "debug_only"
        )
        paper_main_candidate = False
    else:
        tier = "generic_author_style_pool"
        paper_plot_role = "analysis_pool"
        paper_main_candidate = False

    low_signal_tier = tier in {"cross_topic_low_signal", "temporal_context_dominated"}
    if failures:
        status = "fail"
        paper_plot_eligible = False
        paper_plot_role = "debug_only"
        paper_main_candidate = False
    elif warnings or tier.endswith("moderate_signal") or low_signal_tier:
        status = "warn"
        paper_plot_eligible = paper_plot_role != "debug_only"
    else:
        status = "pass"
        paper_plot_eligible = paper_plot_role != "debug_only"

    return {
        "status": status,
        "tier": tier,
        "policy": policy,
        "reasons": sorted(set(failures + warnings)),
        "thresholds": thresholds,
        "paper_plot_eligible": paper_plot_eligible,
        "paper_main_candidate": paper_main_candidate,
        "paper_plot_role": paper_plot_role,
    }


def source_pool_impostor_baseline_payload(
    deterministic_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pool_kind": "accepted_private_heldout_pool",
        "selection_policy": "deterministic_full_source_pool_v1",
        "mean_train_source_pool_impostor_style_similarity": deterministic_metrics.get(
            "mean_train_source_pool_impostor_style_similarity"
        ),
        "source_pool_impostor_style_similarity_stats": deterministic_metrics.get(
            "source_pool_impostor_style_similarity_stats"
        ),
        "source_pool_impostors_per_task": deterministic_metrics.get(
            "source_pool_impostors_per_task"
        ),
        "margin_vs_source_pool_impostors": deterministic_metrics.get(
            "margin_vs_source_pool_impostors"
        ),
        "source_pool_pairwise_win_rate": deterministic_metrics.get(
            "source_pool_pairwise_win_rate"
        ),
        "source_pool_impostor_content_tag_jaccard": deterministic_metrics.get(
            "source_pool_impostor_content_tag_jaccard"
        ),
        "source_pool_impostor_length_ratio": deterministic_metrics.get(
            "source_pool_impostor_length_ratio"
        ),
        "source_pool_impostor_same_topic_rate": deterministic_metrics.get(
            "source_pool_impostor_same_topic_rate"
        ),
    }


def pack_quality_rows(
    *,
    label: str,
    packs: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    private_by_pack = row_by_pack(private_rows)
    audit_by_pack = row_by_pack(audit_rows)
    negatives = negatives_by_task(hard_negative_rows)
    source_pool_heldout = source_pool_heldout_rows(private_rows)
    rows = []
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        private = private_by_pack.get(pack_id, {})
        audit = gpt_audit_payload(audit_by_pack.get(pack_id))
        refs = task_refs(pack)
        task_negative_counts = {ref: len(negatives.get(ref, [])) for ref in refs}
        pack_negatives = [row for ref in refs for row in negatives.get(ref, [])]
        deterministic_metrics = deterministic_pack_metrics(
            private, negatives, source_pool_heldout
        )
        style_extractability = numeric(audit.get("style_extractability_1_to_5"))
        topic_leakage = numeric(audit.get("topic_leakage_risk_1_to_5"))
        model_familiarity = numeric(audit.get("model_familiarity_risk_1_to_5"))
        negative_strength = numeric(audit.get("negative_strength_1_to_5"))
        gpt_tags = [
            str(tag)
            for tag in audit.get("style_cluster_tags", [])
            if isinstance(tag, str) and tag
        ]
        canonical_tags, topic_like_tags = split_canonical_tags(gpt_tags)
        deterministic_tags = [
            str(tag)
            for tag in private.get("cluster_tags", [])
            if isinstance(tag, str) and tag
        ]
        flags = []
        if style_extractability is not None and style_extractability < 4:
            flags.append("low_style_extractability")
        if topic_leakage is not None and topic_leakage >= 4:
            flags.append("high_topic_leakage")
        if model_familiarity is not None and model_familiarity >= 4:
            flags.append("high_model_familiarity")
        if negative_strength is not None and negative_strength < 4:
            flags.append("weak_negative_set")
        if task_negative_counts and min(task_negative_counts.values()) < 4:
            flags.append("insufficient_hard_negative_coverage")
        hard_metrics = hard_negative_metrics(pack_negatives)
        policy = source_policy(pack)
        source_pool_baseline = source_pool_impostor_baseline_payload(
            deterministic_metrics
        )
        caveats = diagnostic_caveats(
            quality_flags=flags,
            policy=policy,
            topic_like_tags=topic_like_tags,
            deterministic_metrics=deterministic_metrics,
        )
        quality_gate = assign_quality_gate(
            pack=pack,
            task_negative_counts=task_negative_counts,
            hard_negative_metrics_payload=hard_metrics,
            deterministic_metrics=deterministic_metrics,
            caveats=caveats,
        )
        rows.append(
            {
                "schema_version": PACK_SCHEMA_VERSION,
                "dataset_label": label,
                "pack_id": pack_id,
                "source": pack.get("source"),
                "domain_kind": (pack.get("domain") or {}).get("kind"),
                "train_examples": len(pack.get("train_examples", [])),
                "heldout_tasks": len(refs),
                "deterministic_cluster_tags": deterministic_tags,
                "gpt_style_cluster_tags": gpt_tags,
                "canonical_gpt_style_tags": canonical_tags,
                "topic_like_gpt_tags": topic_like_tags,
                "style_extractability_1_to_5": style_extractability,
                "topic_leakage_risk_1_to_5": topic_leakage,
                "model_familiarity_risk_1_to_5": model_familiarity,
                "negative_strength_1_to_5": negative_strength,
                "usable": audit.get("usable"),
                "should_use_for_smoke": audit.get("should_use_for_smoke"),
                "reject_reasons": audit.get("reject_reasons", []),
                "hard_negative_coverage_by_task": task_negative_counts,
                "hard_negative_metrics": hard_metrics,
                "source_pool_impostor_baseline": source_pool_baseline,
                "deterministic_metrics": deterministic_metrics,
                "quality_flags": flags,
                "diagnostic_caveats": caveats,
                "quality_gate": quality_gate,
                "paper_quality_tier": quality_gate["tier"],
                "paper_plot_role": quality_gate["paper_plot_role"],
                "paper_plot_eligible": quality_gate["paper_plot_eligible"],
                "paper_main_candidate": quality_gate["paper_main_candidate"],
                "paper_plot_caveats": sorted(set(caveats + quality_gate["reasons"])),
                "metric_provenance": METRIC_PROVENANCE,
                "diagnostic_only_fields": DIAGNOSTIC_ONLY_FIELDS,
            }
        )
    return rows


def summarize_pack_quality(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    style_scores = [
        value
        for row in rows
        if (value := numeric(row.get("style_extractability_1_to_5"))) is not None
    ]
    topic_scores = [
        value
        for row in rows
        if (value := numeric(row.get("topic_leakage_risk_1_to_5"))) is not None
    ]
    familiarity_scores = [
        value
        for row in rows
        if (value := numeric(row.get("model_familiarity_risk_1_to_5"))) is not None
    ]
    negative_scores = [
        value
        for row in rows
        if (value := numeric(row.get("negative_strength_1_to_5"))) is not None
    ]
    confusability_scores = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("hard_negative_metrics", {}).get("mean_style_confusability_1_to_5")
            )
        )
        is not None
    ]
    train_heldout_style = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "mean_train_heldout_style_similarity"
                )
            )
        )
        is not None
    ]
    train_negative_style = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "mean_train_negative_style_similarity"
                )
            )
        )
        is not None
    ]
    style_margins = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "style_similarity_margin_vs_negatives"
                )
            )
        )
        is not None
    ]
    source_pool_style = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "mean_train_source_pool_impostor_style_similarity"
                )
            )
        )
        is not None
    ]
    source_pool_margins = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "margin_vs_source_pool_impostors"
                )
            )
        )
        is not None
    ]
    source_pool_win_rates = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "source_pool_pairwise_win_rate"
                )
            )
        )
        is not None
    ]
    source_pool_counts = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {})
                .get("source_pool_impostors_per_task", {})
                .get("min")
            )
        )
        is not None
    ]
    heldout_topic_seen_rates = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "heldout_topic_seen_in_train_rate"
                )
            )
        )
        is not None
    ]
    content_tag_jaccards = [
        value
        for row in rows
        if (
            value := numeric(
                row.get("deterministic_metrics", {}).get(
                    "mean_train_heldout_content_tag_jaccard"
                )
            )
        )
        is not None
    ]
    coverage_values = []
    for row in rows:
        coverage = row.get("hard_negative_coverage_by_task")
        if isinstance(coverage, dict):
            coverage_values.extend(
                int(value) for value in coverage.values() if isinstance(value, int)
            )
    deterministic_tag_lists = [
        row.get("deterministic_cluster_tags", [])
        for row in rows
        if isinstance(row.get("deterministic_cluster_tags"), list)
    ]
    gpt_tag_lists = [
        row.get("gpt_style_cluster_tags", [])
        for row in rows
        if isinstance(row.get("gpt_style_cluster_tags"), list)
    ]
    canonical_tag_lists = [
        row.get("canonical_gpt_style_tags", [])
        for row in rows
        if isinstance(row.get("canonical_gpt_style_tags"), list)
    ]
    topic_like_tag_lists = [
        row.get("topic_like_gpt_tags", [])
        for row in rows
        if isinstance(row.get("topic_like_gpt_tags"), list)
    ]
    quality_flag_counts = Counter(
        str(flag)
        for row in rows
        for flag in row.get("quality_flags", [])
        if isinstance(flag, str)
    )
    diagnostic_caveat_counts = Counter(
        str(caveat)
        for row in rows
        for caveat in row.get("diagnostic_caveats", [])
        if isinstance(caveat, str)
    )
    quality_gate_status_counts = Counter(
        str((row.get("quality_gate") or {}).get("status") or "")
        for row in rows
    )
    quality_gate_policy_counts = Counter(
        str((row.get("quality_gate") or {}).get("policy") or "")
        for row in rows
    )
    paper_quality_tier_counts = Counter(
        str(row.get("paper_quality_tier") or "") for row in rows
    )
    paper_plot_role_counts = Counter(
        str(row.get("paper_plot_role") or "") for row in rows
    )
    top_combo_counts = Counter(
        "+".join(sorted(row.get("gpt_style_cluster_tags", []))[:3])
        for row in rows
        if row.get("gpt_style_cluster_tags")
    )
    return {
        "label": label,
        "packs": len(rows),
        "source_counts": top_items(Counter(str(row.get("source") or "") for row in rows)),
        "domain_kind_counts": top_items(
            Counter(str(row.get("domain_kind") or "") for row in rows)
        ),
        "train_examples_per_pack": top_items(
            Counter(str(row.get("train_examples")) for row in rows)
        ),
        "heldout_tasks_per_pack": top_items(
            Counter(str(row.get("heldout_tasks")) for row in rows)
        ),
        "mean_style_extractability_1_to_5": mean(style_scores),
        "style_extractability_distribution": distribution(style_scores),
        "mean_topic_leakage_risk_1_to_5": mean(topic_scores),
        "topic_leakage_distribution": distribution(topic_scores),
        "mean_model_familiarity_risk_1_to_5": mean(familiarity_scores),
        "model_familiarity_distribution": distribution(familiarity_scores),
        "mean_negative_strength_1_to_5": mean(negative_scores),
        "negative_strength_distribution": distribution(negative_scores),
        "mean_hard_negative_style_confusability_1_to_5": mean(confusability_scores),
        "deterministic_style_stability": {
            "mean_train_heldout_style_similarity": mean(train_heldout_style),
            "mean_train_negative_style_similarity": mean(train_negative_style),
            "mean_style_similarity_margin_vs_negatives": mean(style_margins),
        },
        "source_pool_impostor_baseline": {
            "mean_train_source_pool_impostor_style_similarity": mean(source_pool_style),
            "mean_margin_vs_source_pool_impostors": mean(source_pool_margins),
            "mean_source_pool_pairwise_win_rate": mean(source_pool_win_rates),
            "source_pool_impostors_per_task_min": stats(source_pool_counts),
        },
        "deterministic_topic_decoupling": {
            "heldout_topic_seen_in_train_rate": mean(heldout_topic_seen_rates),
            "mean_train_heldout_content_tag_jaccard": mean(content_tag_jaccards),
        },
        "hard_negative_coverage": {
            "min": min(coverage_values) if coverage_values else None,
            "mean": mean([float(value) for value in coverage_values]),
            "max": max(coverage_values) if coverage_values else None,
        },
        "deterministic_cluster_tag_counts": tag_counts(deterministic_tag_lists),
        "gpt_style_cluster_tag_counts": tag_counts(gpt_tag_lists),
        "canonical_gpt_style_tag_counts": tag_counts(canonical_tag_lists),
        "topic_like_gpt_tag_counts": tag_counts(topic_like_tag_lists),
        "gpt_style_tag_combo_counts": top_items(top_combo_counts),
        "style_tag_vocabulary_size": len(tag_counts(gpt_tag_lists)),
        "canonical_style_tag_vocabulary_size": len(tag_counts(canonical_tag_lists)),
        "mean_gpt_style_tags_per_pack": mean([float(len(tags)) for tags in gpt_tag_lists]),
        "mean_canonical_style_tags_per_pack": mean(
            [float(len(tags)) for tags in canonical_tag_lists]
        ),
        "quality_flag_counts": dict(quality_flag_counts),
        "diagnostic_caveat_counts": dict(diagnostic_caveat_counts),
        "quality_gate_status_counts": dict(quality_gate_status_counts),
        "quality_gate_policy_counts": dict(quality_gate_policy_counts),
        "paper_quality_tier_counts": dict(paper_quality_tier_counts),
        "paper_plot_role_counts": dict(paper_plot_role_counts),
        "paper_plot_eligible_count": sum(
            1 for row in rows if row.get("paper_plot_eligible") is True
        ),
        "paper_main_candidate_count": sum(
            1 for row in rows if row.get("paper_main_candidate") is True
        ),
        "pack_ids_with_flags": [
            row["pack_id"] for row in rows if row.get("quality_flags")
        ][:50],
    }


def summarize_quality_datasets(datasets: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    dataset_summaries = [
        summarize_pack_quality(label, rows)
        for label, rows in datasets
    ]
    all_rows = [row for _, rows in datasets for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "datasets": dataset_summaries,
        "aggregate": summarize_pack_quality("all", all_rows),
    }


def load_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], ...]:
    from auto_skill.cleaning.packs import load_jsonl

    return (
        load_jsonl(run_dir / "accepted_author_style_packs.jsonl"),
        load_jsonl(run_dir / "accepted_author_style_private_eval.jsonl"),
        load_jsonl(run_dir / "gpt55_author_audits.jsonl"),
        load_jsonl(run_dir / "accepted_hard_negatives.jsonl"),
    )
