"""Metric helpers for cleaned personal author-style artifacts."""

from __future__ import annotations

import math as math
from collections import Counter
from collections import defaultdict as defaultdict
from typing import Any

from auto_skill.cleaning.author_style.quality import metric_helpers as _metric_helpers

TOPIC_MARKERS = _metric_helpers.TOPIC_MARKERS
STYLE_NORMALIZERS = _metric_helpers.STYLE_NORMALIZERS
average_features = _metric_helpers.average_features
content_tag_jaccard = _metric_helpers.content_tag_jaccard
distribution = _metric_helpers.distribution
features_from_text = _metric_helpers.features_from_text
gpt_audit_payload = _metric_helpers.gpt_audit_payload
mean = _metric_helpers.mean
negatives_by_task = _metric_helpers.negatives_by_task
normalize_style_tag = _metric_helpers.normalize_style_tag
numeric = _metric_helpers.numeric
ratio = _metric_helpers.ratio
row_by_pack = _metric_helpers.row_by_pack
source_pool_heldout_rows = _metric_helpers.source_pool_heldout_rows
split_canonical_tags = _metric_helpers.split_canonical_tags
stats = _metric_helpers.stats
style_similarity_from_features = _metric_helpers.style_similarity_from_features
tag_counts = _metric_helpers.tag_counts
task_refs = _metric_helpers.task_refs
top_items = _metric_helpers.top_items


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
