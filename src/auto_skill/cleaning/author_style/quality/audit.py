"""Audit gates for cleaned personal author-style artifacts."""

from __future__ import annotations

from typing import Any

from auto_skill.cleaning.author_style.quality.metrics import numeric

SOURCE_POLICIES = {
    "cross_topic_online_comment_history": "mendeley_cross_topic_v1",
    "personal_blog_history": "blog_temporal_same_context_v1",
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
