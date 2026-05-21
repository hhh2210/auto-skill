"""Cluster profile helpers for author-style manifests."""

from __future__ import annotations

from collections import Counter
from typing import Any

from auto_skill.cleaning.author_style.quality import mean, numeric, tag_counts, top_items

BACKGROUND_STYLE_TAGS = {
    "casual_conversational",
    "conversational_contractions",
}
ARTIFACT_BOUNDARY = {
    "must_not_use_for_induction": True,
    "may_use_for": [
        "paper_figure",
        "post_hoc_eval_stratification",
        "subset_sampling_after_freeze",
    ],
    "must_not_use_for": [
        "skill_prompt",
        "heldout_generation",
        "judge_prompt",
        "hard_negative_rerank",
    ],
    "intended_use": "post_hoc_dataset_characterization_and_eval_stratification",
}

STYLE_FAMILY_BY_TAG = {
    "abrupt_topic_shifts": "pacing_punctuation",
    "caps_emphasis": "orthography_emphasis",
    "casual_conversational": "register_voice",
    "conversational_contractions": "register_voice",
    "direct_address": "interaction_stance",
    "edit_self_correction": "interaction_stance",
    "ellipsis_heavy": "pacing_punctuation",
    "emoticon_laughter": "orthography_emphasis",
    "exclamation_heavy": "pacing_punctuation",
    "first_person_diary": "narrative_persona",
    "hedging": "interaction_stance",
    "long_run_on": "pacing_punctuation",
    "lowercase_i": "orthography_emphasis",
    "nonstandard_orthography": "orthography_emphasis",
    "parenthetical_asides": "pacing_punctuation",
    "profanity": "register_voice",
    "question_heavy": "pacing_punctuation",
    "quote_response": "interaction_stance",
    "sarcasm_snark": "register_voice",
    "short_sentences": "pacing_punctuation",
    "supportive_polite": "interaction_stance",
}
DETERMINISTIC_SUPPORT_TAGS = {
    "conversational_contractions": ("conversational_contractions",),
    "ellipsis_heavy": ("ellipsis_heavy",),
    "exclamation_heavy": ("exclamation_heavy",),
    "long_run_on": ("long_sentence",),
    "parenthetical_asides": ("parenthetical",),
    "question_heavy": ("question_heavy",),
    "short_sentences": ("short_sentence",),
}

METRIC_PATHS = {
    "style_extractability_1_to_5": ("style_extractability_1_to_5",),
    "topic_leakage_risk_1_to_5": ("topic_leakage_risk_1_to_5",),
    "model_familiarity_risk_1_to_5": ("model_familiarity_risk_1_to_5",),
    "negative_strength_1_to_5": ("negative_strength_1_to_5",),
    "hard_negative_style_confusability_1_to_5": (
        "hard_negative_metrics",
        "mean_style_confusability_1_to_5",
    ),
    "hard_negative_topic_shortcut_risk_1_to_5": (
        "hard_negative_metrics",
        "mean_topic_shortcut_risk_1_to_5",
    ),
    "train_heldout_style_similarity": (
        "deterministic_metrics",
        "mean_train_heldout_style_similarity",
    ),
    "train_negative_style_similarity": (
        "deterministic_metrics",
        "mean_train_negative_style_similarity",
    ),
    "style_similarity_margin_vs_selected_negatives": (
        "deterministic_metrics",
        "style_similarity_margin_vs_negatives",
    ),
    "source_pool_margin": (
        "deterministic_metrics",
        "margin_vs_source_pool_impostors",
    ),
    "source_pool_pairwise_win_rate": (
        "deterministic_metrics",
        "source_pool_pairwise_win_rate",
    ),
    "heldout_topic_seen_in_train_rate": (
        "deterministic_metrics",
        "heldout_topic_seen_in_train_rate",
    ),
    "train_heldout_content_tag_jaccard": (
        "deterministic_metrics",
        "mean_train_heldout_content_tag_jaccard",
    ),
}


def nested_get(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def canonical_style_tags(row: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(tag)
            for tag in row.get("canonical_gpt_style_tags", [])
            if isinstance(tag, str) and tag
        }
    )


def style_families(tags: list[str]) -> list[str]:
    return sorted({STYLE_FAMILY_BY_TAG.get(tag, "other_style") for tag in tags})


def paper_eligible_style_tags(tags: list[str]) -> list[str]:
    return sorted(tag for tag in tags if tag not in BACKGROUND_STYLE_TAGS)


def background_style_tags(tags: list[str]) -> list[str]:
    return sorted(tag for tag in tags if tag in BACKGROUND_STYLE_TAGS)


def deterministic_supported_style_tags(row: dict[str, Any], tags: list[str]) -> list[str]:
    deterministic_tags = {
        str(tag)
        for tag in row.get("deterministic_cluster_tags", [])
        if isinstance(tag, str) and tag
    }
    supported = []
    for tag in tags:
        support_tags = DETERMINISTIC_SUPPORT_TAGS.get(tag, ())
        if deterministic_tags & set(support_tags):
            supported.append(tag)
    return sorted(supported)


def metric_means(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name, path in METRIC_PATHS.items():
        values = [
            value
            for row in rows
            if (value := numeric(nested_get(row, path))) is not None
        ]
        output[name] = mean(values)
    return output


def risk_flag_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get(field, [])
        if isinstance(values, list):
            counts.update(str(value) for value in values if isinstance(value, str) and value)
    return dict(counts.most_common())


def row_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return top_items(Counter(str(row.get(field) or "") for row in rows))


def representative_pack_ids(
    rows: list[dict[str, Any]], *, limit: int
) -> list[str]:
    def score(row: dict[str, Any]) -> tuple[float, float, float, str]:
        return (
            numeric(row.get("style_extractability_1_to_5")) or 0.0,
            numeric(
                row.get("deterministic_metrics", {}).get("margin_vs_source_pool_impostors")
            )
            or 0.0,
            numeric(
                row.get("hard_negative_metrics", {}).get(
                    "mean_style_confusability_1_to_5"
                )
            )
            or 0.0,
            str(row.get("pack_id") or ""),
        )

    sorted_rows = sorted(rows, key=score, reverse=True)
    return [
        str(row.get("pack_id"))
        for row in sorted_rows[:limit]
        if isinstance(row.get("pack_id"), str) and row.get("pack_id")
    ]


def tag_profile(
    tag: str,
    rows: list[dict[str, Any]],
    *,
    total_packs: int,
    min_cluster_size: int,
    representative_limit: int,
) -> dict[str, Any]:
    member_rows = [row for row in rows if tag in canonical_style_tags(row)]
    support_tags = DETERMINISTIC_SUPPORT_TAGS.get(tag, ())
    deterministic_support_count = sum(
        1
        for row in member_rows
        if set(row.get("deterministic_cluster_tags", [])) & set(support_tags)
    )
    co_tags = Counter(
        co_tag
        for row in member_rows
        for co_tag in canonical_style_tags(row)
        if co_tag != tag
    )
    deterministic_tags = Counter(
        str(value)
        for row in member_rows
        for value in row.get("deterministic_cluster_tags", [])
        if isinstance(value, str) and value
    )
    topic_like_tags = Counter(
        str(value)
        for row in member_rows
        for value in row.get("topic_like_gpt_tags", [])
        if isinstance(value, str) and value
    )
    is_background = tag in BACKGROUND_STYLE_TAGS
    paper_eligible = len(member_rows) >= min_cluster_size and not is_background
    cluster_kind = (
        "background_register"
        if is_background
        else "deterministic_plus_llm"
        if support_tags
        else "llm_diagnostic_only"
    )
    return {
        "cluster_id": tag,
        "tag": tag,
        "display_label": tag.replace("_", " "),
        "family": STYLE_FAMILY_BY_TAG.get(tag, "other_style"),
        "paper_eligible": paper_eligible,
        "background_style": is_background,
        "cluster_kind": cluster_kind,
        "included_canonical_tags": [tag],
        "supporting_deterministic_tags": list(support_tags),
        "excluded_topic_like_tags": sorted(topic_like_tags),
        "pack_count": len(member_rows),
        "pack_fraction": round(len(member_rows) / total_packs, 3) if total_packs else None,
        "deterministic_support_pack_count": deterministic_support_count,
        "llm_only_pack_count": len(member_rows) - deterministic_support_count,
        "pack_ids": [
            str(row.get("pack_id"))
            for row in member_rows
            if isinstance(row.get("pack_id"), str) and row.get("pack_id")
        ],
        "representative_pack_ids": representative_pack_ids(
            member_rows, limit=representative_limit
        ),
        "co_style_tag_counts": dict(co_tags.most_common(12)),
        "deterministic_tag_counts": dict(deterministic_tags.most_common(12)),
        "topic_like_exclusion_counts": dict(topic_like_tags.most_common(12)),
        "source_counts": row_counts(member_rows, "source"),
        "paper_quality_tier_counts": row_counts(member_rows, "paper_quality_tier"),
        "quality_tier_counts": row_counts(member_rows, "paper_quality_tier"),
        "paper_plot_role_counts": row_counts(member_rows, "paper_plot_role"),
        "quality_flag_counts": risk_flag_counts(member_rows, "quality_flags"),
        "diagnostic_caveat_counts": risk_flag_counts(member_rows, "diagnostic_caveats"),
        "metric_means": metric_means(member_rows),
    }


def family_profile(
    family: str,
    rows: list[dict[str, Any]],
    *,
    total_packs: int,
    representative_limit: int,
) -> dict[str, Any]:
    member_rows = [
        row
        for row in rows
        if family in style_families(canonical_style_tags(row))
    ]
    tags = Counter(
        tag
        for row in member_rows
        for tag in canonical_style_tags(row)
        if STYLE_FAMILY_BY_TAG.get(tag, "other_style") == family
    )
    return {
        "family": family,
        "pack_count": len(member_rows),
        "pack_fraction": round(len(member_rows) / total_packs, 3) if total_packs else None,
        "style_tag_counts": dict(tags.most_common()),
        "representative_pack_ids": representative_pack_ids(
            member_rows, limit=representative_limit
        ),
        "paper_quality_tier_counts": row_counts(member_rows, "paper_quality_tier"),
        "quality_flag_counts": risk_flag_counts(member_rows, "quality_flags"),
        "topic_like_exclusion_counts": tag_counts(
            [
                row.get("topic_like_gpt_tags", [])
                for row in member_rows
                if isinstance(row.get("topic_like_gpt_tags"), list)
            ]
        ),
        "metric_means": metric_means(member_rows),
    }
