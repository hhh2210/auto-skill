"""Cluster manifests for cleaned personal author-style packs."""

from __future__ import annotations

from typing import Any

from auto_skill.cleaning.author_style.cluster_profile import (
    ARTIFACT_BOUNDARY,
    BACKGROUND_STYLE_TAGS,
    background_style_tags,
    canonical_style_tags,
    deterministic_supported_style_tags,
    family_profile,
    metric_means,
    paper_eligible_style_tags,
    risk_flag_counts,
    row_counts,
    style_families,
    tag_profile,
)
from auto_skill.cleaning.author_style.quality import tag_counts

SCHEMA_VERSION = "author-style-cluster-manifest/v1"
PACK_PROFILE_SCHEMA_VERSION = "author-style-pack-cluster-profile/v1"


def build_pack_cluster_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for row in rows:
        tags = canonical_style_tags(row)
        eligible_tags = paper_eligible_style_tags(tags)
        deterministic_supported_tags = deterministic_supported_style_tags(row, tags)
        deterministic_supported_set = set(deterministic_supported_tags)
        deterministic_metrics = row.get("deterministic_metrics", {})
        hard_negative_metrics = row.get("hard_negative_metrics", {})
        quality_gate = row.get("quality_gate", {})
        if not isinstance(deterministic_metrics, dict):
            deterministic_metrics = {}
        if not isinstance(hard_negative_metrics, dict):
            hard_negative_metrics = {}
        if not isinstance(quality_gate, dict):
            quality_gate = {}
        profiles.append(
            {
                "schema_version": PACK_PROFILE_SCHEMA_VERSION,
                "pack_id": row.get("pack_id"),
                "source": row.get("source"),
                "paper_quality_tier": row.get("paper_quality_tier"),
                "paper_plot_role": row.get("paper_plot_role"),
                "quality_gate_status": quality_gate.get("status"),
                "paper_plot_caveats": row.get("paper_plot_caveats", []),
                "canonical_style_tags": tags,
                "eligible_style_tags": eligible_tags,
                "background_style_tags": background_style_tags(tags),
                "llm_only_style_tags": sorted(
                    tag for tag in eligible_tags if tag not in deterministic_supported_set
                ),
                "deterministic_supported_style_tags": sorted(
                    tag for tag in eligible_tags if tag in deterministic_supported_set
                ),
                "style_families": style_families(tags),
                "deterministic_cluster_tags": row.get("deterministic_cluster_tags", []),
                "raw_gpt_style_tags": row.get("gpt_style_cluster_tags", []),
                "topic_like_exclusions": row.get("topic_like_gpt_tags", []),
                "excluded_topic_like_tags": row.get("topic_like_gpt_tags", []),
                "topic_like_exclusion_count": len(row.get("topic_like_gpt_tags", [])),
                "topic_decoupling": {
                    "train_topic_count": deterministic_metrics.get("train_topic_count"),
                    "heldout_topic_seen_in_train_rate": deterministic_metrics.get(
                        "heldout_topic_seen_in_train_rate"
                    ),
                    "mean_train_heldout_content_tag_jaccard": deterministic_metrics.get(
                        "mean_train_heldout_content_tag_jaccard"
                    ),
                },
                "source_pool": {
                    "margin_vs_source_pool_impostors": deterministic_metrics.get(
                        "margin_vs_source_pool_impostors"
                    ),
                    "source_pool_pairwise_win_rate": deterministic_metrics.get(
                        "source_pool_pairwise_win_rate"
                    ),
                },
                "hard_negatives": {
                    "mean_style_confusability_1_to_5": hard_negative_metrics.get(
                        "mean_style_confusability_1_to_5"
                    ),
                    "selected_count": hard_negative_metrics.get("selected_count"),
                },
                "quality_flags": row.get("quality_flags", []),
                "diagnostic_caveats": row.get("diagnostic_caveats", []),
                "metric_means": metric_means([row]),
                "artifact_boundary": ARTIFACT_BOUNDARY,
            }
        )
    return profiles


def build_cluster_manifest(
    rows: list[dict[str, Any]],
    *,
    dataset_label: str | None = None,
    eval_context: dict[str, Any] | None = None,
    min_cluster_size: int = 5,
    representative_limit: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sorted_rows = sorted(rows, key=lambda row: str(row.get("pack_id") or ""))
    total_packs = len(sorted_rows)
    canonical_tag_lists = [canonical_style_tags(row) for row in sorted_rows]
    canonical_counts = tag_counts(canonical_tag_lists)
    family_counts = tag_counts([style_families(tags) for tags in canonical_tag_lists])
    deterministic_counts = tag_counts(
        [
            row.get("deterministic_cluster_tags", [])
            for row in sorted_rows
            if isinstance(row.get("deterministic_cluster_tags"), list)
        ]
    )
    topic_like_counts = tag_counts(
        [
            row.get("topic_like_gpt_tags", [])
            for row in sorted_rows
            if isinstance(row.get("topic_like_gpt_tags"), list)
        ]
    )
    eligible_tags = [
        tag for tag, count in canonical_counts.items() if count >= min_cluster_size
    ]
    paper_eligible_tags = [
        tag
        for tag, count in canonical_counts.items()
        if count >= min_cluster_size and tag not in BACKGROUND_STYLE_TAGS
    ]
    background_counts = {
        tag: count for tag, count in canonical_counts.items() if tag in BACKGROUND_STYLE_TAGS
    }
    paper_eligible_counts = {
        tag: count
        for tag, count in canonical_counts.items()
        if tag not in BACKGROUND_STYLE_TAGS
    }
    cluster_profiles = [
        tag_profile(
            tag,
            sorted_rows,
            total_packs=total_packs,
            min_cluster_size=min_cluster_size,
            representative_limit=representative_limit,
        )
        for tag in canonical_counts
    ]
    family_profiles = [
        family_profile(
            family,
            sorted_rows,
            total_packs=total_packs,
            representative_limit=representative_limit,
        )
        for family in family_counts
    ]
    packs_with_topic_like = sum(
        1 for row in sorted_rows if row.get("topic_like_gpt_tags")
    )
    packs_with_any_paper_tag = [
        row
        for row in sorted_rows
        if set(paper_eligible_style_tags(canonical_style_tags(row))) & set(paper_eligible_tags)
    ]
    packs_with_two_paper_tags = [
        row
        for row in sorted_rows
        if len(
            set(paper_eligible_style_tags(canonical_style_tags(row)))
            & set(paper_eligible_tags)
        )
        >= 2
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_boundary": ARTIFACT_BOUNDARY,
        "population": {
            "dataset_label": dataset_label,
            "pack_count": total_packs,
            "paper_plot_role_counts": row_counts(sorted_rows, "paper_plot_role"),
            "quality_tier_counts": row_counts(sorted_rows, "paper_quality_tier"),
            "eval_context": eval_context or {},
        },
        "source_pack_schema_versions": sorted(
            {
                str(row.get("schema_version"))
                for row in sorted_rows
                if isinstance(row.get("schema_version"), str)
            }
        ),
        "pack_count": total_packs,
        "cluster_policy": {
            "style_tag_source": "canonical_gpt_style_tags",
            "family_mapping": "deterministic_static_map_v1",
            "topic_like_tags": "excluded_from_cluster_keys_and_reported_as_exclusions",
            "background_style_tags": sorted(BACKGROUND_STYLE_TAGS),
            "min_cluster_size": min_cluster_size,
            "representative_limit": representative_limit,
        },
        "cluster_readiness": {
            "status": "ok" if eligible_tags else "insufficient_clusters",
            "eligible_cluster_count": len(eligible_tags),
            "paper_eligible_cluster_count": len(paper_eligible_tags),
            "background_cluster_count": sum(
                1 for tag in eligible_tags if tag in BACKGROUND_STYLE_TAGS
            ),
            "eligible_pack_coverage": (
                round(
                    len(
                        {
                            row.get("pack_id")
                            for row in sorted_rows
                            if set(canonical_style_tags(row)) & set(eligible_tags)
                        }
                    )
                    / total_packs,
                    3,
                )
                if total_packs
                else None
            ),
            "paper_eligible_pack_coverage": (
                round(len(packs_with_any_paper_tag) / total_packs, 3)
                if total_packs
                else None
            ),
            "packs_with_at_least_2_paper_eligible_style_tags": len(
                packs_with_two_paper_tags
            ),
            "pack_rate_with_at_least_2_paper_eligible_style_tags": (
                round(len(packs_with_two_paper_tags) / total_packs, 3)
                if total_packs
                else None
            ),
        },
        "style_tag_counts": canonical_counts,
        "paper_eligible_style_tag_counts": paper_eligible_counts,
        "background_style_tag_counts": background_counts,
        "style_family_counts": family_counts,
        "deterministic_cluster_tag_counts": deterministic_counts,
        "topic_like_exclusion": {
            "pack_count_with_topic_like_tags": packs_with_topic_like,
            "pack_fraction_with_topic_like_tags": (
                round(packs_with_topic_like / total_packs, 3) if total_packs else None
            ),
            "topic_like_tag_counts": topic_like_counts,
        },
        "excluded_tag_summary": {
            "topic_like_tag_counts": topic_like_counts,
            "background_style_tag_counts": background_counts,
            "exclusion_rules": [
                "drop topic-like tags from cluster membership",
                "treat platform-wide conversational register tags as background",
            ],
        },
        "eligible_cluster_tags": [
            profile
            for profile in cluster_profiles
            if profile["pack_count"] >= min_cluster_size
        ],
        "paper_eligible_cluster_tags": [
            profile
            for profile in cluster_profiles
            if profile["pack_count"] >= min_cluster_size
            and profile["tag"] not in BACKGROUND_STYLE_TAGS
        ],
        "background_cluster_tags": [
            profile
            for profile in cluster_profiles
            if profile["pack_count"] >= min_cluster_size
            and profile["tag"] in BACKGROUND_STYLE_TAGS
        ],
        "small_cluster_tags": [
            profile
            for profile in cluster_profiles
            if profile["pack_count"] < min_cluster_size
        ],
        "family_profiles": family_profiles,
        "aggregate_metric_means": metric_means(sorted_rows),
        "aggregate_quality_flag_counts": risk_flag_counts(sorted_rows, "quality_flags"),
        "aggregate_diagnostic_caveat_counts": risk_flag_counts(
            sorted_rows, "diagnostic_caveats"
        ),
    }
    return manifest, build_pack_cluster_profiles(sorted_rows)
