"""Report builders for cleaned personal author-style artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from auto_skill.cleaning.author_style.quality.audit import (
    assign_quality_gate,
    diagnostic_caveats,
    source_policy,
)
from auto_skill.cleaning.author_style.quality.metrics import (
    deterministic_pack_metrics,
    distribution,
    gpt_audit_payload,
    hard_negative_metrics,
    mean,
    negatives_by_task,
    numeric,
    row_by_pack,
    source_pool_heldout_rows,
    split_canonical_tags,
    stats,
    tag_counts,
    task_refs,
    top_items,
)

SCHEMA_VERSION = "author-style-quality-summary/v3"
PACK_SCHEMA_VERSION = "author-style-pack-quality/v3"
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
