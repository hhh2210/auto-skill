"""Post-hoc diagnostics for author-style skill compression failures."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from auto_skill.cleaning.author_style.eval_summary import (
    latest_eval_cells,
    mean,
)
from auto_skill.diagnostics.compression_helpers import (
    TAG_PATTERNS,
    canonical_tags_from_judge_tags,
    canonical_tags_in_text,
    compact_counts,
    generation_shape_stats,
    judge_tags,
    list_field,
    loss_bucket,
    model_inventory,
    normalize_tag,
    pack_id,
    pack_profile_index,
    profile_tags,
    row_index,
    row_topic_risk,
    score_snapshot,
    sign_counts,
    skill_index,
    skill_mode_for_eval_mode,
    skill_texts,
)

__all__ = [
    "ARTIFACT_BOUNDARY",
    "SCHEMA_VERSION",
    "TAG_PATTERNS",
    "aggregate_clusters",
    "aggregate_mode",
    "canonical_tags_from_judge_tags",
    "canonical_tags_in_text",
    "compact_counts",
    "coverage_summary",
    "diagnostic_row",
    "generation_shape_stats",
    "judge_tags",
    "list_field",
    "loss_bucket",
    "model_inventory",
    "normalize_tag",
    "pack_id",
    "pack_profile_index",
    "profile_tags",
    "row_index",
    "row_topic_risk",
    "skill_index",
    "skill_mode_for_eval_mode",
    "skill_texts",
    "summarize_author_style_compression_failures",
]

SCHEMA_VERSION = "author-style-compression-failure-diagnostic/v1"
ARTIFACT_BOUNDARY = {
    "must_not_use_for_induction": True,
    "may_use_for": [
        "post_hoc_skill_compression_debug",
        "paper_failure_analysis",
        "prompt_pipeline_redesign",
    ],
    "must_not_use_for": [
        "skill_prompt",
        "skill_induction_input",
        "heldout_generation",
        "judge_prompt",
        "hard_negative_rerank",
        "future_subset_selection_by_outcome",
    ],
    "intended_use": "diagnose what current skill compression loses relative to examples-only",
}



def diagnostic_row(
    *,
    prompt: dict[str, Any] | None,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    profile: dict[str, Any],
    skill_row: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    scores = score_snapshot(prompt, baseline, candidate)
    baseline_score = scores["baseline_score"]
    candidate_score = scores["candidate_score"]
    prompt_score = scores["prompt_score"]
    baseline_win = scores["baseline_win"]
    candidate_win = scores["candidate_win"]
    prompt_win = scores["prompt_win"]
    style_delta = (
        round(candidate_score - baseline_score, 3)
        if baseline_score is not None and candidate_score is not None
        else None
    )
    win_delta = (
        round(candidate_win - baseline_win, 3)
        if baseline_win is not None and candidate_win is not None
        else None
    )
    example_signal = (
        round(baseline_score - prompt_score, 3)
        if baseline_score is not None and prompt_score is not None
        else None
    )
    compression_loss = (
        round(baseline_score - candidate_score, 3)
        if baseline_score is not None and candidate_score is not None
        else None
    )
    retention_ratio = None
    if (
        baseline_score is not None
        and candidate_score is not None
        and prompt_score is not None
        and baseline_score > prompt_score
    ):
        retention_ratio = round(
            (candidate_score - prompt_score) / (baseline_score - prompt_score),
            3,
        )
    win_example_signal = (
        round(baseline_win - prompt_win, 3)
        if baseline_win is not None and prompt_win is not None
        else None
    )
    canonical = profile_tags(profile, "canonical_style_tags")
    eligible = profile_tags(profile, "eligible_style_tags")
    baseline_matched = judge_tags(baseline, "style_tags_matched")
    baseline_missed = judge_tags(baseline, "style_tags_missed")
    candidate_matched = judge_tags(candidate, "style_tags_matched")
    candidate_missed = judge_tags(candidate, "style_tags_missed")
    candidate_missed_canonical = canonical_tags_from_judge_tags(candidate_missed, canonical)
    baseline_matched_canonical = canonical_tags_from_judge_tags(baseline_matched, canonical)
    texts = skill_texts(skill_row)
    skill_hits = canonical_tags_in_text(texts["skill_md"], canonical)
    pipeline_hits = canonical_tags_in_text(texts["feature_pipeline"], canonical)
    return {
        "pack_id": str(candidate.get("pack_id")),
        "task_id": str(candidate.get("task_id")),
        "mode": mode,
        "skill_row_mode": skill_mode_for_eval_mode(mode),
        "style_likeness_delta_vs_examples": style_delta,
        "hard_negative_win_rate_delta_vs_examples": win_delta,
        "example_signal_vs_prompt_only": example_signal,
        "compression_loss_vs_examples": compression_loss,
        "compression_retention_ratio_vs_prompt_only": retention_ratio,
        "hard_negative_win_rate_signal_vs_prompt_only": win_example_signal,
        "loss_bucket": loss_bucket(style_delta),
        "prompt_only_style_likeness": prompt_score,
        "baseline_style_likeness": baseline_score,
        "candidate_style_likeness": candidate_score,
        "prompt_only_hard_negative_win_rate": prompt_win,
        "baseline_hard_negative_win_rate": baseline_win,
        "candidate_hard_negative_win_rate": candidate_win,
        "topic_shortcut_risk_delta_vs_examples": (
            round(row_topic_risk(candidate) - row_topic_risk(baseline), 3)
            if row_topic_risk(candidate) is not None and row_topic_risk(baseline) is not None
            else None
        ),
        "generation_shape": generation_shape_stats(candidate),
        "paper_quality_tier": profile.get("paper_quality_tier"),
        "paper_plot_role": profile.get("paper_plot_role"),
        "eligible_style_tags": eligible,
        "canonical_style_tags": canonical,
        "style_families": profile_tags(profile, "style_families"),
        "topic_like_exclusions": profile_tags(profile, "excluded_topic_like_tags"),
        "candidate_missed_tags": candidate_missed,
        "candidate_matched_tags": candidate_matched,
        "baseline_matched_tags": baseline_matched,
        "baseline_missed_tags": baseline_missed,
        "candidate_missed_canonical_tags": sorted(candidate_missed_canonical),
        "baseline_matched_canonical_tags": sorted(baseline_matched_canonical),
        "baseline_matched_but_candidate_missed_canonical_tags": sorted(
            baseline_matched_canonical & candidate_missed_canonical
        ),
        "skill_md_canonical_hits": sorted(skill_hits),
        "feature_pipeline_canonical_hits": sorted(pipeline_hits),
        "canonical_tags_missing_from_skill_md": sorted(set(canonical) - skill_hits),
        "feature_pipeline_hits_not_compiled_to_skill_md": sorted(pipeline_hits - skill_hits),
        "baseline_matched_canonical_missing_from_skill_md": sorted(
            baseline_matched_canonical - skill_hits
        ),
        "hard_negative_confusability": (
            profile.get("hard_negatives", {}).get("mean_style_confusability_1_to_5")
            if isinstance(profile.get("hard_negatives"), dict)
            else None
        ),
        "source_pool_margin": (
            profile.get("source_pool", {}).get("margin_vs_source_pool_impostors")
            if isinstance(profile.get("source_pool"), dict)
            else None
        ),
    }


def aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    style_deltas = [
        float(row["style_likeness_delta_vs_examples"])
        for row in rows
        if isinstance(row.get("style_likeness_delta_vs_examples"), (int, float))
    ]
    win_deltas = [
        float(row["hard_negative_win_rate_delta_vs_examples"])
        for row in rows
        if isinstance(row.get("hard_negative_win_rate_delta_vs_examples"), (int, float))
    ]
    retention = [
        float(row["compression_retention_ratio_vs_prompt_only"])
        for row in rows
        if isinstance(row.get("compression_retention_ratio_vs_prompt_only"), (int, float))
    ]
    compression_losses = [
        float(row["compression_loss_vs_examples"])
        for row in rows
        if isinstance(row.get("compression_loss_vs_examples"), (int, float))
    ]
    loss_rows = [row for row in rows if str(row.get("loss_bucket", "")).endswith("loss")]
    missed = Counter()
    missed_canonical = Counter()
    lost_canonical = Counter()
    missing_skill = Counter()
    pipeline_not_compiled = Counter()
    for row in loss_rows:
        missed.update(row.get("candidate_missed_tags", []))
        missed_canonical.update(row.get("candidate_missed_canonical_tags", []))
        lost_canonical.update(row.get("baseline_matched_but_candidate_missed_canonical_tags", []))
        missing_skill.update(row.get("baseline_matched_canonical_missing_from_skill_md", []))
        pipeline_not_compiled.update(row.get("feature_pipeline_hits_not_compiled_to_skill_md", []))
    return {
        "paired_rows": len(rows),
        "mean_style_likeness_delta": mean(style_deltas),
        "mean_hard_negative_win_rate_delta": mean(win_deltas),
        "mean_compression_loss_vs_examples": mean(compression_losses),
        "mean_compression_retention_ratio_vs_prompt_only": mean(retention),
        "style_delta_sign_counts": sign_counts(style_deltas),
        "loss_bucket_counts": dict(Counter(str(row.get("loss_bucket")) for row in rows)),
        "loss_rows": len(loss_rows),
        "top_candidate_missed_tags_on_losses": compact_counts(missed),
        "top_candidate_missed_canonical_tags_on_losses": compact_counts(missed_canonical),
        "top_baseline_matched_but_candidate_missed_canonical_tags": compact_counts(lost_canonical),
        "top_baseline_matched_canonical_missing_from_skill_md": compact_counts(missing_skill),
        "top_feature_pipeline_hits_not_compiled_to_skill_md": compact_counts(pipeline_not_compiled),
    }


def aggregate_clusters(rows: list[dict[str, Any]], *, tag_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for tag in row.get(tag_field, []):
            if isinstance(tag, str) and tag:
                groups[tag].append(row)
    summaries = []
    for tag, tag_rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        mode_summary = aggregate_mode(tag_rows)
        mode_summary["tag"] = tag
        mode_summary["pack_count"] = len({str(row.get("pack_id")) for row in tag_rows})
        summaries.append(mode_summary)
    return summaries


def coverage_summary(
    *,
    eval_rows: list[dict[str, Any]],
    latest_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    pack_profiles: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    prompt_mode: str,
    baseline_mode: str,
) -> dict[str, Any]:
    profile_ids = {value for row in pack_profiles if (value := pack_id(row))}
    eval_ids = {value for row in latest_rows if (value := pack_id(row))}
    diagnostic_ids = {str(row.get("pack_id")) for row in diagnostics}
    return {
        "eval_raw_rows": len(eval_rows),
        "eval_latest_rows": len(latest_rows),
        "duplicate_eval_cells_dropped": max(0, len(eval_rows) - len(latest_rows)),
        "skill_rows": len(skill_rows),
        "pack_profile_rows": len(pack_profiles),
        "diagnostic_rows": len(diagnostics),
        "diagnostic_pack_count": len(diagnostic_ids),
        "prompt_mode": prompt_mode,
        "baseline_mode": baseline_mode,
        "missing_profile_packs_in_eval": sorted(profile_ids - eval_ids),
        "eval_packs_missing_profile": sorted(eval_ids - profile_ids),
        "profile_packs_missing_diagnostic_pair": sorted(profile_ids - diagnostic_ids),
    }


def summarize_author_style_compression_failures(
    *,
    eval_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    pack_profiles: list[dict[str, Any]],
    prompt_mode: str = "prompt_only",
    baseline_mode: str = "few_shot_examples_only",
    treatment_modes: list[str] | None = None,
) -> dict[str, Any]:
    latest_rows = latest_eval_cells(eval_rows)
    eval_index = row_index(eval_rows)
    profile_index = pack_profile_index(pack_profiles)
    skills = skill_index(skill_rows)
    modes = sorted({mode for _, _, mode in eval_index})
    treatments = treatment_modes or [mode for mode in modes if mode != baseline_mode]
    baseline_cells = [
        (pack, task, row)
        for (pack, task, mode), row in eval_index.items()
        if mode in {baseline_mode}
    ]

    diagnostics = []
    missing_pairs = Counter()
    for pack, task, baseline in baseline_cells:
        profile = profile_index.get(pack)
        if profile is None:
            missing_pairs["missing_profile"] += 1
            continue
        prompt = eval_index.get((pack, task, prompt_mode))
        if prompt is None:
            missing_pairs[f"missing_prompt::{prompt_mode}"] += 1
        for mode in treatments:
            candidate = eval_index.get((pack, task, mode))
            if candidate is None:
                missing_pairs[f"missing_candidate::{mode}"] += 1
                continue
            skill_mode = skill_mode_for_eval_mode(mode)
            skill_row = skills.get((pack, skill_mode)) if skill_mode else None
            if skill_mode and skill_row is None:
                missing_pairs[f"missing_skill::{skill_mode}"] += 1
            diagnostics.append(
                diagnostic_row(
                    prompt=prompt,
                    baseline=baseline,
                    candidate=candidate,
                    profile=profile,
                    skill_row=skill_row,
                    mode=mode,
                )
            )

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        by_mode[str(row.get("mode"))].append(row)

    modes_summary = {}
    for mode, rows in sorted(by_mode.items()):
        modes_summary[mode] = {
            **aggregate_mode(rows),
            "paper_eligible_style_clusters": aggregate_clusters(
                rows,
                tag_field="eligible_style_tags",
            ),
            "style_families": aggregate_clusters(rows, tag_field="style_families"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_boundary": ARTIFACT_BOUNDARY,
        "comparison": {
            "prompt_mode": prompt_mode,
            "baseline_mode": baseline_mode,
            "treatment_modes": treatments,
            "delta_direction": "treatment - baseline",
        },
        "rows": len(diagnostics),
        "coverage": coverage_summary(
            eval_rows=eval_rows,
            latest_rows=latest_rows,
            skill_rows=skill_rows,
            pack_profiles=pack_profiles,
            diagnostics=diagnostics,
            prompt_mode=prompt_mode,
            baseline_mode=baseline_mode,
        ),
        "status_counts": dict(Counter(str(row.get("status") or "unknown") for row in latest_rows)),
        "model_inventory": model_inventory(latest_rows),
        "missing_pairs": dict(missing_pairs),
        "modes": modes_summary,
        "per_pack": diagnostics,
        "notes": [
            (
                "This diagnostic intentionally excludes raw candidate output "
                "and private heldout references."
            ),
            (
                "Judge tags are free-form and deterministic canonical mapping is approximate; "
                "use top tags as failure hypotheses, not ground-truth labels."
            ),
            "Cluster summaries are multi-label: use per-pack rows for single-count claims.",
        ],
    }
