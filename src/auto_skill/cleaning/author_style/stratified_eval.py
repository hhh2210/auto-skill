"""Stratified summaries for author-style heldout eval rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from auto_skill.cleaning.author_style.eval_summary import (
    latest_eval_cells,
    mean,
    row_score,
    row_win_rate,
    sign,
)

SCHEMA_VERSION = "author-style-stratified-eval-summary/v1"
ARTIFACT_BOUNDARY = {
    "must_not_use_for_induction": True,
    "may_use_for": [
        "paper_figure",
        "post_hoc_eval_stratification",
    ],
    "must_not_use_for": [
        "skill_prompt",
        "skill_induction_input",
        "heldout_generation",
        "judge_prompt",
        "hard_negative_rerank",
        "future_subset_selection_by_outcome",
    ],
    "intended_use": "post_hoc_eval_stratification",
}


def pack_id(row: dict[str, Any]) -> str | None:
    value = row.get("pack_id")
    return value if isinstance(value, str) and value else None


def mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row.get("status") == "success"]
    scores = [score for row in success_rows if (score := row_score(row)) is not None]
    win_rates = [
        rate for row in success_rows if (rate := row_win_rate(row)) is not None
    ]
    return {
        "rows": len(rows),
        "success": len(success_rows),
        "status_counts": dict(
            Counter(str(row.get("status") or "unknown") for row in rows)
        ),
        "mean_style_likeness": mean(scores),
        "hard_negative_win_rate": mean(win_rates),
    }


def paired_delta_summary(
    rows: list[dict[str, Any]], *, baseline_mode: str
) -> dict[str, Any]:
    success_index = {
        (str(row.get("pack_id")), str(row.get("task_id")), str(row.get("mode"))): row
        for row in rows
        if row.get("status") == "success"
    }
    modes = sorted({str(row.get("mode")) for row in rows if row.get("mode")})
    baseline_cells = [
        (pack, task, row)
        for (pack, task, mode), row in success_index.items()
        if mode == baseline_mode
    ]
    deltas = {}
    for mode in modes:
        if mode == baseline_mode:
            continue
        style_deltas = []
        win_rate_deltas = []
        signs = Counter()
        missing_pairs = 0
        for pack, task, baseline in baseline_cells:
            candidate = success_index.get((pack, task, mode))
            if candidate is None:
                missing_pairs += 1
                continue
            baseline_score = row_score(baseline)
            candidate_score = row_score(candidate)
            baseline_win_rate = row_win_rate(baseline)
            candidate_win_rate = row_win_rate(candidate)
            if baseline_score is not None and candidate_score is not None:
                delta = candidate_score - baseline_score
                style_deltas.append(delta)
                signs[sign(delta)] += 1
            if baseline_win_rate is not None and candidate_win_rate is not None:
                win_rate_deltas.append(candidate_win_rate - baseline_win_rate)
        deltas[mode] = {
            "baseline_mode": baseline_mode,
            "paired_rows": len(style_deltas),
            "missing_pairs": missing_pairs,
            "mean_style_likeness_delta": mean(style_deltas),
            "mean_hard_negative_win_rate_delta": mean(win_rate_deltas),
            "style_delta_sign_counts": dict(signs),
        }
    return deltas


def eval_summary_for_pack_ids(
    rows: list[dict[str, Any]],
    pack_ids: set[str],
    *,
    baseline_mode: str,
) -> dict[str, Any]:
    scoped_rows = [row for row in rows if pack_id(row) in pack_ids]
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoped_rows:
        mode = row.get("mode")
        if isinstance(mode, str) and mode:
            by_mode[mode].append(row)
    return {
        "rows": len(scoped_rows),
        "modes": {
            mode: mode_summary(mode_rows)
            for mode, mode_rows in sorted(by_mode.items())
        },
        "paired_deltas": paired_delta_summary(scoped_rows, baseline_mode=baseline_mode),
    }


def manifest_paper_tags(manifest: dict[str, Any]) -> list[str]:
    profiles = manifest.get("paper_eligible_cluster_tags", [])
    if not isinstance(profiles, list):
        return []
    tags = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        tag = profile.get("tag")
        if isinstance(tag, str) and tag:
            tags.append(tag)
    return tags


def pack_profile_index(pack_profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for profile in pack_profiles:
        profile_pack_id = pack_id(profile)
        if profile_pack_id:
            indexed[profile_pack_id] = profile
    return indexed


def profile_tags(profile: dict[str, Any], field: str) -> list[str]:
    values = profile.get(field, [])
    if not isinstance(values, list):
        return []
    return sorted(str(value) for value in values if isinstance(value, str) and value)


def build_strata(
    pack_profiles: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    paper_tags = set(manifest_paper_tags(manifest))
    strata = [
        {
            "stratum_kind": "population",
            "stratum_id": "all_packs",
            "label": "All packs",
            "paper_facing": True,
            "claim_scope": "single_count_population",
            "pack_ids": sorted(
                pack_id(profile)
                for profile in pack_profiles
                if pack_id(profile) is not None
            ),
        }
    ]

    tag_members: dict[str, set[str]] = defaultdict(set)
    family_members: dict[str, set[str]] = defaultdict(set)
    background_members: dict[str, set[str]] = defaultdict(set)
    topic_like_members: dict[str, set[str]] = defaultdict(set)
    topic_like_any: set[str] = set()
    no_topic_like: set[str] = set()
    for profile in pack_profiles:
        profile_pack_id = pack_id(profile)
        if profile_pack_id is None:
            continue
        for tag in profile_tags(profile, "eligible_style_tags"):
            if tag in paper_tags:
                tag_members[tag].add(profile_pack_id)
        for family in profile_tags(profile, "style_families"):
            family_members[family].add(profile_pack_id)
        for tag in profile_tags(profile, "background_style_tags"):
            background_members[tag].add(profile_pack_id)
        topic_tags = profile_tags(profile, "excluded_topic_like_tags")
        if topic_tags:
            topic_like_any.add(profile_pack_id)
        else:
            no_topic_like.add(profile_pack_id)
        for tag in topic_tags:
            topic_like_members[tag].add(profile_pack_id)

    for tag in sorted(tag_members, key=lambda value: (-len(tag_members[value]), value)):
        strata.append(
            {
                "stratum_kind": "paper_eligible_style_tag",
                "stratum_id": tag,
                "label": tag,
                "paper_facing": True,
                "claim_scope": "multi_label_cluster_macro",
                "pack_ids": sorted(tag_members[tag]),
            }
        )
    for family in sorted(
        family_members, key=lambda value: (-len(family_members[value]), value)
    ):
        strata.append(
            {
                "stratum_kind": "style_family",
                "stratum_id": family,
                "label": family,
                "paper_facing": family != "register_voice",
                "claim_scope": (
                    "family_stratification"
                    if family != "register_voice"
                    else "diagnostic_family_contains_background_register"
                ),
                "pack_ids": sorted(family_members[family]),
            }
        )
    for tag in sorted(
        background_members, key=lambda value: (-len(background_members[value]), value)
    ):
        strata.append(
            {
                "stratum_kind": "background_style_tag",
                "stratum_id": tag,
                "label": tag,
                "paper_facing": False,
                "claim_scope": "diagnostic_background_register",
                "pack_ids": sorted(background_members[tag]),
            }
        )
    topic_groups = {
        "has_topic_like_exclusion": topic_like_any,
        "no_topic_like_exclusion": no_topic_like,
    }
    for tag, members in topic_groups.items():
        strata.append(
            {
                "stratum_kind": "topic_like_exclusion_group",
                "stratum_id": tag,
                "label": tag,
                "paper_facing": False,
                "claim_scope": "diagnostic_topic_like_exclusion_check",
                "pack_ids": sorted(members),
            }
        )
    for tag in sorted(
        topic_like_members, key=lambda value: (-len(topic_like_members[value]), value)
    ):
        strata.append(
            {
                "stratum_kind": "topic_like_exclusion_tag",
                "stratum_id": tag,
                "label": tag,
                "paper_facing": False,
                "claim_scope": "diagnostic_excluded_topic_tag",
                "pack_ids": sorted(topic_like_members[tag]),
            }
        )
    return strata


def macro_averages(
    stratum_summaries: list[dict[str, Any]], *, mode: str
) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in stratum_summaries:
        if summary.get("stratum_kind") == "population":
            continue
        paired = summary.get("paired_deltas", {})
        if not isinstance(paired, dict) or mode not in paired:
            continue
        if not isinstance(paired[mode], dict):
            continue
        if paired[mode].get("paired_rows", 0) <= 0:
            continue
        by_kind[str(summary.get("stratum_kind"))].append(summary)

    output = {}
    for kind, summaries in sorted(by_kind.items()):
        style_deltas = []
        win_deltas = []
        positive = 0
        nonpositive = 0
        for summary in summaries:
            paired = summary["paired_deltas"][mode]
            style_delta = paired.get("mean_style_likeness_delta")
            win_delta = paired.get("mean_hard_negative_win_rate_delta")
            if isinstance(style_delta, (int, float)):
                style_deltas.append(float(style_delta))
                if style_delta > 0:
                    positive += 1
                else:
                    nonpositive += 1
            if isinstance(win_delta, (int, float)):
                win_deltas.append(float(win_delta))
        output[kind] = {
            mode: {
                "mode": mode,
                "strata": len(summaries),
                "mean_of_mean_style_likeness_delta": mean(style_deltas),
                "mean_of_mean_hard_negative_win_rate_delta": mean(win_deltas),
                "positive_style_delta_strata": positive,
                "nonpositive_style_delta_strata": nonpositive,
            }
        }
    return output


def assignment_stats(strata: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    kind_strata = [stratum for stratum in strata if stratum.get("stratum_kind") == kind]
    assignments = sum(len(stratum.get("pack_ids", [])) for stratum in kind_strata)
    memberships: Counter[str] = Counter()
    for stratum in kind_strata:
        memberships.update(
            pack_id for pack_id in stratum.get("pack_ids", []) if isinstance(pack_id, str)
        )
    unique_packs = set(memberships)
    return {
        "strata": len(kind_strata),
        "assignments": assignments,
        "unique_packs": len(unique_packs),
        "duplicate_factor": (
            round(assignments / len(unique_packs), 3) if unique_packs else None
        ),
        "membership_count_distribution": dict(
            Counter(str(count) for count in memberships.values()).most_common()
        ),
        "max_memberships_per_pack": max(memberships.values()) if memberships else None,
    }


def model_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "solver_model",
        "solver_backend",
        "judge_model",
        "judge_backend",
        "judge_config_prefix",
        "judge_config_model",
    ]
    return {
        key: dict(
            Counter(
                str(row.get(key) if row.get(key) is not None else "null")
                for row in rows
            ).most_common()
        )
        for key in keys
    }


def coverage_summary(
    *,
    eval_rows: list[dict[str, Any]],
    latest_rows: list[dict[str, Any]],
    pack_profiles: list[dict[str, Any]],
    population: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest_pack_ids = {
        value for profile in pack_profiles if (value := pack_id(profile)) is not None
    }
    eval_pack_ids = {value for row in latest_rows if (value := pack_id(row)) is not None}
    paired = {}
    if isinstance(population, dict):
        paired = population.get("paired_deltas", {})
        if not isinstance(paired, dict):
            paired = {}
    paired_cells = [
        payload.get("paired_rows")
        for payload in paired.values()
        if isinstance(payload, dict) and isinstance(payload.get("paired_rows"), int)
    ]
    return {
        "manifest_pack_rows": len(pack_profiles),
        "eval_raw_rows": len(eval_rows),
        "eval_latest_rows": len(latest_rows),
        "duplicate_cells_dropped": max(0, len(eval_rows) - len(latest_rows)),
        "eval_pack_count": len(eval_pack_ids),
        "missing_manifest_packs_in_eval": sorted(manifest_pack_ids - eval_pack_ids),
        "extra_eval_packs_not_in_manifest": sorted(eval_pack_ids - manifest_pack_ids),
        "paired_cells": max(paired_cells) if paired_cells else 0,
    }


def summarize_stratified_author_style_eval(
    *,
    eval_rows: list[dict[str, Any]],
    pack_profiles: list[dict[str, Any]],
    manifest: dict[str, Any],
    baseline_mode: str = "prompt_only",
    macro_mode: str = "few_shot_examples_only",
) -> dict[str, Any]:
    latest_rows = latest_eval_cells(eval_rows)
    strata = build_strata(pack_profiles, manifest)
    stratum_summaries = []
    for stratum in strata:
        pack_ids = {
            value for value in stratum.get("pack_ids", []) if isinstance(value, str)
        }
        eval_summary = eval_summary_for_pack_ids(
            latest_rows, pack_ids, baseline_mode=baseline_mode
        )
        stratum_summaries.append(
            {
                "stratum_kind": stratum["stratum_kind"],
                "stratum_id": stratum["stratum_id"],
                "label": stratum["label"],
                "paper_facing": stratum["paper_facing"],
                "claim_scope": stratum["claim_scope"],
                "pack_count": len(pack_ids),
                "pack_ids": sorted(pack_ids),
                **eval_summary,
            }
        )
    population = stratum_summaries[0] if stratum_summaries else None
    macro = macro_averages(stratum_summaries, mode=macro_mode)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_boundary": ARTIFACT_BOUNDARY,
        "comparison": {
            "baseline_mode": baseline_mode,
            "treatment_mode": macro_mode,
            "delta_direction": f"{macro_mode} - {baseline_mode}",
        },
        "baseline_mode": baseline_mode,
        "macro_mode": macro_mode,
        "rows": len(latest_rows),
        "status_counts": dict(
            Counter(str(row.get("status") or "unknown") for row in latest_rows)
        ),
        "model_inventory": model_inventory(latest_rows),
        "coverage": coverage_summary(
            eval_rows=eval_rows,
            latest_rows=latest_rows,
            pack_profiles=pack_profiles,
            population=population,
        ),
        "overall_unique_pack_effect": population,
        "population": population,
        "strata": stratum_summaries,
        "macro_averages": macro,
        "cluster_balanced_macro": macro.get("paper_eligible_style_tag", {}).get(
            macro_mode
        ),
        "multi_label_counting": {
            "paper_eligible_style_tag": assignment_stats(
                strata, kind="paper_eligible_style_tag"
            ),
            "style_family": assignment_stats(strata, kind="style_family"),
            "background_style_tag": assignment_stats(
                strata, kind="background_style_tag"
            ),
            "topic_like_exclusion_tag": assignment_stats(
                strata, kind="topic_like_exclusion_tag"
            ),
            "claim_note": (
                "Use population rows for single-count overall claims; use macro "
                "averages for cluster-level claims because packs can belong to "
                "multiple style tags."
            ),
        },
    }
