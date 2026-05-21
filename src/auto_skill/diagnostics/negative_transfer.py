"""Diagnostics for wrong-author author-style negative-transfer controls."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from auto_skill.cleaning.author_style.eval_summary import (
    latest_eval_cells,
    mean,
    row_score,
    row_win_rate,
)

SCHEMA_VERSION = "author-style-negative-transfer-diagnostic/v1"
ARTIFACT_BOUNDARY = {
    "must_not_use_for_induction": True,
    "may_use_for": [
        "paper_figure",
        "post_hoc_error_analysis",
        "negative_transfer_diagnostics",
    ],
    "must_not_use_for": [
        "skill_prompt",
        "skill_induction_input",
        "heldout_generation",
        "judge_prompt",
        "hard_negative_rerank",
        "future_subset_selection_by_outcome",
    ],
    "intended_use": "post_hoc_target_vs_wrong_author_analysis",
}


def _pack_id(row: dict[str, Any]) -> str | None:
    value = row.get("pack_id")
    return value if isinstance(value, str) and value else None


def _task_id(row: dict[str, Any]) -> str | None:
    value = row.get("task_id")
    return value if isinstance(value, str) and value else None


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value if isinstance(item, str) and item)


def _scalar_cell_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (
            str(row.get("pack_id")),
            str(row.get("task_id")),
            str(row.get("mode")),
        ): row
        for row in latest_eval_cells(rows)
        if row.get("status") == "success"
        and isinstance(row.get("pack_id"), str)
        and isinstance(row.get("task_id"), str)
        and isinstance(row.get("mode"), str)
    }


def _manifest_pair_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        return {}
    indexed = {}
    for row in pairs:
        if not isinstance(row, dict):
            continue
        pack_id = row.get("target_pack_id")
        if isinstance(pack_id, str) and pack_id:
            indexed[pack_id] = row
    return indexed


def _profile_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        pack_id: row
        for row in rows
        if (pack_id := _pack_id(row)) is not None
    }


def _pair_index(swap_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pairs = swap_audit.get("pairs")
    if not isinstance(pairs, list):
        return {}
    indexed = {}
    for row in pairs:
        if not isinstance(row, dict):
            continue
        pack_id = row.get("pack_id")
        if isinstance(pack_id, str) and pack_id:
            indexed[pack_id] = row
    return indexed


def _pair_verdict(pair: dict[str, Any] | None) -> str:
    if not pair:
        return "pair_missing"
    status = pair.get("status")
    if status == "swap_stable_anchor_wins":
        return "target_stable_win"
    if status == "swap_stable_candidate_wins":
        return "wrong_author_stable_win"
    if status == "swap_stable_tie":
        return "stable_tie"
    if status == "swap_flip_decisive":
        return "order_flip"
    if status == "swap_one_decisive_one_tie":
        return "one_order_tie"
    if status == "swap_pair_incomplete":
        return "pair_incomplete"
    return str(status or "pair_unknown")


def _metric(profile: dict[str, Any], name: str) -> float | None:
    metrics = profile.get("metric_means")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scalar_delta_payload(
    *,
    index: dict[tuple[str, str, str], dict[str, Any]],
    pack_id: str,
    task_id: str,
    target_mode: str,
    wrong_mode: str,
    prompt_mode: str,
) -> dict[str, Any]:
    target = index.get((pack_id, task_id, target_mode))
    wrong = index.get((pack_id, task_id, wrong_mode))
    prompt = index.get((pack_id, task_id, prompt_mode))
    target_score = row_score(target or {})
    wrong_score = row_score(wrong or {})
    prompt_score = row_score(prompt or {})
    target_win = row_win_rate(target or {})
    wrong_win = row_win_rate(wrong or {})
    prompt_win = row_win_rate(prompt or {})
    return {
        "target_style_likeness": target_score,
        "wrong_author_style_likeness": wrong_score,
        "prompt_style_likeness": prompt_score,
        "target_minus_wrong_style_likeness": (
            round(target_score - wrong_score, 6)
            if target_score is not None and wrong_score is not None
            else None
        ),
        "target_minus_prompt_style_likeness": (
            round(target_score - prompt_score, 6)
            if target_score is not None and prompt_score is not None
            else None
        ),
        "wrong_minus_prompt_style_likeness": (
            round(wrong_score - prompt_score, 6)
            if wrong_score is not None and prompt_score is not None
            else None
        ),
        "target_hard_negative_win_rate": target_win,
        "wrong_author_hard_negative_win_rate": wrong_win,
        "prompt_hard_negative_win_rate": prompt_win,
        "target_minus_wrong_hard_negative_win_rate": (
            round(target_win - wrong_win, 6)
            if target_win is not None and wrong_win is not None
            else None
        ),
    }


def build_pack_diagnostics(
    *,
    pack_profiles: list[dict[str, Any]],
    scalar_eval_sets: dict[str, list[dict[str, Any]]],
    swap_audit: dict[str, Any],
    negative_transfer_manifest: dict[str, Any],
    target_mode: str = "few_shot_examples_only",
    wrong_mode: str = "wrong_author_examples_same_source",
    prompt_mode: str = "prompt_only",
) -> list[dict[str, Any]]:
    profile_by_pack = _profile_index(pack_profiles)
    scalar_indexes = {
        label: _scalar_cell_index(rows) for label, rows in scalar_eval_sets.items()
    }
    pair_by_pack = _pair_index(swap_audit)
    manifest_by_pack = _manifest_pair_index(negative_transfer_manifest)
    diagnostics = []
    for pack_id in sorted(profile_by_pack):
        profile = profile_by_pack[pack_id]
        task_id = None
        for rows in scalar_indexes.values():
            for row_pack_id, row_task_id, mode in rows:
                if row_pack_id == pack_id and mode == target_mode:
                    task_id = row_task_id
                    break
            if task_id:
                break
        if task_id is None:
            pair = pair_by_pack.get(pack_id)
            task_id = str(pair.get("task_id")) if isinstance(pair, dict) else ""
        pair = pair_by_pack.get(pack_id)
        pair_verdict = _pair_verdict(pair)
        scalar_payload = {
            label: _scalar_delta_payload(
                index=index,
                pack_id=pack_id,
                task_id=task_id,
                target_mode=target_mode,
                wrong_mode=wrong_mode,
                prompt_mode=prompt_mode,
            )
            for label, index in sorted(scalar_indexes.items())
        }
        diagnostics.append(
            {
                "schema_version": "author-style-negative-transfer-pack-diagnostic/v1",
                "pack_id": pack_id,
                "task_id": task_id,
                "pairwise_verdict": pair_verdict,
                "pairwise_status": pair.get("status") if isinstance(pair, dict) else None,
                "pairwise_confidence_by_order": (
                    pair.get("confidence_by_order") if isinstance(pair, dict) else {}
                ),
                "pairwise_winner_by_order": (
                    pair.get("winner_by_order") if isinstance(pair, dict) else {}
                ),
                "scalar_eval": scalar_payload,
                "negative_transfer_match": manifest_by_pack.get(pack_id, {}),
                "eligible_style_tags": _as_list(profile.get("eligible_style_tags")),
                "background_style_tags": _as_list(profile.get("background_style_tags")),
                "style_families": _as_list(profile.get("style_families")),
                "topic_like_exclusions": _as_list(profile.get("topic_like_exclusions")),
                "quality_flags": _as_list(profile.get("quality_flags")),
                "diagnostic_caveats": _as_list(profile.get("diagnostic_caveats")),
                "paper_quality_tier": profile.get("paper_quality_tier"),
                "source_pool_margin": _metric(profile, "source_pool_margin"),
                "source_pool_pairwise_win_rate": _metric(profile, "source_pool_pairwise_win_rate"),
                "style_similarity_margin_vs_selected_negatives": _metric(
                    profile, "style_similarity_margin_vs_selected_negatives"
                ),
                "train_heldout_content_tag_jaccard": _metric(
                    profile, "train_heldout_content_tag_jaccard"
                ),
            }
        )
    return diagnostics


def _sign(value: float | None) -> str:
    if value is None:
        return "missing"
    if value > 0:
        return "target_higher"
    if value < 0:
        return "wrong_author_higher"
    return "tie"


def _scalar_summary(pack_rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    deltas = []
    win_deltas = []
    signs = Counter()
    for row in pack_rows:
        payload = row.get("scalar_eval", {}).get(label, {})
        if not isinstance(payload, dict):
            continue
        delta = payload.get("target_minus_wrong_style_likeness")
        if isinstance(delta, (int, float)) and not isinstance(delta, bool):
            deltas.append(float(delta))
        signs[_sign(delta if isinstance(delta, (int, float)) else None)] += 1
        win_delta = payload.get("target_minus_wrong_hard_negative_win_rate")
        if isinstance(win_delta, (int, float)) and not isinstance(win_delta, bool):
            win_deltas.append(float(win_delta))
    return {
        "packs": len(pack_rows),
        "mean_target_minus_wrong_style_likeness": mean(deltas),
        "mean_target_minus_wrong_hard_negative_win_rate": mean(win_deltas),
        "target_minus_wrong_style_sign_counts": dict(signs),
    }


def _stratum_summary(
    pack_rows: list[dict[str, Any]],
    *,
    stratum_kind: str,
    stratum_id: str,
    scalar_labels: list[str],
) -> dict[str, Any]:
    verdict_counts = Counter(str(row.get("pairwise_verdict") or "unknown") for row in pack_rows)
    decisive = (
        verdict_counts["target_stable_win"]
        + verdict_counts["wrong_author_stable_win"]
        + verdict_counts["order_flip"]
    )
    return {
        "stratum_kind": stratum_kind,
        "stratum_id": stratum_id,
        "pack_count": len(pack_rows),
        "pairwise_verdict_counts": dict(verdict_counts),
        "target_stable_win_rate_all": (
            round(verdict_counts["target_stable_win"] / len(pack_rows), 6)
            if pack_rows
            else None
        ),
        "wrong_or_order_sensitive_rate_all": (
            round(
                (verdict_counts["wrong_author_stable_win"] + verdict_counts["order_flip"])
                / len(pack_rows),
                6,
            )
            if pack_rows
            else None
        ),
        "wrong_author_stable_win_rate_decisive": (
            round(verdict_counts["wrong_author_stable_win"] / decisive, 6)
            if decisive
            else None
        ),
        "scalar_summaries": {
            label: _scalar_summary(pack_rows, label) for label in scalar_labels
        },
    }


def _build_strata(
    pack_rows: list[dict[str, Any]], scalar_labels: list[str]
) -> list[dict[str, Any]]:
    strata = [
        _stratum_summary(
            pack_rows,
            stratum_kind="population",
            stratum_id="all_packs",
            scalar_labels=scalar_labels,
        )
    ]
    memberships: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pack_rows:
        for tag in row.get("eligible_style_tags", []):
            memberships[("eligible_style_tag", str(tag))].append(row)
        for family in row.get("style_families", []):
            memberships[("style_family", str(family))].append(row)
        for tag in row.get("background_style_tags", []):
            memberships[("background_style_tag", str(tag))].append(row)
        if row.get("topic_like_exclusions"):
            memberships[("topic_like_group", "has_topic_like_exclusion")].append(row)
        else:
            memberships[("topic_like_group", "no_topic_like_exclusion")].append(row)
        for flag in row.get("quality_flags", []):
            memberships[("quality_flag", str(flag))].append(row)
    for (kind, stratum_id), rows in sorted(
        memberships.items(), key=lambda item: (item[0][0], -len(item[1]), item[0][1])
    ):
        strata.append(
            _stratum_summary(
                rows,
                stratum_kind=kind,
                stratum_id=stratum_id,
                scalar_labels=scalar_labels,
            )
        )
    return strata


def _hard_case_score(row: dict[str, Any], scalar_labels: list[str]) -> tuple[int, float, str]:
    verdict_priority = {
        "wrong_author_stable_win": 0,
        "order_flip": 1,
        "pair_incomplete": 2,
        "one_order_tie": 3,
        "stable_tie": 4,
        "target_stable_win": 5,
    }.get(str(row.get("pairwise_verdict")), 6)
    deltas = []
    for label in scalar_labels:
        payload = row.get("scalar_eval", {}).get(label, {})
        if isinstance(payload, dict):
            delta = payload.get("target_minus_wrong_style_likeness")
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                deltas.append(float(delta))
    return (verdict_priority, mean(deltas) if deltas else 999.0, str(row.get("pack_id")))


def summarize_negative_transfer_diagnostic(
    *,
    pack_rows: list[dict[str, Any]],
    scalar_labels: list[str],
    negative_transfer_manifest: dict[str, Any],
    swap_audit: dict[str, Any],
    top_k: int = 20,
) -> dict[str, Any]:
    strata = _build_strata(pack_rows, scalar_labels)
    hard_cases = sorted(pack_rows, key=lambda row: _hard_case_score(row, scalar_labels))[:top_k]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_boundary": ARTIFACT_BOUNDARY,
        "comparison": {
            "anchor_mode": "few_shot_examples_only",
            "wrong_author_mode": "wrong_author_examples_same_source",
            "claim": "target-author examples vs public-only same-source wrong-author examples",
        },
        "pack_count": len(pack_rows),
        "negative_transfer_manifest_summary": {
            key: negative_transfer_manifest.get(key)
            for key in (
                "policy",
                "seed",
                "pack_count",
                "unique_impostor_pack_count",
                "mean_heldout_to_impostor_train_tag_jaccard",
                "mean_length_log_ratio",
                "max_impostor_reuse_count",
            )
        },
        "swap_audit_summary": {
            "status_counts": swap_audit.get("status_counts"),
            "by_candidate": swap_audit.get("by_candidate"),
        },
        "scalar_labels": scalar_labels,
        "strata": strata,
        "population": strata[0] if strata else None,
        "top_hard_cases": [
            {
                "pack_id": row["pack_id"],
                "pairwise_verdict": row["pairwise_verdict"],
                "pairwise_status": row.get("pairwise_status"),
                "scalar_eval": row.get("scalar_eval"),
                "eligible_style_tags": row.get("eligible_style_tags"),
                "style_families": row.get("style_families"),
                "quality_flags": row.get("quality_flags"),
                "source_pool_margin": row.get("source_pool_margin"),
                "negative_transfer_match": row.get("negative_transfer_match"),
            }
            for row in hard_cases
        ],
    }
