"""Audit threshold and filtering helpers for author-style GPT outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any


def audit_threshold_policy(args: argparse.Namespace) -> str:
    return str(getattr(args, "audit_thresholds", "none") or "none")


def audit_passes_thresholds(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if row.get("status") != "success":
        return False
    audit = row.get("audit")
    if not isinstance(audit, dict):
        return False
    policy = audit_threshold_policy(args)
    if policy == "none":
        return True
    if policy != "smoke":
        raise ValueError(f"unknown audit threshold policy: {policy}")
    if not (audit.get("usable") and audit.get("should_use_for_smoke")):
        return False
    return (
        float(audit.get("style_extractability_1_to_5") or 0) >= args.min_style_extractability
        and float(audit.get("negative_strength_1_to_5") or 0) >= args.min_negative_strength
        and float(audit.get("topic_leakage_risk_1_to_5") or 6) <= args.max_topic_leakage
        and float(audit.get("model_familiarity_risk_1_to_5") or 6) <= args.max_model_familiarity
    )


def filter_by_pack_ids(
    rows: list[dict[str, Any]],
    accepted_pack_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("pack_id") in accepted_pack_ids]


def filter_negatives_by_pack_ids(
    rows: list[dict[str, Any]],
    accepted_pack_ids: set[str],
    *,
    require_gpt_selected: bool,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        pack_id = str(row.get("target_task_ref", "")).split("::heldout::", maxsplit=1)[0]
        if pack_id not in accepted_pack_ids:
            continue
        if require_gpt_selected and row.get("gpt_rerank", {}).get("status") != "selected":
            continue
        filtered.append(row)
    return filtered


def pack_ids_with_negative_coverage(
    private_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    accepted_pack_ids: set[str],
    *,
    min_negatives_per_heldout: int,
) -> set[str]:
    if min_negatives_per_heldout <= 0:
        return set(accepted_pack_ids)
    counts = Counter(str(row.get("target_task_ref") or "") for row in negative_rows)
    covered_pack_ids: set[str] = set()
    for row in private_rows:
        pack_id = str(row.get("pack_id") or "")
        if pack_id not in accepted_pack_ids:
            continue
        heldout_rows = row.get("heldout_private")
        if not isinstance(heldout_rows, list) or not heldout_rows:
            continue
        task_refs = [
            str(heldout.get("task_ref") or "")
            for heldout in heldout_rows
            if isinstance(heldout, dict)
        ]
        has_enough_negatives = all(
            counts[task_ref] >= min_negatives_per_heldout for task_ref in task_refs
        )
        if task_refs and has_enough_negatives:
            covered_pack_ids.add(pack_id)
    return covered_pack_ids
