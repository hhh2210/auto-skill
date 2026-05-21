"""Selection helpers for accepted author-style artifact subsets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto_skill.example_packs import load_jsonl, write_jsonl

TIER_RANK = {
    "cross_topic_high_signal": 5,
    "temporal_same_context_high_signal": 4,
    "cross_topic_moderate_signal": 3,
    "temporal_same_context_moderate_signal": 2,
    "cross_topic_low_signal": 1,
    "temporal_context_dominated": 0,
}


def numeric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def nested_numeric(row: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return numeric(current)


def quality_rank_key(indexed_row: tuple[int, dict[str, Any]]) -> tuple[float, ...]:
    index, row = indexed_row
    return (
        float(TIER_RANK.get(str(row.get("paper_quality_tier") or ""), -1)),
        nested_numeric(row, ("deterministic_metrics", "margin_vs_source_pool_impostors")),
        nested_numeric(row, ("source_pool_impostor_baseline", "source_pool_pairwise_win_rate")),
        nested_numeric(row, ("deterministic_metrics", "mean_train_heldout_style_similarity")),
        numeric(row.get("mean_hard_negative_style_confusability_1_to_5")),
        -float(index),
    )


def select_quality_rows(
    quality_rows: list[dict[str, Any]],
    *,
    dataset_label: str | None = None,
    paper_plot_role: str | None = None,
    require_paper_eligible: bool = False,
    limit: int | None = None,
    sort_by: str = "quality_rank",
) -> list[dict[str, Any]]:
    indexed = []
    for index, row in enumerate(quality_rows):
        if dataset_label and row.get("dataset_label") != dataset_label:
            continue
        if paper_plot_role and row.get("paper_plot_role") != paper_plot_role:
            continue
        if require_paper_eligible and not row.get("paper_plot_eligible"):
            continue
        if not row.get("pack_id"):
            continue
        indexed.append((index, row))

    if sort_by == "quality_rank":
        indexed.sort(key=quality_rank_key, reverse=True)
    elif sort_by != "source_order":
        raise ValueError(f"unknown sort_by: {sort_by}")

    rows = [row for _, row in indexed]
    return rows[:limit] if limit is not None else rows


def pack_id_from_task_ref(task_ref: str) -> str:
    marker = "::heldout::"
    if marker in task_ref:
        return task_ref.split(marker, 1)[0]
    return task_ref.split("::", 1)[0]


def filter_hard_negatives_by_pack_ids(
    rows: list[dict[str, Any]],
    pack_ids: set[str],
    *,
    require_gpt_selected: bool = False,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        ref = row.get("target_task_ref")
        if isinstance(ref, str) and pack_id_from_task_ref(ref) in pack_ids:
            if (
                require_gpt_selected
                and row.get("gpt_rerank", {}).get("status") != "selected"
            ):
                continue
            selected.append(row)
    return selected


def audit_payload(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") != "success":
        return {}
    audit = row.get("audit")
    return audit if isinstance(audit, dict) else {}


def audit_passes_selection(
    row: dict[str, Any],
    *,
    require_usable: bool,
    require_should_use: bool,
    min_style_extractability: float,
    min_negative_strength: float,
    max_topic_leakage: float,
    max_model_familiarity: float,
) -> bool:
    audit = audit_payload(row)
    if not audit:
        return False
    if require_usable and not audit.get("usable"):
        return False
    if require_should_use and not audit.get("should_use_for_smoke"):
        return False
    return (
        numeric(audit.get("style_extractability_1_to_5")) >= min_style_extractability
        and numeric(audit.get("negative_strength_1_to_5")) >= min_negative_strength
        and numeric(audit.get("topic_leakage_risk_1_to_5")) <= max_topic_leakage
        and numeric(audit.get("model_familiarity_risk_1_to_5")) <= max_model_familiarity
    )


def audit_rank_key(indexed_row: tuple[int, dict[str, Any]]) -> tuple[float, ...]:
    index, row = indexed_row
    audit = audit_payload(row)
    return (
        numeric(audit.get("style_extractability_1_to_5")),
        numeric(audit.get("negative_strength_1_to_5")),
        -numeric(audit.get("topic_leakage_risk_1_to_5")),
        -numeric(audit.get("model_familiarity_risk_1_to_5")),
        -float(index),
    )


def select_audit_rows(
    audit_rows: list[dict[str, Any]],
    *,
    require_usable: bool = True,
    require_should_use: bool = True,
    min_style_extractability: float = 3.0,
    min_negative_strength: float = 4.0,
    max_topic_leakage: float = 4.0,
    max_model_familiarity: float = 3.0,
    limit: int | None = None,
    sort_by: str = "audit_rank",
) -> list[dict[str, Any]]:
    indexed = []
    for index, row in enumerate(audit_rows):
        if not row.get("pack_id"):
            continue
        if audit_passes_selection(
            row,
            require_usable=require_usable,
            require_should_use=require_should_use,
            min_style_extractability=min_style_extractability,
            min_negative_strength=min_negative_strength,
            max_topic_leakage=max_topic_leakage,
            max_model_familiarity=max_model_familiarity,
        ):
            indexed.append((index, row))

    if sort_by == "audit_rank":
        indexed.sort(key=audit_rank_key, reverse=True)
    elif sort_by != "source_order":
        raise ValueError(f"unknown sort_by: {sort_by}")

    rows = [row for _, row in indexed]
    return rows[:limit] if limit is not None else rows


def build_author_style_subset(
    *,
    quality_rows: list[dict[str, Any]],
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    hard_negatives: list[dict[str, Any]],
    dataset_label: str | None,
    paper_plot_role: str | None,
    require_paper_eligible: bool,
    limit: int | None,
    sort_by: str,
) -> dict[str, Any]:
    selected_quality = select_quality_rows(
        quality_rows,
        dataset_label=dataset_label,
        paper_plot_role=paper_plot_role,
        require_paper_eligible=require_paper_eligible,
        limit=limit,
        sort_by=sort_by,
    )
    pack_ids = {str(row["pack_id"]) for row in selected_quality}
    selected_packs = [row for row in packs if row.get("pack_id") in pack_ids]
    selected_private = [row for row in private_eval if row.get("pack_id") in pack_ids]
    selected_negatives = filter_hard_negatives_by_pack_ids(hard_negatives, pack_ids)
    expected_task_refs = {
        str(task.get("task_id"))
        for pack in selected_packs
        for task in pack.get("heldout_tasks", [])
        if isinstance(task, dict) and task.get("task_id")
    }
    covered_task_refs = {
        str(row.get("target_task_ref"))
        for row in selected_negatives
        if row.get("target_task_ref") in expected_task_refs
    }
    missing_pack_ids = sorted(pack_ids - {str(row.get("pack_id")) for row in selected_packs})
    return {
        "quality_rows": selected_quality,
        "packs": selected_packs,
        "private_eval": selected_private,
        "hard_negatives": selected_negatives,
        "summary": {
            "schema_version": "author-style-artifact-selection/v1",
            "dataset_label": dataset_label,
            "paper_plot_role": paper_plot_role,
            "require_paper_eligible": require_paper_eligible,
            "limit": limit,
            "sort_by": sort_by,
            "selected_pack_ids": [str(row["pack_id"]) for row in selected_quality],
            "quality_rows": len(selected_quality),
            "packs": len(selected_packs),
            "private_eval_rows": len(selected_private),
            "hard_negatives": len(selected_negatives),
            "expected_task_refs": len(expected_task_refs),
            "task_refs_with_hard_negatives": len(covered_task_refs),
            "missing_pack_ids": missing_pack_ids,
        },
    }


def build_author_style_audit_subset(
    *,
    audit_rows: list[dict[str, Any]],
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    hard_negatives: list[dict[str, Any]],
    require_usable: bool,
    require_should_use: bool,
    min_style_extractability: float,
    min_negative_strength: float,
    max_topic_leakage: float,
    max_model_familiarity: float,
    limit: int | None,
    sort_by: str,
    min_negatives_per_heldout: int,
    require_gpt_selected_negatives: bool,
) -> dict[str, Any]:
    selected_audits = select_audit_rows(
        audit_rows,
        require_usable=require_usable,
        require_should_use=require_should_use,
        min_style_extractability=min_style_extractability,
        min_negative_strength=min_negative_strength,
        max_topic_leakage=max_topic_leakage,
        max_model_familiarity=max_model_familiarity,
        limit=limit,
        sort_by=sort_by,
    )
    pack_ids = {str(row["pack_id"]) for row in selected_audits}
    selected_packs = [row for row in packs if row.get("pack_id") in pack_ids]
    selected_private = [row for row in private_eval if row.get("pack_id") in pack_ids]
    selected_negatives = filter_hard_negatives_by_pack_ids(
        hard_negatives,
        pack_ids,
        require_gpt_selected=require_gpt_selected_negatives,
    )
    expected_task_refs = {
        str(task.get("task_id"))
        for pack in selected_packs
        for task in pack.get("heldout_tasks", [])
        if isinstance(task, dict) and task.get("task_id")
    }
    covered_task_refs = {
        str(row.get("target_task_ref"))
        for row in selected_negatives
        if row.get("target_task_ref") in expected_task_refs
    }
    counts_by_task = {
        ref: sum(1 for row in selected_negatives if row.get("target_task_ref") == ref)
        for ref in expected_task_refs
    }
    dropped_insufficient_negative_refs = sorted(
        ref for ref, count in counts_by_task.items() if count < min_negatives_per_heldout
    )
    covered_pack_ids = {
        pack_id_from_task_ref(ref)
        for ref in covered_task_refs
        if counts_by_task.get(ref, 0) >= min_negatives_per_heldout
    }
    final_pack_ids = pack_ids & covered_pack_ids
    selected_audits = [row for row in selected_audits if row.get("pack_id") in final_pack_ids]
    selected_packs = [row for row in selected_packs if row.get("pack_id") in final_pack_ids]
    selected_private = [row for row in selected_private if row.get("pack_id") in final_pack_ids]
    selected_negatives = filter_hard_negatives_by_pack_ids(
        selected_negatives,
        final_pack_ids,
        require_gpt_selected=require_gpt_selected_negatives,
    )
    expected_task_refs = {
        str(task.get("task_id"))
        for pack in selected_packs
        for task in pack.get("heldout_tasks", [])
        if isinstance(task, dict) and task.get("task_id")
    }
    covered_task_refs = {
        str(row.get("target_task_ref"))
        for row in selected_negatives
        if row.get("target_task_ref") in expected_task_refs
    }
    counts_by_task = {
        ref: sum(1 for row in selected_negatives if row.get("target_task_ref") == ref)
        for ref in expected_task_refs
    }
    insufficient_negative_refs = sorted(
        ref for ref, count in counts_by_task.items() if count < min_negatives_per_heldout
    )
    missing_pack_ids = sorted(final_pack_ids - {str(row.get("pack_id")) for row in selected_packs})
    return {
        "audits": selected_audits,
        "packs": selected_packs,
        "private_eval": selected_private,
        "hard_negatives": selected_negatives,
        "summary": {
            "schema_version": "author-style-audit-selection/v1",
            "require_usable": require_usable,
            "require_should_use": require_should_use,
            "min_style_extractability": min_style_extractability,
            "min_negative_strength": min_negative_strength,
            "max_topic_leakage": max_topic_leakage,
            "max_model_familiarity": max_model_familiarity,
            "limit": limit,
            "sort_by": sort_by,
            "min_negatives_per_heldout": min_negatives_per_heldout,
            "require_gpt_selected_negatives": require_gpt_selected_negatives,
            "selected_pack_ids": [str(row["pack_id"]) for row in selected_audits],
            "audits": len(selected_audits),
            "packs": len(selected_packs),
            "private_eval_rows": len(selected_private),
            "hard_negatives": len(selected_negatives),
            "expected_task_refs": len(expected_task_refs),
            "task_refs_with_hard_negatives": len(covered_task_refs),
            "insufficient_negative_refs": insufficient_negative_refs,
            "dropped_insufficient_negative_refs": dropped_insufficient_negative_refs,
            "missing_pack_ids": missing_pack_ids,
        },
    }
def write_author_style_subset(out_dir: Path, subset: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "selected_quality_packs.jsonl", subset["quality_rows"])
    write_jsonl(out_dir / "accepted_author_style_packs.jsonl", subset["packs"])
    write_jsonl(out_dir / "accepted_author_style_private_eval.jsonl", subset["private_eval"])
    write_jsonl(out_dir / "accepted_hard_negatives.jsonl", subset["hard_negatives"])
    (out_dir / "selected_pack_ids.txt").write_text(
        "\n".join(subset["summary"]["selected_pack_ids"]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "selection_summary.json").write_text(
        json.dumps(subset["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_author_style_audit_subset(out_dir: Path, subset: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "gpt55_author_audits.jsonl", subset["audits"])
    write_jsonl(out_dir / "accepted_author_style_packs.jsonl", subset["packs"])
    write_jsonl(out_dir / "accepted_author_style_private_eval.jsonl", subset["private_eval"])
    write_jsonl(out_dir / "accepted_hard_negatives.jsonl", subset["hard_negatives"])
    (out_dir / "selected_pack_ids.txt").write_text(
        "\n".join(subset["summary"]["selected_pack_ids"]) + "\n",
        encoding="utf-8",
    )
    (out_dir / "selection_summary.json").write_text(
        json.dumps(subset["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_author_style_subset_inputs(
    *,
    quality_packs_path: Path,
    packs_path: Path,
    private_eval_path: Path,
    hard_negatives_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "quality_rows": load_jsonl(quality_packs_path),
        "packs": load_jsonl(packs_path),
        "private_eval": load_jsonl(private_eval_path),
        "hard_negatives": load_jsonl(hard_negatives_path),
    }
