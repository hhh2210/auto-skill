"""Prompt builders and rerank parsing for author-style GPT helpers."""

from __future__ import annotations

import json
from typing import Any


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


def build_gpt_audit_prompt(
    pack: dict[str, Any],
    negatives: list[dict[str, Any]],
    private_eval: dict[str, Any] | None = None,
) -> str:
    examples = []
    for example in pack["train_examples"]:
        examples.append(
            {
                "content_tags": example["metadata"]["public_content_tags"],
                "text": truncate(example["desired_output"]["text"], 900),
            }
        )
    heldout = []
    heldout_private_by_task = {
        row.get("task_ref"): row
        for row in (private_eval or {}).get("heldout_private", [])
        if isinstance(row, dict)
    }
    for task in pack["heldout_tasks"]:
        private_task = heldout_private_by_task.get(task.get("task_id")) or {}
        text = private_task.get("reference_output_private")
        if not isinstance(text, str) or not text.strip():
            continue
        heldout.append(
            {
                "content_tags": task["metadata"]["public_content_tags"],
                "text": truncate(text, 650),
            }
        )
    negative_samples = [
        {
            "negative_type": row["negative_type"],
            "content_tags": row["public_negative_content_tags"],
            "text": truncate(row["public_negative_text"], 650),
            "match_features": {
                key: value
                for key, value in row["match_features"].items()
                if not key.endswith("_private")
            },
        }
        for row in negatives[:3]
    ]
    payload = {
        "candidate_cluster_tags": (private_eval or {}).get("cluster_tags"),
        "style_summary": (private_eval or {}).get("style_summary"),
        "train_examples": examples,
        "heldout_samples": heldout,
        "hard_negative_samples": negative_samples,
    }
    return (
        "Audit this anonymous personal-writing style candidate for an auto-skill "
        "benchmark. Judge whether the examples have a stable, learnable personal "
        "style signal beyond topic. Also judge whether the negatives are strong. "
        "Do not identify the person. Return compact JSON with keys: usable "
        "(boolean), style_extractability_1_to_5, style_cluster_tags (array of "
        "short snake_case tags), topic_leakage_risk_1_to_5, model_familiarity_risk_1_to_5, "
        "negative_strength_1_to_5, should_use_for_smoke (boolean), evidence "
        "(array), reject_reasons (array).\n\n" + json.dumps(payload, ensure_ascii=False)
    )


def visible_match_features(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.get("match_features", {}).items()
        if not key.endswith("_private")
    }


def private_heldout_by_task(private_eval: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("task_ref")): row
        for row in (private_eval or {}).get("heldout_private", [])
        if isinstance(row, dict) and row.get("task_ref")
    }


def build_negative_rerank_prompt(
    pack: dict[str, Any],
    private_eval: dict[str, Any],
    target_task_ref: str,
    candidates: list[dict[str, Any]],
    *,
    candidate_chars: int,
) -> str:
    train_examples = [
        {
            "example_id": example.get("example_id"),
            "content_tags": example.get("metadata", {}).get("public_content_tags", []),
            "text": truncate(example["desired_output"]["text"], 520),
        }
        for example in pack.get("train_examples", [])
        if isinstance(example.get("desired_output"), dict)
        and isinstance(example["desired_output"].get("text"), str)
    ]
    private_heldout = private_heldout_by_task(private_eval).get(target_task_ref, {})
    heldout_text = str(private_heldout.get("reference_output_private") or "")
    payload = {
        "target_author_train_examples": train_examples,
        "target_heldout_reference": truncate(heldout_text, 620),
        "candidate_negatives": [
            {
                "candidate_id": row["negative_id"],
                "content_tags": row.get("public_negative_content_tags", []),
                "match_features": visible_match_features(row),
                "text": truncate(row["public_negative_text"], candidate_chars),
            }
            for row in candidates
        ],
    }
    return (
        "Select hard negative examples for an anonymous author-style benchmark.\n"
        "Goal: choose different-author candidates that would be hard to reject "
        "using style alone. Prefer candidates with similar punctuation rhythm, "
        "sentence shape, register, spelling habits, discourse moves, and era. "
        "Do not prefer candidates merely because they mention the same topic, "
        "people, place, date, or event. Reject candidates that are copied text, "
        "lyrics, templates, list memes, metadata/header artifacts, or obviously "
        "different genre/register.\n\n"
        "Return compact JSON only with key `ranked_negatives`, an array of objects: "
        "{candidate_id, style_confusability_1_to_5, topic_shortcut_risk_1_to_5, "
        "reject, reason}. Include every candidate_id exactly once, ordered from "
        "hardest to easiest negative.\n\n" + json.dumps(payload, ensure_ascii=False)
    )


def rerank_candidates_from_json(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = parsed.get("ranked_negatives")
    if isinstance(ranked, list):
        rows = [row for row in ranked if isinstance(row, dict)]
        if rows:
            return rows
    ids = parsed.get("ranked_candidate_ids")
    if isinstance(ids, list):
        return [{"candidate_id": str(candidate_id)} for candidate_id in ids]
    return []


def select_reranked_negatives(
    candidates: list[dict[str, Any]],
    parsed: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_id = {row["negative_id"]: row for row in candidates}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, item in enumerate(rerank_candidates_from_json(parsed), start=1):
        candidate_id = str(item.get("candidate_id") or item.get("negative_id") or item.get("id"))
        if candidate_id in seen or candidate_id not in by_id:
            continue
        seen.add(candidate_id)
        if item.get("reject") is True:
            continue
        row = dict(by_id[candidate_id])
        row["prefilter_negative_id"] = row["negative_id"]
        row["negative_id"] = f"{row['target_task_ref']}::negative::{len(selected) + 1}"
        row["gpt_rerank"] = {
            "status": "selected",
            "rank": rank,
            "style_confusability_1_to_5": item.get("style_confusability_1_to_5"),
            "topic_shortcut_risk_1_to_5": item.get("topic_shortcut_risk_1_to_5"),
            "reason": item.get("reason"),
        }
        selected.append(row)
        if len(selected) >= limit:
            break
    if len(selected) >= limit:
        return selected
    for row in candidates:
        if row["negative_id"] in seen:
            continue
        fallback = dict(row)
        fallback["prefilter_negative_id"] = fallback["negative_id"]
        fallback["negative_id"] = f"{fallback['target_task_ref']}::negative::{len(selected) + 1}"
        fallback["gpt_rerank"] = {"status": "fallback_prefilter"}
        selected.append(fallback)
        if len(selected) >= limit:
            break
    return selected
