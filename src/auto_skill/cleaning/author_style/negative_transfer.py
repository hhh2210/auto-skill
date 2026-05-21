"""Build public wrong-author packs for author-style negative transfer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from typing import Any


def _stable_float(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def public_tags_for_task(task: dict[str, Any]) -> set[str]:
    metadata = task.get("metadata")
    if not isinstance(metadata, dict):
        return set()
    tags = metadata.get("public_content_tags")
    if not isinstance(tags, list):
        return set()
    return {str(tag).lower() for tag in tags if str(tag).strip()}


def public_tags_for_examples(pack: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    for example in pack.get("train_examples", []):
        if isinstance(example, dict):
            tags.update(public_tags_for_task(example))
    return tags


def mean_train_words(pack: dict[str, Any]) -> float:
    counts = []
    for example in pack.get("train_examples", []):
        metadata = example.get("metadata") if isinstance(example, dict) else None
        word_count = metadata.get("word_count") if isinstance(metadata, dict) else None
        if isinstance(word_count, (int, float)) and not isinstance(word_count, bool):
            counts.append(float(word_count))
    return sum(counts) / len(counts) if counts else 0.0


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def rewrite_impostor_examples(
    *,
    target_pack_id: str,
    impostor_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    rewritten = []
    for index, example in enumerate(impostor_pack.get("train_examples", [])):
        new_example = copy.deepcopy(example)
        new_example["example_id"] = f"{target_pack_id}::wrong_author_train::{index}"
        metadata = new_example.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["negative_transfer_source_pack_id"] = impostor_pack.get("pack_id")
            metadata["negative_transfer_train_role"] = "wrong_author_example"
        rewritten.append(new_example)
    return rewritten


def select_impostor_pack(
    *,
    target_pack: dict[str, Any],
    candidates: list[dict[str, Any]],
    usage_counts: Counter[str],
    seed: int,
    max_impostor_reuse_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_pack_id = str(target_pack.get("pack_id"))
    target_task = next(iter(target_pack.get("heldout_tasks", [])), {})
    target_tags = public_tags_for_task(target_task)
    target_train_words = mean_train_words(target_pack)
    scored = []
    for candidate in candidates:
        candidate_pack_id = str(candidate.get("pack_id"))
        if candidate_pack_id == target_pack_id:
            continue
        if (
            max_impostor_reuse_count is not None
            and usage_counts[candidate_pack_id] >= max_impostor_reuse_count
        ):
            continue
        candidate_tags = public_tags_for_examples(candidate)
        tag_overlap = jaccard(target_tags, candidate_tags)
        candidate_train_words = mean_train_words(candidate)
        if target_train_words > 0 and candidate_train_words > 0:
            length_log_ratio = abs(math.log(candidate_train_words / target_train_words))
        else:
            length_log_ratio = 1.0
        source_mismatch = 0 if candidate.get("source") == target_pack.get("source") else 1
        reuse_penalty = usage_counts[candidate_pack_id] * 0.03
        tie_break = _stable_float(f"{seed}:{target_pack_id}:{candidate_pack_id}") * 0.001
        score = (
            source_mismatch * 10
            + tag_overlap * 3
            + length_log_ratio
            + reuse_penalty
            + tie_break
        )
        scored.append(
            (
                score,
                candidate,
                {
                    "target_pack_id": target_pack_id,
                    "impostor_pack_id": candidate_pack_id,
                    "source_mismatch": source_mismatch,
                    "heldout_to_impostor_train_tag_jaccard": round(tag_overlap, 4),
                    "target_mean_train_words": round(target_train_words, 3),
                    "impostor_mean_train_words": round(candidate_train_words, 3),
                    "length_log_ratio": round(length_log_ratio, 4),
                    "previous_impostor_reuse_count": usage_counts[candidate_pack_id],
                    "selection_score": round(score, 6),
                },
            )
        )
    if not scored:
        raise ValueError(f"no impostor candidate for pack {target_pack_id}")
    scored.sort(key=lambda item: item[0])
    _, candidate, metrics = scored[0]
    usage_counts[str(candidate.get("pack_id"))] += 1
    return candidate, metrics


def build_negative_transfer_packs(
    packs: list[dict[str, Any]],
    *,
    seed: int = 20260518,
    policy: str = "same_source_topic_disjoint_length_matched",
    max_impostor_reuse_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_impostor_reuse_count is not None and max_impostor_reuse_count <= 0:
        raise ValueError("max_impostor_reuse_count must be positive when set")
    usage_counts: Counter[str] = Counter()
    output_packs = []
    pair_rows = []
    for target_pack in packs:
        target_pack_id = str(target_pack.get("pack_id"))
        impostor_pack, metrics = select_impostor_pack(
            target_pack=target_pack,
            candidates=packs,
            usage_counts=usage_counts,
            seed=seed,
            max_impostor_reuse_count=max_impostor_reuse_count,
        )
        new_pack = copy.deepcopy(target_pack)
        new_pack["train_examples"] = rewrite_impostor_examples(
            target_pack_id=target_pack_id,
            impostor_pack=impostor_pack,
        )
        new_pack["negative_transfer"] = {
            "schema_version": "author-style-negative-transfer/v1",
            "policy": policy,
            "target_pack_id": target_pack_id,
            "impostor_pack_id": impostor_pack.get("pack_id"),
            "same_source": impostor_pack.get("source") == target_pack.get("source"),
            "used_private_text_for_selection": False,
            "used_private_eval_for_selection": False,
            "used_hard_negatives_for_selection": False,
            "used_cluster_tags_for_selection": False,
            "must_not_use_for_induction": True,
            "max_impostor_reuse_count_allowed": max_impostor_reuse_count,
        }
        output_packs.append(new_pack)
        pair_rows.append({**new_pack["negative_transfer"], **metrics})

    manifest = {
        "schema_version": "author-style-negative-transfer-manifest/v1",
        "policy": policy,
        "seed": seed,
        "selection_algorithm": "greedy_sorted_by_source_tag_length_reuse_tiebreak",
        "max_impostor_reuse_count_allowed": max_impostor_reuse_count,
        "pack_count": len(output_packs),
        "artifact_boundary": {
            "must_not_use_for_induction": True,
            "may_use_for": [
                "negative_transfer_generation",
                "wrong_author_control_eval",
                "post_hoc_error_analysis",
            ],
            "must_not_use_for": [
                "skill_prompt",
                "target_author_training",
                "hard_negative_rerank",
            ],
            "selection_inputs": [
                "public pack source",
                "public train word counts",
                "public heldout content tags",
                "public impostor train content tags",
            ],
            "uses_private_text": False,
            "uses_private_eval": False,
            "uses_hard_negative_text": False,
            "uses_cluster_tags": False,
        },
        "pair_count": len(pair_rows),
        "unique_impostor_pack_count": len({row["impostor_pack_id"] for row in pair_rows}),
        "reuse_histogram": {
            str(reuse_count): sum(1 for count in usage_counts.values() if count == reuse_count)
            for reuse_count in sorted(set(usage_counts.values()))
        },
        "source_mismatch_count": sum(row["source_mismatch"] for row in pair_rows),
        "nonzero_tag_overlap_count": sum(
            1 for row in pair_rows if row["heldout_to_impostor_train_tag_jaccard"] > 0
        ),
        "mean_heldout_to_impostor_train_tag_jaccard": (
            round(
                sum(row["heldout_to_impostor_train_tag_jaccard"] for row in pair_rows)
                / len(pair_rows),
                4,
            )
            if pair_rows
            else None
        ),
        "p90_heldout_to_impostor_train_tag_jaccard": (
            round(
                percentile(
                    [row["heldout_to_impostor_train_tag_jaccard"] for row in pair_rows],
                    0.90,
                )
                or 0.0,
                4,
            )
            if pair_rows
            else None
        ),
        "max_heldout_to_impostor_train_tag_jaccard": (
            max(row["heldout_to_impostor_train_tag_jaccard"] for row in pair_rows)
            if pair_rows
            else None
        ),
        "mean_length_log_ratio": (
            round(sum(row["length_log_ratio"] for row in pair_rows) / len(pair_rows), 4)
            if pair_rows
            else None
        ),
        "p90_length_log_ratio": (
            round(
                percentile([row["length_log_ratio"] for row in pair_rows], 0.90) or 0.0,
                4,
            )
            if pair_rows
            else None
        ),
        "max_length_log_ratio": (
            max(row["length_log_ratio"] for row in pair_rows) if pair_rows else None
        ),
        "max_impostor_reuse_count": max(usage_counts.values()) if usage_counts else 0,
        "pairs": pair_rows,
    }
    # Assert manifest JSON serializability at construction time.
    json.dumps(manifest, ensure_ascii=False)
    return output_packs, manifest
