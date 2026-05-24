"""JSON record builders for author-style probe artifacts."""

from __future__ import annotations

from typing import Any

from auto_skill.probes.author_style.negatives import match_features
from auto_skill.probes.author_style.sampling import bucket


def public_pack(
    pack_id: str,
    corpus: str,
    task_id: str,
    target: dict[str, Any],
    reference: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    source = "personal_blog_history" if corpus == "blog" else "cross_topic_online_comment_history"
    return {
        "schema_version": "author-style-pack/v1",
        "pack_id": pack_id,
        "source": source,
        "train_examples": [
            public_post(
                reference,
                row_id=f"{pack_id}::train::0",
                row_key="example_id",
                include_output=True,
            )
        ],
        "heldout_tasks": [
            public_post(target, row_id=task_id, row_key="task_id", include_output=False)
        ],
        "probe_metadata": {
            "corpus": corpus,
            "hard_neg_variant": variant,
            "length_bucket": bucket(int(target["word_count"])),
        },
    }


def public_post(
    row: dict[str, Any],
    *,
    row_id: str,
    row_key: str,
    include_output: bool,
) -> dict[str, Any]:
    post = {
        "schema_version": "author-style-pack/v1",
        row_key: row_id,
        "source_task_id": row["source_id"],
        "task_input": "Write in the same anonymous author's style.",
        "metadata": {
            "public_content_tags": row.get("content_tags", [])[:5],
            "word_count": row.get("word_count"),
        },
    }
    post["desired_output"] = (
        {
            "status": "generated",
            "text": row["text"],
            "source": "original_user_history",
            "provenance": "source_provided",
        }
        if include_output
        else None
    )
    return post


def private_eval_row(pack_id: str, task_id: str, target: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "author-style-pack/v1",
        "pack_id": pack_id,
        "author_hash": target["author_hash"],
        "source": target["source"],
        "heldout_private": [
            {
                "task_ref": task_id,
                "source_task_id": target["source_id"],
                "private_topic": target.get("private_topic"),
                "word_count": target.get("word_count"),
                "reference_output_private": target["text"],
            }
        ],
    }


def negative_row(
    pack_id: str,
    task_id: str,
    rank: int,
    target: dict[str, Any],
    negative: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    return {
        "schema_version": "author-style-hard-negative/v1",
        "negative_id": f"{pack_id}::negative::{rank}",
        "target_task_ref": task_id,
        "negative_type": f"{variant}_other_author",
        "public_negative_text": negative["text"],
        "match_features": match_features(target, negative),
        "private_label": "different_author",
        "must_not_use_for_induction": True,
    }
