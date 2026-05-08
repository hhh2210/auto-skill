"""Validation helpers for benchmark-derived few-shot split files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REQUIRED_SPLIT_FIELDS = (
    "split_id",
    "source",
    "learning_problem",
    "domain",
    "train_examples",
    "heldout_tasks",
)

REQUIRED_TASK_FIELDS = (
    "source",
    "source_id",
    "domain",
    "task_input",
    "supervision",
    "judge",
)


class ValidationError(ValueError):
    """Raised when a split artifact does not match the expected schema."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _missing_fields(record: dict[str, Any], required: Iterable[str]) -> list[str]:
    return [field for field in required if field not in record]


def validate_task(task: dict[str, Any], *, label: str) -> None:
    missing = _missing_fields(task, REQUIRED_TASK_FIELDS)
    if missing:
        raise ValidationError(f"{label}: missing fields: {', '.join(missing)}")
    if not str(task["task_input"]).strip():
        raise ValidationError(f"{label}: task_input must be non-empty")
    if not isinstance(task["domain"], dict):
        raise ValidationError(f"{label}: domain must be an object")
    if not isinstance(task["supervision"], dict):
        raise ValidationError(f"{label}: supervision must be an object")
    if not isinstance(task["judge"], dict):
        raise ValidationError(f"{label}: judge must be an object")


def validate_split(split: dict[str, Any]) -> None:
    missing = _missing_fields(split, REQUIRED_SPLIT_FIELDS)
    split_id = split.get("split_id", "<unknown>")
    if missing:
        raise ValidationError(f"{split_id}: missing fields: {', '.join(missing)}")
    if not isinstance(split["train_examples"], list) or not split["train_examples"]:
        raise ValidationError(f"{split_id}: train_examples must be a non-empty list")
    if not isinstance(split["heldout_tasks"], list) or not split["heldout_tasks"]:
        raise ValidationError(f"{split_id}: heldout_tasks must be a non-empty list")

    seen_in_split: dict[tuple[str, str], str] = {}
    for role in ("train_examples", "heldout_tasks"):
        for index, task in enumerate(split[role]):
            validate_task(task, label=f"{split_id}.{role}[{index}]")
            task_id = (task["source"], str(task["source_id"]))
            task_label = f"{role}[{index}]"
            previous = seen_in_split.get(task_id)
            if previous is not None:
                raise ValidationError(
                    f"{split_id}: duplicate task within split: "
                    f"{task_id} appears in {previous} and {task_label}"
                )
            seen_in_split[task_id] = task_label


def validate_splits(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValidationError("split file is empty")
    seen_split_ids = set()
    seen_tasks: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        validate_split(row)
        split_id = row["split_id"]
        if split_id in seen_split_ids:
            raise ValidationError(f"duplicate split_id: {split_id}")
        seen_split_ids.add(split_id)

        for role in ("train_examples", "heldout_tasks"):
            for task in row[role]:
                task_id = (task["source"], str(task["source_id"]))
                previous = seen_tasks.get(task_id)
                if previous is not None:
                    previous_split, previous_role = previous
                    raise ValidationError(
                        "duplicate task across splits: "
                        f"{task_id} appears in {previous_split}.{previous_role} "
                        f"and {split_id}.{role}"
                    )
                seen_tasks[task_id] = (split_id, role)


def summarize_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_splits": len(rows),
        "sources": {},
        "train_examples": 0,
        "heldout_tasks": 0,
    }
    for row in rows:
        source = row.get("source", "unknown")
        summary["sources"][source] = summary["sources"].get(source, 0) + 1
        summary["train_examples"] += len(row.get("train_examples", []))
        summary["heldout_tasks"] += len(row.get("heldout_tasks", []))
    return summary
