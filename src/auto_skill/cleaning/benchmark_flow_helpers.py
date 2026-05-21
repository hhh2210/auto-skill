"""Helper functions for benchmark-flow artifact audits."""

from __future__ import annotations

import hashlib
from typing import Any

PRIVATE_KEYS = {"supervision", "judge", "statistics"}
PRIVATE_KEY_PREFIXES = ("private_",)
ALLOWED_INDUCTION_FIELDS = {
    "train_examples.task_input",
    "train_examples.materials",
    "train_examples.desired_output.text",
}
REQUIRED_INDUCTION_FIELDS = {"train_examples.desired_output.text"}


def task_ids(tasks: list[dict[str, Any]], id_key: str) -> set[str]:
    return {str(task.get(id_key)) for task in tasks if task.get(id_key)}


def source_task_id(task: dict[str, Any]) -> tuple[str, str] | None:
    source = task.get("source")
    source_task_id = task.get("source_task_id")
    if source is None or source_task_id is None:
        return None
    return (str(source), str(source_task_id))


def source_task_map(tasks: list[dict[str, Any]], id_key: str) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for task in tasks:
        visible_id = task.get(id_key)
        source_task = source_task_id(task)
        if visible_id and source_task is not None:
            mapping[str(visible_id)] = source_task
    return mapping


def split_task_map(
    split: dict[str, Any],
    role: str,
) -> set[tuple[str, str]]:
    values = set()
    for task in split.get(role, []):
        source = task.get("source")
        source_id = task.get("source_id")
        if source is not None and source_id is not None:
            values.add((str(source), str(source_id)))
    return values


def split_task_by_source(
    split: dict[str, Any] | None,
    role: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if split is None:
        return {}
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for task in split.get(role, []):
        source = task.get("source")
        source_id = task.get("source_id")
        if source is not None and source_id is not None:
            mapping[(str(source), str(source_id))] = task
    return mapping


def find_private_keys(value: Any, *, path: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            has_private_prefix = any(
                key.startswith(prefix) for prefix in PRIVATE_KEY_PREFIXES
            )
            if key in PRIVATE_KEYS or has_private_prefix:
                matches.append(child_path)
            matches.extend(find_private_keys(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(find_private_keys(child, path=f"{path}[{index}]"))
    return matches


def private_task_map(tasks: list[dict[str, Any]]) -> dict[str, tuple[str, str] | None]:
    mapping: dict[str, tuple[str, str] | None] = {}
    for task in tasks:
        task_ref = task.get("task_ref")
        if task_ref:
            mapping[str(task_ref)] = source_task_id(task)
    return mapping


def pack_private_refs(
    private_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, tuple[str, str] | None]]]:
    refs: dict[str, dict[str, dict[str, tuple[str, str] | None]]] = {}
    for row in private_rows:
        pack_id = str(row.get("pack_id") or "")
        if not pack_id:
            continue
        refs[pack_id] = {
            "train": private_task_map(row.get("train_private", [])),
            "heldout": private_task_map(row.get("heldout_private", [])),
        }
    return refs


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
