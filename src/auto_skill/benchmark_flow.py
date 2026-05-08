"""Audit benchmark-derived artifacts against the documented benchmark flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auto_skill.data_cleaning import ValidationError, validate_splits
from auto_skill.generated_outputs import private_leak_matches

PRIVATE_KEYS = {"supervision", "judge", "statistics"}
PRIVATE_KEY_PREFIXES = ("private_",)


@dataclass(frozen=True)
class FlowAuditResult:
    """Result of checking data artifacts against the benchmark-flow contract."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _task_ids(tasks: list[dict[str, Any]], id_key: str) -> set[str]:
    return {str(task.get(id_key)) for task in tasks if task.get(id_key)}


def _source_task_id(task: dict[str, Any]) -> tuple[str, str] | None:
    source = task.get("source")
    source_task_id = task.get("source_task_id")
    if source is None or source_task_id is None:
        return None
    return (str(source), str(source_task_id))


def _source_task_map(tasks: list[dict[str, Any]], id_key: str) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for task in tasks:
        visible_id = task.get(id_key)
        source_task = _source_task_id(task)
        if visible_id and source_task is not None:
            mapping[str(visible_id)] = source_task
    return mapping


def _split_task_map(
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


def _find_private_keys(value: Any, *, path: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            has_private_prefix = any(
                key.startswith(prefix) for prefix in PRIVATE_KEY_PREFIXES
            )
            if key in PRIVATE_KEYS or has_private_prefix:
                matches.append(child_path)
            matches.extend(_find_private_keys(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_find_private_keys(child, path=f"{path}[{index}]"))
    return matches


def _private_task_map(tasks: list[dict[str, Any]]) -> dict[str, tuple[str, str] | None]:
    mapping: dict[str, tuple[str, str] | None] = {}
    for task in tasks:
        task_ref = task.get("task_ref")
        if task_ref:
            mapping[str(task_ref)] = _source_task_id(task)
    return mapping


def _pack_private_refs(
    private_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, tuple[str, str] | None]]]:
    refs: dict[str, dict[str, dict[str, tuple[str, str] | None]]] = {}
    for row in private_rows:
        pack_id = str(row.get("pack_id") or "")
        if not pack_id:
            continue
        refs[pack_id] = {
            "train": _private_task_map(row.get("train_private", [])),
            "heldout": _private_task_map(row.get("heldout_private", [])),
        }
    return refs


def audit_benchmark_flow(
    *,
    splits: list[dict[str, Any]],
    packs: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
) -> FlowAuditResult:
    """Validate the example-driven benchmark flow boundary.

    This intentionally checks artifact-level invariants rather than benchmark
    quality. It answers whether the checked-in cleaned artifacts preserve the
    documented separation between user-visible examples, heldout evaluation, and
    private benchmark metadata.
    """

    errors: list[str] = []
    warnings: list[str] = []

    try:
        validate_splits(splits)
    except ValidationError as exc:
        errors.append(f"splits invalid: {exc}")

    split_by_id = {
        str(row.get("split_id")): row for row in splits if isinstance(row.get("split_id"), str)
    }
    split_sources = {str(row.get("source")) for row in splits}
    missing_sources = {"WritingBench", "PresentBench"} - split_sources
    if missing_sources:
        errors.append(f"splits missing benchmark sources: {sorted(missing_sources)}")

    if not packs:
        errors.append("example packs are empty")
    if not private_rows:
        errors.append("private eval rows are empty")

    pack_ids = [str(pack.get("pack_id") or "") for pack in packs]
    if len(pack_ids) != len(set(pack_ids)):
        errors.append("duplicate pack_id in example packs")
    private_ids = [str(row.get("pack_id") or "") for row in private_rows]
    if len(private_ids) != len(set(private_ids)):
        errors.append("duplicate pack_id in private eval rows")
    private_by_pack = _pack_private_refs(private_rows)
    private_row_by_pack = {
        str(row.get("pack_id")): row
        for row in private_rows
        if isinstance(row.get("pack_id"), str)
    }
    private_pack_ids = set(private_by_pack)

    for pack in packs:
        pack_id = str(pack.get("pack_id") or "<unknown>")
        split_id = str(pack.get("split_id") or "")
        split = split_by_id.get(split_id)
        if split is None:
            errors.append(f"{pack_id}: split_id {split_id!r} does not exist in splits")
        else:
            if pack.get("source") != split.get("source"):
                errors.append(f"{pack_id}: source does not match split {split_id}")
            if pack.get("domain") != split.get("domain"):
                errors.append(f"{pack_id}: domain does not match split {split_id}")
        private_paths = _find_private_keys(pack)
        if private_paths:
            errors.append(f"{pack_id}: user-visible pack contains private keys {private_paths}")

        boundary = pack.get("input_boundary")
        if not isinstance(boundary, dict):
            errors.append(f"{pack_id}: missing input_boundary contract")
        else:
            forbidden = boundary.get("must_not_use_for_induction")
            if not isinstance(forbidden, list) or not forbidden:
                errors.append(f"{pack_id}: input_boundary lacks must_not_use_for_induction")
            allowed = boundary.get("auto_skill_module_can_use")
            if not isinstance(allowed, list) or "train_examples.desired_output.text" not in {
                str(item) for item in allowed
            }:
                errors.append(
                    f"{pack_id}: input_boundary must expose only desired_output.text"
                )

        train_examples = pack.get("train_examples", [])
        heldout_tasks = pack.get("heldout_tasks", [])
        train_ids = _task_ids(train_examples, "example_id")
        heldout_ids = _task_ids(heldout_tasks, "task_id")
        overlap = train_ids & heldout_ids
        if overlap:
            errors.append(f"{pack_id}: train/heldout ids overlap: {sorted(overlap)}")
        train_source_tasks = _source_task_map(train_examples, "example_id")
        heldout_source_tasks = _source_task_map(heldout_tasks, "task_id")
        source_overlap = set(train_source_tasks.values()) & set(heldout_source_tasks.values())
        if source_overlap:
            errors.append(
                f"{pack_id}: train/heldout source tasks overlap: {sorted(source_overlap)}"
            )
        if split is not None:
            split_train_sources = _split_task_map(split, "train_examples")
            split_heldout_sources = _split_task_map(split, "heldout_tasks")
            if set(train_source_tasks.values()) != split_train_sources:
                errors.append(f"{pack_id}: train source tasks do not match split {split_id}")
            if set(heldout_source_tasks.values()) != split_heldout_sources:
                errors.append(f"{pack_id}: heldout source tasks do not match split {split_id}")

        if pack_id not in private_pack_ids:
            errors.append(f"{pack_id}: missing private eval row")
        else:
            private_row = private_row_by_pack.get(pack_id, {})
            if private_row.get("split_id") != pack.get("split_id"):
                errors.append(f"{pack_id}: private row split_id does not match visible pack")
            if private_row.get("source") != pack.get("source"):
                errors.append(f"{pack_id}: private row source does not match visible pack")
            refs = private_by_pack[pack_id]
            if set(refs["train"]) != train_ids:
                errors.append(
                    f"{pack_id}: private train refs do not match visible train examples"
                )
            if set(refs["heldout"]) != heldout_ids:
                errors.append(
                    f"{pack_id}: private heldout refs do not match visible heldout tasks"
                )
            for task_ref, private_source_task in refs["train"].items():
                if private_source_task != train_source_tasks.get(task_ref):
                    errors.append(
                        f"{pack_id}: private train source id mismatch for {task_ref}"
                    )
            for task_ref, private_source_task in refs["heldout"].items():
                if private_source_task != heldout_source_tasks.get(task_ref):
                    errors.append(
                        f"{pack_id}: private heldout source id mismatch for {task_ref}"
                    )

        for index, example in enumerate(pack.get("train_examples", [])):
            label = f"{pack_id}.train_examples[{index}]"
            desired = example.get("desired_output")
            if not isinstance(desired, dict) or desired.get("status") != "generated":
                errors.append(f"{label}: desired_output is not frozen/generated")
                continue
            text = desired.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{label}: generated desired_output text is empty")
                continue
            leaks = private_leak_matches(text)
            if leaks:
                errors.append(f"{label}: desired_output leaks private metadata {leaks}")

        for index, task in enumerate(pack.get("heldout_tasks", [])):
            label = f"{pack_id}.heldout_tasks[{index}]"
            if task.get("desired_output") is not None:
                errors.append(f"{label}: heldout task must not include desired_output")

    for row in private_rows:
        pack_id = str(row.get("pack_id") or "<unknown>")
        split_id = str(row.get("split_id") or "")
        if split_id not in split_by_id:
            errors.append(f"{pack_id}: private row split_id {split_id!r} does not exist")
        for role in ("train_private", "heldout_private"):
            private_tasks = row.get(role, [])
            if not isinstance(private_tasks, list):
                errors.append(f"{pack_id}: {role} must be a list")
                continue
            task_refs = [
                str(task.get("task_ref"))
                for task in private_tasks
                if isinstance(task, dict) and task.get("task_ref")
            ]
            if len(task_refs) != len(set(task_refs)):
                errors.append(f"{pack_id}.{role}: duplicate task_ref")
            for index, task in enumerate(private_tasks):
                if "supervision" not in task or "judge" not in task:
                    errors.append(f"{pack_id}.{role}[{index}]: missing supervision or judge")

    orphan_private = private_pack_ids - set(pack_ids)
    if orphan_private:
        warnings.append(f"private eval rows have no visible pack: {sorted(orphan_private)}")

    return FlowAuditResult(errors=tuple(errors), warnings=tuple(warnings))
