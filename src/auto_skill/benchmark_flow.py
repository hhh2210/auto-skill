"""Audit benchmark-derived artifacts against the documented benchmark flow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from auto_skill.data_cleaning import ValidationError, validate_splits
from auto_skill.example_packs import generation_prompt
from auto_skill.generated_outputs import private_leak_matches
from auto_skill.schemas import SchemaValidationError, validate_artifact_rows

PRIVATE_KEYS = {"supervision", "judge", "statistics"}
PRIVATE_KEY_PREFIXES = ("private_",)
ALLOWED_INDUCTION_FIELDS = {
    "train_examples.task_input",
    "train_examples.materials",
    "train_examples.desired_output.text",
    "optional user notes if added later",
}
REQUIRED_INDUCTION_FIELDS = {"train_examples.desired_output.text"}


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


def _split_task_by_source(
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


def _latest_generation_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        prompt_sha = str(row.get("prompt_sha256") or "")
        if job_id and prompt_sha:
            latest[(job_id, prompt_sha)] = row
    return latest


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_benchmark_flow(
    *,
    splits: list[dict[str, Any]],
    packs: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    generation_jobs: list[dict[str, Any]] | None = None,
    generated_rows: list[dict[str, Any]] | None = None,
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
    if generated_rows is not None:
        try:
            validate_artifact_rows(
                generated_rows,
                kind="generated_outputs",
                label="generated outputs",
            )
        except SchemaValidationError as exc:
            errors.append(f"generated outputs invalid: {exc}")

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
    jobs_by_id = {
        str(job.get("job_id")): job
        for job in generation_jobs or []
        if isinstance(job.get("job_id"), str)
    }
    latest_rows = _latest_generation_rows(generated_rows or [])

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
            allowed_set = {str(item) for item in allowed} if isinstance(allowed, list) else set()
            if not isinstance(allowed, list) or not REQUIRED_INDUCTION_FIELDS <= allowed_set:
                errors.append(
                    f"{pack_id}: input_boundary must expose only desired_output.text"
                )
            extra_allowed = allowed_set - ALLOWED_INDUCTION_FIELDS
            if extra_allowed:
                errors.append(
                    f"{pack_id}: input_boundary exposes non-canonical fields "
                    f"{sorted(extra_allowed)}"
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
        split_train_by_source = _split_task_by_source(split, "train_examples")

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
            if generation_jobs is not None:
                job_id = desired.get("generation_job_id")
                prompt_sha = desired.get("prompt_sha256")
                if not isinstance(job_id, str) or not job_id:
                    errors.append(f"{label}: generated desired_output lacks generation_job_id")
                elif job_id not in jobs_by_id:
                    errors.append(f"{label}: generation_job_id {job_id!r} is missing from jobs")
                else:
                    job = jobs_by_id[job_id]
                    if job.get("pack_id") != pack_id:
                        errors.append(f"{label}: generation job pack_id mismatch")
                    if job.get("example_id") != example.get("example_id"):
                        errors.append(f"{label}: generation job example_id mismatch")
                    if job.get("source") != example.get("source"):
                        errors.append(f"{label}: generation job source mismatch")
                    if job.get("source_task_id") != example.get("source_task_id"):
                        errors.append(f"{label}: generation job source_task_id mismatch")
                    if prompt_sha and job.get("prompt_sha256") != prompt_sha:
                        errors.append(f"{label}: generation job prompt_sha256 mismatch")
                    expected_template = desired.get("prompt_template_version")
                    if (
                        expected_template
                        and job.get("prompt_template_version") != expected_template
                    ):
                        errors.append(f"{label}: generation job prompt_template_version mismatch")
                    prompt = job.get("prompt")
                    if not isinstance(prompt, str) or not prompt.strip():
                        errors.append(f"{label}: generation job prompt is empty")
                    else:
                        source_task = split_train_by_source.get(
                            (str(example.get("source")), str(example.get("source_task_id")))
                        )
                        if source_task is not None:
                            expected_prompt = generation_prompt(
                                source_task,
                                example_id=str(example.get("example_id")),
                            )
                            if prompt != expected_prompt:
                                errors.append(
                                    f"{label}: generation job prompt does not match "
                                    "visible source task/materials"
                                )
                        actual_prompt_sha = _sha256_text(prompt)
                        if job.get("prompt_sha256") != actual_prompt_sha:
                            errors.append(
                                f"{label}: generation job prompt_sha256 does not match prompt"
                            )
                        prompt_leaks = private_leak_matches(prompt)
                        if prompt_leaks:
                            errors.append(
                                f"{label}: generation job prompt leaks private metadata "
                                f"{prompt_leaks}"
                            )
                if generated_rows is not None and isinstance(job_id, str) and job_id:
                    if not isinstance(prompt_sha, str) or not prompt_sha:
                        errors.append(f"{label}: generated desired_output lacks prompt_sha256")
                    else:
                        generated = latest_rows.get((job_id, prompt_sha))
                        if generated is None:
                            errors.append(f"{label}: missing generated output row for {job_id!r}")
                        else:
                            if generated.get("pack_id") != pack_id:
                                errors.append(f"{label}: generated output pack_id mismatch")
                            if generated.get("example_id") != example.get("example_id"):
                                errors.append(f"{label}: generated output example_id mismatch")
                            if generated.get("source") != example.get("source"):
                                errors.append(f"{label}: generated output source mismatch")
                            if generated.get("source_task_id") != example.get("source_task_id"):
                                errors.append(
                                    f"{label}: generated output source_task_id mismatch"
                                )
                            expected_template = desired.get("prompt_template_version")
                            if (
                                expected_template
                                and generated.get("prompt_template_version")
                                != expected_template
                            ):
                                errors.append(
                                    f"{label}: generated output "
                                    "prompt_template_version mismatch"
                                )
                            if generated.get("status") != "success":
                                errors.append(
                                    f"{label}: latest generated output is "
                                    f"{generated.get('status')!r}"
                                )
                            if generated.get("finish_reason") != "stop":
                                errors.append(
                                    f"{label}: generated output finish_reason is not stop"
                                )
                            if generated.get("desired_output") != text:
                                errors.append(
                                    f"{label}: frozen desired_output text does not match "
                                    "generated output row"
                                )

        for index, task in enumerate(pack.get("heldout_tasks", [])):
            label = f"{pack_id}.heldout_tasks[{index}]"
            if task.get("desired_output") is not None:
                errors.append(f"{label}: heldout task must not include desired_output")

    if generation_jobs is not None:
        expected_job_ids = {
            example.get("desired_output", {}).get("generation_job_id")
            for pack in packs
            for example in pack.get("train_examples", [])
            if isinstance(example.get("desired_output"), dict)
            and example["desired_output"].get("generation_job_id")
        }
        orphan_jobs = set(jobs_by_id) - {str(job_id) for job_id in expected_job_ids}
        if orphan_jobs:
            warnings.append(f"generation jobs have no frozen example: {sorted(orphan_jobs)[:5]}")

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
