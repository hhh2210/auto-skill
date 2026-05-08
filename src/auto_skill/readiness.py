"""Experiment readiness checks for the auto-skill prototype."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from auto_skill.example_packs import load_jsonl
from auto_skill.mvp import user_examples_from_pack
from auto_skill.schemas import (
    SchemaValidationError,
    validate_eval_row,
    validate_skill_row,
)

DEFAULT_SKILL_MODES = (
    "one_shot_skill_from_examples",
    "auto_skill_feature_driven_no_validation",
    "auto_skill_ours_full",
)

DEFAULT_EVAL_MODES = (
    "prompt_only",
    "few_shot_examples_only",
    "one_shot_skill_from_examples",
    "ours_no_validation",
    "auto_skill",
)


def load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def pack_index(packs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(pack.get("pack_id")): pack for pack in packs if pack.get("pack_id")}


def schema_validation_errors(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    label: str,
) -> list[str]:
    validators = {
        "skill": validate_skill_row,
        "eval": validate_eval_row,
    }
    validator = validators[kind]
    errors = []
    for index, row in enumerate(rows, start=1):
        try:
            validator(row, label=f"{label}:{index}")
        except SchemaValidationError as exc:
            errors.append(str(exc))
    return errors


def generated_example_counts(packs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {}
    for pack in packs:
        pack_id = str(pack.get("pack_id"))
        train_examples = pack.get("train_examples") or []
        generated = 0
        needs_generation = 0
        invalid = 0
        for example in train_examples:
            desired = example.get("desired_output") if isinstance(example, dict) else None
            if not isinstance(desired, dict):
                invalid += 1
                continue
            if desired.get("status") == "generated" and isinstance(desired.get("text"), str):
                generated += 1
            elif desired.get("status") == "needs_generation":
                needs_generation += 1
            else:
                invalid += 1
        counts[pack_id] = {
            "train_examples": len(train_examples),
            "generated_desired_outputs": generated,
            "needs_generation": needs_generation,
            "invalid_desired_outputs": invalid,
        }
    return counts


def skill_coverage(
    skill_rows: list[dict[str, Any]],
    *,
    required_modes: tuple[str, ...] = DEFAULT_SKILL_MODES,
) -> dict[str, dict[str, Any]]:
    by_pack: dict[str, dict[str, Any]] = defaultdict(lambda: {"modes": {}, "failures": []})
    for row in skill_rows:
        pack_id = str(row.get("pack_id") or "")
        if not pack_id:
            continue
        if row.get("status") == "induction_error":
            by_pack[pack_id]["failures"].append(row)
            continue
        mode = row.get("mode")
        skill_md = row.get("skill_md")
        if isinstance(mode, str) and isinstance(skill_md, str) and skill_md.strip():
            by_pack[pack_id]["modes"][mode] = {
                "chars": len(skill_md),
                "model_calls": len(row.get("model_calls") or []),
            }
    result = {}
    for pack_id, value in by_pack.items():
        missing = [mode for mode in required_modes if mode not in value["modes"]]
        result[pack_id] = {
            "modes": value["modes"],
            "missing_modes": missing,
            "failures": value["failures"],
            "ready": not missing and not value["failures"],
        }
    return result


def eval_coverage(
    rows: list[dict[str, Any]],
    *,
    required_modes: tuple[str, ...] = DEFAULT_EVAL_MODES,
) -> dict[str, dict[str, Any]]:
    by_pack_task: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"modes": {}, "status_counts": Counter()}
    )
    for row in rows:
        pack_id = str(row.get("pack_id") or "")
        task_id = str(row.get("task_id") or "")
        mode = str(row.get("mode") or "")
        if not pack_id or not task_id or not mode:
            continue
        status = str(row.get("status") or "unknown")
        item = by_pack_task[(pack_id, task_id)]
        item["modes"][mode] = status
        item["status_counts"][status] += 1

    result = {}
    for (pack_id, task_id), item in by_pack_task.items():
        missing = [mode for mode in required_modes if mode not in item["modes"]]
        non_success = {
            mode: status for mode, status in item["modes"].items() if status != "success"
        }
        key = f"{pack_id}::{task_id}"
        result[key] = {
            "pack_id": pack_id,
            "task_id": task_id,
            "modes": item["modes"],
            "missing_modes": missing,
            "non_success_modes": non_success,
            "status_counts": dict(item["status_counts"]),
            "ready": not missing and not non_success,
        }
    return result


def collect_models(rows: list[dict[str, Any]], *, keys: tuple[str, ...]) -> set[str]:
    """Collect non-empty model identifiers from row top-level fields."""

    models: set[str] = set()
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                models.add(value)
    return models


def model_inventory(
    *,
    skill_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Inventory of solver/judge models seen across skill and eval artifacts."""

    skill_solvers = collect_models(skill_rows, keys=("solver_model",))
    eval_solvers = collect_models(eval_rows, keys=("solver_model",))
    eval_judges = collect_models(eval_rows, keys=("judge_model",))
    union = skill_solvers | eval_solvers | eval_judges
    return {
        "skill_solver_models": sorted(skill_solvers),
        "eval_solver_models": sorted(eval_solvers),
        "eval_judge_models": sorted(eval_judges),
        "all_models": sorted(union),
        "monoculture": len(union) == 1 if union else False,
        "monoculture_model": next(iter(union)) if len(union) == 1 else None,
    }


def expected_eval_targets(
    packs: list[dict[str, Any]],
    *,
    source: str | None = None,
    limit_heldout: int | None = None,
) -> set[tuple[str, str]]:
    targets = set()
    for pack in packs:
        if source and pack.get("source") != source:
            continue
        if not user_examples_from_pack(pack):
            continue
        heldout_tasks = pack.get("heldout_tasks") or []
        if limit_heldout is not None:
            heldout_tasks = heldout_tasks[:limit_heldout]
        for task in heldout_tasks:
            task_id = str(task.get("task_id") or "")
            if task_id:
                targets.add((str(pack["pack_id"]), task_id))
    return targets


def readiness_report(
    *,
    packs: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    writing_eval_rows: list[dict[str, Any]],
    present_surrogate_rows: list[dict[str, Any]],
    present_official_rows: list[dict[str, Any]],
    limit_heldout: int | None = None,
    require_presentbench_official: bool = True,
    artifact_blockers: list[str] | None = None,
    artifact_warnings: list[str] | None = None,
) -> dict[str, Any]:
    pack_by_id = pack_index(packs)
    generated_counts = generated_example_counts(packs)
    skill_cov = skill_coverage(skill_rows)
    writing_cov = eval_coverage(writing_eval_rows)
    present_surrogate_cov = eval_coverage(present_surrogate_rows)
    present_official_cov = eval_coverage(
        present_official_rows,
        required_modes=("prompt_only", "auto_skill"),
    )

    expected_writing = expected_eval_targets(
        packs,
        source="WritingBench",
        limit_heldout=limit_heldout,
    )
    expected_present = expected_eval_targets(
        packs,
        source="PresentBench",
        limit_heldout=limit_heldout,
    )

    blockers: list[str] = []
    warnings: list[str] = []
    blockers.extend(artifact_blockers or [])
    warnings.extend(artifact_warnings or [])

    inventory = model_inventory(
        skill_rows=skill_rows,
        eval_rows=writing_eval_rows + present_surrogate_rows + present_official_rows,
    )
    if inventory["monoculture"] and inventory["monoculture_model"]:
        warnings.append(
            "model_monoculture: all skill induction, generation, and judge rows "
            f"share model {inventory['monoculture_model']}; results are smoke-only "
            "(see README cross-model recipe)."
        )

    if not packs:
        blockers.append("no example packs loaded")

    schema_errors = {
        "skills": schema_validation_errors(skill_rows, kind="skill", label="skills"),
        "writingbench_eval": schema_validation_errors(
            writing_eval_rows,
            kind="eval",
            label="writingbench_eval",
        ),
        "presentbench_surrogate_eval": schema_validation_errors(
            present_surrogate_rows,
            kind="eval",
            label="presentbench_surrogate_eval",
        ),
        "presentbench_official": schema_validation_errors(
            present_official_rows,
            kind="eval",
            label="presentbench_official",
        ),
    }
    for artifact, errors in schema_errors.items():
        for error in errors:
            blockers.append(f"{artifact}: invalid artifact row: {error}")

    for pack_id, counts in generated_counts.items():
        if counts["train_examples"] == 0:
            blockers.append(f"{pack_id}: no train examples")
        if counts["generated_desired_outputs"] < 2:
            blockers.append(f"{pack_id}: fewer than 2 generated desired outputs")
        if counts["needs_generation"] or counts["invalid_desired_outputs"]:
            blockers.append(f"{pack_id}: desired outputs are not fully frozen")

    for pack_id in pack_by_id:
        coverage = skill_cov.get(pack_id)
        if coverage is None:
            blockers.append(f"{pack_id}: no skill rows")
            continue
        if not coverage["ready"]:
            blockers.append(f"{pack_id}: missing skill modes {coverage['missing_modes']}")

    writing_keys = {(value["pack_id"], value["task_id"]) for value in writing_cov.values()}
    for target in sorted(expected_writing - writing_keys):
        blockers.append(f"{target[0]}::{target[1]}: missing WritingBench eval rows")
    for key, coverage in sorted(writing_cov.items()):
        if not coverage["ready"]:
            blockers.append(f"{key}: incomplete WritingBench eval coverage")

    present_surrogate_keys = {
        (value["pack_id"], value["task_id"]) for value in present_surrogate_cov.values()
    }
    for target in sorted(expected_present - present_surrogate_keys):
        warnings.append(f"{target[0]}::{target[1]}: missing PresentBench surrogate eval rows")
    for key, coverage in sorted(present_surrogate_cov.items()):
        if not coverage["ready"]:
            warnings.append(f"{key}: incomplete PresentBench surrogate eval coverage")

    present_official_issues = blockers if require_presentbench_official else warnings
    present_official_keys = {
        (value["pack_id"], value["task_id"]) for value in present_official_cov.values()
    }
    if expected_present and not present_official_rows:
        present_official_issues.append("PresentBench official score rows are absent")
    for target in sorted(expected_present - present_official_keys):
        present_official_issues.append(
            f"{target[0]}::{target[1]}: missing PresentBench official score rows"
        )
    for key, coverage in sorted(present_official_cov.items()):
        if not coverage["ready"]:
            present_official_issues.append(
                f"{key}: incomplete PresentBench official score coverage"
            )

    return {
        "packs": {
            "total": len(packs),
            "sources": dict(Counter(str(pack.get("source")) for pack in packs)),
            "generated_desired_outputs": generated_counts,
        },
        "skills": {
            "rows": len(skill_rows),
            "coverage": skill_cov,
        },
        "evaluations": {
            "writingbench": {
                "rows": len(writing_eval_rows),
                "expected_targets": len(expected_writing),
                "coverage": writing_cov,
            },
            "presentbench_surrogate": {
                "rows": len(present_surrogate_rows),
                "expected_targets": len(expected_present),
                "coverage": present_surrogate_cov,
            },
            "presentbench_official": {
                "rows": len(present_official_rows),
                "expected_targets": len(expected_present),
                "required": require_presentbench_official,
                "coverage": present_official_cov,
            },
        },
        "status": "ready" if not blockers else "not_ready",
        "blockers": blockers,
        "warnings": warnings,
        "schema_errors": schema_errors,
        "model_inventory": inventory,
    }
