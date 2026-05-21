"""Model identity inventory helpers for readiness reports."""

from __future__ import annotations

from typing import Any


def _string_values_from_path(value: Any, path: tuple[str, ...]) -> list[str]:
    if not path:
        if isinstance(value, str) and value.strip():
            return [value]
        return []
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_string_values_from_path(item, path))
        return values
    if not isinstance(value, dict):
        return []
    return _string_values_from_path(value.get(path[0]), path[1:])


def row_model_values(row: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> set[str]:
    """Collect model identifiers from a row using top-level and legacy nested paths."""

    values: set[str] = set()
    for path in paths:
        values.update(_string_values_from_path(row, path))
    return values


def collect_models(rows: list[dict[str, Any]], *, paths: tuple[tuple[str, ...], ...]) -> set[str]:
    """Collect non-empty model identifiers from row fields."""

    models: set[str] = set()
    for row in rows:
        models.update(row_model_values(row, paths))
    return models


def missing_model_count(rows: list[dict[str, Any]], *, paths: tuple[tuple[str, ...], ...]) -> int:
    """Count rows that do not expose a non-empty model identifier."""

    missing = 0
    for row in rows:
        if not row_model_values(row, paths):
            missing += 1
    return missing


def model_inventory(
    *,
    skill_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Inventory of solver/judge models seen across skill and eval artifacts."""

    skill_solver_paths = (("solver_model",), ("model_calls", "model"))
    eval_solver_paths = (
        ("solver_model",),
        ("generation", "model"),
        ("run", "model"),
        ("signature_generation", "model"),
    )
    eval_judge_paths = (("judge_model",), ("judge", "model"), ("judge_calls", "model"))
    skill_solvers = collect_models(skill_rows, paths=skill_solver_paths)
    eval_solvers = collect_models(eval_rows, paths=eval_solver_paths)
    eval_judges = collect_models(eval_rows, paths=eval_judge_paths)
    successful_eval_rows = [row for row in eval_rows if row.get("status") == "success"]
    scored_eval_solvers = collect_models(successful_eval_rows, paths=eval_solver_paths)
    scored_eval_judges = collect_models(successful_eval_rows, paths=eval_judge_paths)
    union = skill_solvers | eval_solvers | eval_judges
    scored_union = skill_solvers | scored_eval_solvers | scored_eval_judges
    missing = {
        "skill_solver_model": missing_model_count(skill_rows, paths=skill_solver_paths),
        "eval_solver_model": missing_model_count(eval_rows, paths=eval_solver_paths),
        "eval_judge_model": missing_model_count(eval_rows, paths=eval_judge_paths),
    }
    scored_missing = {
        "skill_solver_model": missing_model_count(skill_rows, paths=skill_solver_paths),
        "eval_solver_model": missing_model_count(
            successful_eval_rows,
            paths=eval_solver_paths,
        ),
        "eval_judge_model": missing_model_count(
            successful_eval_rows,
            paths=eval_judge_paths,
        ),
    }
    return {
        "skill_solver_models": sorted(skill_solvers),
        "eval_solver_models": sorted(eval_solvers),
        "eval_judge_models": sorted(eval_judges),
        "all_models": sorted(union),
        "missing_model_identity": missing,
        "model_identity_complete": not any(missing.values()),
        "monoculture": len(union) == 1 if union else False,
        "monoculture_model": next(iter(union)) if len(union) == 1 else None,
        "scored": {
            "skill_solver_models": sorted(skill_solvers),
            "eval_solver_models": sorted(scored_eval_solvers),
            "eval_judge_models": sorted(scored_eval_judges),
            "all_models": sorted(scored_union),
            "successful_eval_rows": len(successful_eval_rows),
            "missing_model_identity": scored_missing,
            "model_identity_complete": not any(scored_missing.values()),
            "monoculture": len(scored_union) == 1 if scored_union else False,
            "monoculture_model": (
                next(iter(scored_union)) if len(scored_union) == 1 else None
            ),
        },
    }
