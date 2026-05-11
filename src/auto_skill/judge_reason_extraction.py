"""Join WritingBench judge reasons for losing candidate tasks.

This module is pure post-processing over already-evaluated WritingBench rows.
It pairs two modes on ``(pack_id, task_id)``, keeps only successful cells with
matching criterion sets, and emits the per-criterion score/reason payload for
both sides. It does not call any model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Any

TaskKey = tuple[str, str]


@dataclass(frozen=True)
class MissingReasonCell:
    pack_id: str
    task_id: str
    missing_modes: tuple[str, ...]


@dataclass(frozen=True)
class MismatchedCriteriaCell:
    pack_id: str
    task_id: str
    baseline_only: tuple[str, ...]
    mode_only: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateReasonCell:
    mode: str
    pack_id: str
    task_id: str
    count: int


@dataclass(frozen=True)
class JudgeReasonExtraction:
    rows: tuple[dict[str, Any], ...]
    missing_cells: tuple[MissingReasonCell, ...]
    mismatched_criteria: tuple[MismatchedCriteriaCell, ...]
    duplicate_cells: tuple[DuplicateReasonCell, ...]
    paired_success_cells: int
    loss_count: int
    win_count: int
    tie_count: int


def _task_key(row: dict[str, Any]) -> TaskKey:
    return (str(row.get("pack_id") or ""), str(row.get("task_id") or ""))


def _first_score_reason_by_criterion(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if str(row.get("status") or "") != "success":
        return {}
    raw_scores = row.get("scores")
    if not isinstance(raw_scores, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for criterion_name, entries in raw_scores.items():
        if not isinstance(entries, list) or not entries:
            continue
        first = entries[0]
        if not isinstance(first, dict):
            continue
        score = first.get("score")
        if not isinstance(score, (int, float)):
            continue
        reason = first.get("reason")
        out[str(criterion_name)] = {
            "score": float(score),
            "reason": reason if isinstance(reason, str) else "",
        }
    return out


def _index_success_rows(
    rows: Iterable[dict[str, Any]], modes: set[str]
) -> tuple[dict[tuple[str, TaskKey], dict[str, Any]], tuple[DuplicateReasonCell, ...]]:
    cells: dict[tuple[str, TaskKey], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("mode") or "")
        if mode not in modes or str(row.get("status") or "") != "success":
            continue
        cells[(mode, _task_key(row))].append(row)

    index: dict[tuple[str, TaskKey], dict[str, Any]] = {}
    duplicates: list[DuplicateReasonCell] = []
    for (mode, task_key), cell_rows in cells.items():
        if len(cell_rows) == 1:
            index[(mode, task_key)] = cell_rows[0]
            continue
        duplicates.append(
            DuplicateReasonCell(
                mode=mode,
                pack_id=task_key[0],
                task_id=task_key[1],
                count=len(cell_rows),
            )
        )
    duplicates.sort(key=lambda item: (item.mode, item.pack_id, item.task_id))
    return index, tuple(duplicates)


def extract_losing_task_judge_reasons(
    rows: Iterable[dict[str, Any]],
    *,
    mode: str = "ours_no_validation",
    baseline_mode: str = "few_shot_examples_only",
    expected_task_keys: Iterable[TaskKey] | None = None,
    losing_only: bool = True,
) -> JudgeReasonExtraction:
    """Build paired score/reason rows for candidate-vs-baseline tasks.

    ``expected_task_keys`` lets tests or callers verify missing-row handling.
    When omitted, the function pairs all successful common cells for the two
    modes. Rows with duplicate successful cells are reported and excluded.
    """

    row_list = list(rows)
    index, duplicates = _index_success_rows(row_list, {mode, baseline_mode})
    if expected_task_keys is None:
        task_keys = sorted(
            {task_key for row_mode, task_key in index if row_mode == mode}
            | {task_key for row_mode, task_key in index if row_mode == baseline_mode}
        )
    else:
        task_keys = sorted(set(expected_task_keys))

    joined: list[dict[str, Any]] = []
    missing: list[MissingReasonCell] = []
    mismatched: list[MismatchedCriteriaCell] = []
    paired_success_cells = 0
    loss_count = 0
    win_count = 0
    tie_count = 0

    for task_key in task_keys:
        baseline_row = index.get((baseline_mode, task_key))
        mode_row = index.get((mode, task_key))
        missing_modes = [
            row_mode
            for row_mode, row in ((baseline_mode, baseline_row), (mode, mode_row))
            if row is None
        ]
        if missing_modes:
            missing.append(
                MissingReasonCell(
                    pack_id=task_key[0],
                    task_id=task_key[1],
                    missing_modes=tuple(missing_modes),
                )
            )
            continue
        assert baseline_row is not None
        assert mode_row is not None
        baseline_scores = _first_score_reason_by_criterion(baseline_row)
        mode_scores = _first_score_reason_by_criterion(mode_row)
        baseline_set = set(baseline_scores)
        mode_set = set(mode_scores)
        if not baseline_set or not mode_set:
            missing.append(
                MissingReasonCell(
                    pack_id=task_key[0],
                    task_id=task_key[1],
                    missing_modes=tuple(
                        row_mode
                        for row_mode, scores in (
                            (baseline_mode, baseline_scores),
                            (mode, mode_scores),
                        )
                        if not scores
                    ),
                )
            )
            continue
        if baseline_set != mode_set:
            mismatched.append(
                MismatchedCriteriaCell(
                    pack_id=task_key[0],
                    task_id=task_key[1],
                    baseline_only=tuple(sorted(baseline_set - mode_set)),
                    mode_only=tuple(sorted(mode_set - baseline_set)),
                )
            )
            continue

        criteria = []
        for criterion_name in sorted(baseline_set):
            baseline = baseline_scores[criterion_name]
            candidate = mode_scores[criterion_name]
            delta = candidate["score"] - baseline["score"]
            criteria.append(
                {
                    "criterion_name": criterion_name,
                    "baseline": {
                        "mode": baseline_mode,
                        "score": baseline["score"],
                        "reason": baseline["reason"],
                    },
                    "candidate": {
                        "mode": mode,
                        "score": candidate["score"],
                        "reason": candidate["reason"],
                    },
                    "delta": delta,
                }
            )
        baseline_overall = mean(item["baseline"]["score"] for item in criteria)
        mode_overall = mean(item["candidate"]["score"] for item in criteria)
        overall_delta = mode_overall - baseline_overall
        paired_success_cells += 1
        if overall_delta < 0:
            loss_count += 1
        elif overall_delta > 0:
            win_count += 1
        else:
            tie_count += 1
        if losing_only and overall_delta >= 0:
            continue
        worst = min(criteria, key=lambda item: item["delta"])
        joined.append(
            {
                "schema_version": "judge-reason-pairs/v1",
                "pack_id": task_key[0],
                "task_id": task_key[1],
                "source_task_id": str(mode_row.get("source_task_id") or ""),
                "mode": mode,
                "baseline_mode": baseline_mode,
                "judge_model": str(mode_row.get("judge_model") or ""),
                "solver_model": str(mode_row.get("solver_model") or ""),
                "baseline_overall": baseline_overall,
                "mode_overall": mode_overall,
                "overall_delta": overall_delta,
                "worst_criterion": {
                    "criterion_name": worst["criterion_name"],
                    "delta": worst["delta"],
                },
                "criteria": criteria,
            }
        )

    joined.sort(key=lambda item: (item["overall_delta"], item["pack_id"], item["task_id"]))
    return JudgeReasonExtraction(
        rows=tuple(joined),
        missing_cells=tuple(missing),
        mismatched_criteria=tuple(mismatched),
        duplicate_cells=duplicates,
        paired_success_cells=paired_success_cells,
        loss_count=loss_count,
        win_count=win_count,
        tie_count=tie_count,
    )
