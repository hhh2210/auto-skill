"""Shared helpers for benchmark score summaries and paired deltas."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

ScoreCell = tuple[str, str, str]


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def expected_score_cells(
    packs: list[dict[str, Any]],
    modes: Iterable[str],
    *,
    limit_heldout: int | None = None,
) -> list[ScoreCell]:
    """Return the expected pack/task/mode matrix for selected heldout packs."""

    cells: list[ScoreCell] = []
    clean_modes = [str(mode) for mode in modes if str(mode)]
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        if not pack_id:
            continue
        heldout_tasks = pack.get("heldout_tasks") or []
        if limit_heldout is not None:
            heldout_tasks = heldout_tasks[:limit_heldout]
        for task in heldout_tasks:
            task_id = str(task.get("task_id") or "")
            if not task_id:
                continue
            for mode in clean_modes:
                cells.append((pack_id, task_id, mode))
    return cells


def _cell_statuses(statuses: dict[ScoreCell, list[str]], cell: ScoreCell) -> list[str]:
    return statuses.get(cell) or ["missing"]


def _coverage_report(
    *,
    expected_cells: set[ScoreCell],
    observed_cells: set[ScoreCell],
    successful_score_cells: set[ScoreCell],
    cell_statuses: dict[ScoreCell, list[str]],
) -> dict[str, Any]:
    missing_cells = sorted(expected_cells - observed_cells)
    non_success_cells = []
    for cell in sorted(expected_cells & observed_cells):
        if cell in successful_score_cells:
            continue
        pack_id, task_id, mode = cell
        non_success_cells.append(
            {
                "pack_id": pack_id,
                "task_id": task_id,
                "mode": mode,
                "statuses": _cell_statuses(cell_statuses, cell),
            }
        )
    return {
        "expected_cells": len(expected_cells),
        "observed_cells": len(observed_cells & expected_cells),
        "successful_score_cells": len(successful_score_cells & expected_cells),
        "missing_cells": [
            {"pack_id": pack_id, "task_id": task_id, "mode": mode}
            for pack_id, task_id, mode in missing_cells
        ],
        "non_success_cells": non_success_cells,
        "unexpected_cells": [
            {"pack_id": pack_id, "task_id": task_id, "mode": mode}
            for pack_id, task_id, mode in sorted(observed_cells - expected_cells)
        ],
        "complete": not missing_cells and not non_success_cells,
    }


def summarize_score_rows(
    rows: list[dict[str, Any]],
    *,
    evaluator_kind: str,
    score_summary_key: str,
    baseline_mode: str = "prompt_only",
    expected_cells: Iterable[ScoreCell] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    status_counts: dict[str, int] = {}
    status_counts_by_mode: dict[str, dict[str, int]] = {}
    success_by_task: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    observed_cells: set[ScoreCell] = set()
    successful_score_cells: set[ScoreCell] = set()
    cell_statuses: dict[ScoreCell, list[str]] = defaultdict(list)

    for row in rows:
        mode = str(row.get("mode") or "unknown")
        status = str(row.get("status") or "unknown")
        pack_id = str(row.get("pack_id") or "")
        task_id = str(row.get("task_id") or "")
        cell = (pack_id, task_id, mode)
        observed_cells.add(cell)
        cell_statuses[cell].append(status)
        status_counts[status] = status_counts.get(status, 0) + 1
        by_mode = status_counts_by_mode.setdefault(mode, {})
        by_mode[status] = by_mode.get(status, 0) + 1
        if status != "success":
            continue
        score = finite_number(row.get("overall_score"))
        if score is None:
            continue
        grouped[mode].append(score)
        successful_score_cells.add(cell)
        task_key = (pack_id, task_id)
        success_by_task[task_key][mode] = score

    expected_cell_set = set(expected_cells or [])
    expected_modes_by_task: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pack_id, task_id, mode in expected_cell_set:
        expected_modes_by_task[(pack_id, task_id)].add(mode)

    paired_deltas: dict[str, Any] = {}
    modes = sorted(set(grouped) | {mode for _, _, mode in expected_cell_set})
    for mode in modes:
        if mode == baseline_mode:
            continue
        deltas = []
        examples = []
        if expected_cell_set:
            pair_tasks = sorted(
                task_key
                for task_key, task_modes in expected_modes_by_task.items()
                if baseline_mode in task_modes and mode in task_modes
            )
        else:
            pair_tasks = sorted(success_by_task)
        missing_pairs = []
        for pack_id, task_id in pair_tasks:
            scores = success_by_task.get((pack_id, task_id), {})
            if baseline_mode not in scores or mode not in scores:
                missing_pairs.append(
                    {
                        "pack_id": pack_id,
                        "task_id": task_id,
                        "baseline_statuses": _cell_statuses(
                            cell_statuses, (pack_id, task_id, baseline_mode)
                        ),
                        "mode_statuses": _cell_statuses(cell_statuses, (pack_id, task_id, mode)),
                    }
                )
                continue
            delta = scores[mode] - scores[baseline_mode]
            deltas.append(delta)
            examples.append(
                {
                    "pack_id": pack_id,
                    "task_id": task_id,
                    "baseline_score": scores[baseline_mode],
                    "mode_score": scores[mode],
                    "delta": delta,
                }
            )
        if not deltas:
            paired_deltas[mode] = {
                "baseline_mode": baseline_mode,
                "count": 0,
                "expected_pairs": len(pair_tasks),
                "missing_pairs": missing_pairs,
                "mean_delta": None,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "negative_transfer_examples": [],
            }
            continue
        paired_deltas[mode] = {
            "baseline_mode": baseline_mode,
            "count": len(deltas),
            "expected_pairs": len(pair_tasks),
            "missing_pairs": missing_pairs,
            "mean_delta": sum(deltas) / len(deltas),
            "wins": sum(1 for delta in deltas if delta > 0),
            "losses": sum(1 for delta in deltas if delta < 0),
            "ties": sum(1 for delta in deltas if delta == 0),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "negative_transfer_examples": [
                item for item in examples if item["delta"] < 0
            ],
        }

    summary = {
        "evaluator_kind": evaluator_kind,
        "baseline_mode": baseline_mode,
        "status_counts": dict(sorted(status_counts.items())),
        "status_counts_by_mode": {
            mode: dict(sorted(counts.items()))
            for mode, counts in sorted(status_counts_by_mode.items())
        },
        "modes": {
            mode: {
                "count": len(scores),
                score_summary_key: sum(scores) / len(scores) if scores else None,
            }
            for mode, scores in sorted(grouped.items())
        },
        "paired_deltas": paired_deltas,
    }
    if expected_cell_set:
        summary["coverage"] = _coverage_report(
            expected_cells=expected_cell_set,
            observed_cells=observed_cells,
            successful_score_cells=successful_score_cells,
            cell_statuses=cell_statuses,
        )
    return summary
