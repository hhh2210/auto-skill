"""Cross-file summaries for author-style evaluation rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def latest_eval_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    unkeyed = []
    for row in rows:
        pack_id = row.get("pack_id")
        task_id = row.get("task_id")
        mode = row.get("mode")
        if isinstance(pack_id, str) and isinstance(task_id, str) and isinstance(mode, str):
            latest[(pack_id, task_id, mode)] = row
        else:
            unkeyed.append(row)
    return unkeyed + list(latest.values())


def row_score(row: dict[str, Any]) -> float | None:
    score = numeric(row.get("style_likeness_1_to_10"))
    if score is not None:
        return score
    report = row.get("judge_report")
    if isinstance(report, dict):
        return numeric(report.get("style_likeness_1_to_10"))
    return None


def row_win_rate(row: dict[str, Any]) -> float | None:
    wins = numeric(row.get("candidate_beats_negatives"))
    total = numeric(row.get("hard_negative_count"))
    if wins is None:
        report = row.get("judge_report")
        if isinstance(report, dict):
            wins = numeric(report.get("candidate_beats_negatives"))
    if total is None:
        total = numeric(row.get("hard_negative_count"))
    if wins is None or total is None or total <= 0:
        return None
    return wins / total


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def pack_macro_paired_deltas(
    *,
    success_index: dict[tuple[str, str, str], dict[str, Any]],
    baseline_cells: list[tuple[str, str, dict[str, Any]]],
    compared_modes: list[str],
) -> dict[str, Any]:
    output = {}
    for mode in compared_modes:
        pack_style_deltas: dict[str, list[float]] = defaultdict(list)
        pack_win_deltas: dict[str, list[float]] = defaultdict(list)
        for pack_id, task_id, baseline in baseline_cells:
            candidate = success_index.get((pack_id, task_id, mode))
            if candidate is None:
                continue
            baseline_score = row_score(baseline)
            candidate_score = row_score(candidate)
            baseline_win_rate = row_win_rate(baseline)
            candidate_win_rate = row_win_rate(candidate)
            if baseline_score is not None and candidate_score is not None:
                pack_style_deltas[pack_id].append(candidate_score - baseline_score)
            if baseline_win_rate is not None and candidate_win_rate is not None:
                pack_win_deltas[pack_id].append(candidate_win_rate - baseline_win_rate)
        style_means = {
            pack_id: sum(values) / len(values)
            for pack_id, values in pack_style_deltas.items()
            if values
        }
        win_means = {
            pack_id: sum(values) / len(values)
            for pack_id, values in pack_win_deltas.items()
            if values
        }
        output[mode] = {
            "baseline_mode": "few_shot_examples_only",
            "paired_packs": len(style_means),
            "mean_pack_style_likeness_delta": mean(list(style_means.values())),
            "mean_pack_hard_negative_win_rate_delta": mean(list(win_means.values())),
            "pack_style_delta_sign_counts": dict(
                Counter(sign(value) for value in style_means.values())
            ),
        }
    return output


def summarize_author_style_eval_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_mode: str = "few_shot_examples_only",
) -> dict[str, Any]:
    latest_rows = latest_eval_cells(rows)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts = Counter(str(row.get("status") or "unknown") for row in latest_rows)
    for row in latest_rows:
        mode = row.get("mode")
        if isinstance(mode, str) and mode:
            by_mode[mode].append(row)

    modes = {}
    for mode, mode_rows in sorted(by_mode.items()):
        success_rows = [row for row in mode_rows if row.get("status") == "success"]
        scores = [score for row in success_rows if (score := row_score(row)) is not None]
        win_rates = [rate for row in success_rows if (rate := row_win_rate(row)) is not None]
        modes[mode] = {
            "rows": len(mode_rows),
            "success": len(success_rows),
            "status_counts": dict(
                Counter(str(row.get("status") or "unknown") for row in mode_rows)
            ),
            "mean_style_likeness": mean(scores),
            "hard_negative_win_rate": mean(win_rates),
        }

    success_index = {
        (str(row.get("pack_id")), str(row.get("task_id")), str(row.get("mode"))): row
        for row in latest_rows
        if row.get("status") == "success"
    }
    paired_deltas = {}
    compared_modes = sorted(mode for mode in by_mode if mode != baseline_mode)
    baseline_cells = [
        (pack_id, task_id, row)
        for (pack_id, task_id, mode), row in success_index.items()
        if mode in {baseline_mode}
    ]
    for mode in compared_modes:
        style_deltas = []
        win_rate_deltas = []
        signs = Counter()
        for pack_id, task_id, baseline in baseline_cells:
            candidate = success_index.get((pack_id, task_id, mode))
            if candidate is None:
                continue
            baseline_score = row_score(baseline)
            candidate_score = row_score(candidate)
            baseline_win_rate = row_win_rate(baseline)
            candidate_win_rate = row_win_rate(candidate)
            if baseline_score is not None and candidate_score is not None:
                delta = candidate_score - baseline_score
                style_deltas.append(delta)
                signs[sign(delta)] += 1
            if baseline_win_rate is not None and candidate_win_rate is not None:
                win_rate_deltas.append(candidate_win_rate - baseline_win_rate)
        paired_deltas[mode] = {
            "baseline_mode": baseline_mode,
            "paired_rows": len(style_deltas),
            "mean_style_likeness_delta": mean(style_deltas),
            "mean_hard_negative_win_rate_delta": mean(win_rate_deltas),
            "style_delta_sign_counts": dict(signs),
        }
    pack_macro_deltas = pack_macro_paired_deltas(
        success_index=success_index,
        baseline_cells=baseline_cells,
        compared_modes=compared_modes,
    )
    for mode_summary in pack_macro_deltas.values():
        mode_summary["baseline_mode"] = baseline_mode

    return {
        "schema_version": "author-style-cross-eval-summary/v1",
        "rows": len(latest_rows),
        "status_counts": dict(status_counts),
        "baseline_mode": baseline_mode,
        "modes": modes,
        "paired_deltas": paired_deltas,
        "pack_macro_paired_deltas": pack_macro_deltas,
    }
