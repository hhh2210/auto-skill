"""Token usage metric helpers."""

from __future__ import annotations

from typing import Any


def _usage_tokens(usage: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def _empty_usage_tokens() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _add_usage(accumulator: dict[str, int], usage: dict[str, Any] | None) -> None:
    tokens = _usage_tokens(usage)
    for key, value in tokens.items():
        accumulator[key] += value


def _sum_usage_tokens(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        key: left[key] + right[key]
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def token_usage_summary(
    rows: list[dict[str, Any]],
    *,
    examples_only_mode: str = "few_shot_examples_only",
) -> dict[str, Any]:
    """Summarize heldout inference token usage from eval rows."""

    by_mode: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        item = by_mode.setdefault(
            mode,
            {
                "attempted_rows": 0,
                "successful_rows": 0,
                "status_counts": {},
                "layout_plan": _empty_usage_tokens(),
                "generation": _empty_usage_tokens(),
                "judge": _empty_usage_tokens(),
            },
        )
        item["attempted_rows"] += 1
        status = str(row.get("status") or "unknown")
        item["status_counts"][status] = item["status_counts"].get(status, 0) + 1
        generation = row.get("generation")
        if isinstance(generation, dict):
            _add_usage(item["generation"], generation.get("usage"))
        layout_plan = row.get("layout_plan")
        if isinstance(layout_plan, dict):
            _add_usage(item["layout_plan"], layout_plan.get("usage"))
        judge = row.get("judge")
        if isinstance(judge, dict):
            _add_usage(item["judge"], judge.get("usage"))
        for judge_call in row.get("judge_calls") or []:
            if isinstance(judge_call, dict):
                _add_usage(item["judge"], judge_call.get("usage"))
        if row.get("status") == "success":
            item["successful_rows"] += 1

    for item in by_mode.values():
        count = item["successful_rows"]
        attempted = item["attempted_rows"]
        item["combined_model_calls"] = _sum_usage_tokens(
            _sum_usage_tokens(item["layout_plan"], item["generation"]),
            item["judge"],
        )
        item["status_counts"] = dict(sorted(item["status_counts"].items()))
        for bucket in ("layout_plan", "generation", "judge"):
            totals = item[bucket]
            item[f"{bucket}_avg_per_success"] = {
                key: value / count if count else None for key, value in totals.items()
            }
            item[f"{bucket}_avg_per_attempt"] = {
                key: value / attempted if attempted else None for key, value in totals.items()
            }
        item["combined_model_calls_avg_per_success"] = {
            key: value / count if count else None
            for key, value in item["combined_model_calls"].items()
        }
        item["combined_model_calls_avg_per_attempt"] = {
            key: value / attempted if attempted else None
            for key, value in item["combined_model_calls"].items()
        }

    comparisons: dict[str, Any] = {}
    baseline = by_mode.get(examples_only_mode)
    if baseline:
        baseline_avg = baseline["generation_avg_per_success"]
        for mode, item in sorted(by_mode.items()):
            if mode == examples_only_mode:
                continue
            mode_avg = item["generation_avg_per_success"]
            comparisons[mode] = {
                key: {
                    "examples_only_avg": baseline_avg[key],
                    "mode_avg": mode_avg[key],
                    "delta": mode_avg[key] - baseline_avg[key]
                    if mode_avg[key] is not None and baseline_avg[key] is not None
                    else None,
                    "ratio": mode_avg[key] / baseline_avg[key]
                    if mode_avg[key] is not None and baseline_avg[key]
                    else None,
                }
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
    return {
        "by_mode": dict(sorted(by_mode.items())),
        "comparison_to_examples_only_generation": {
            "examples_only_mode": examples_only_mode,
            "modes": comparisons,
        },
    }


def skill_induction_token_usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize upfront skill-induction token usage from skill rows."""

    by_mode: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        item = by_mode.setdefault(
            mode,
            {
                "attempted_rows": 0,
                "successful_rows": 0,
                "status_counts": {},
                "model_calls": _empty_usage_tokens(),
                "by_stage": {},
            },
        )
        item["attempted_rows"] += 1
        status = str(row.get("status") or "unknown")
        item["status_counts"][status] = item["status_counts"].get(status, 0) + 1
        for call in row.get("model_calls") or []:
            if not isinstance(call, dict):
                continue
            _add_usage(item["model_calls"], call.get("usage"))
            stage = str(call.get("stage") or "unknown")
            stage_usage = item["by_stage"].setdefault(stage, _empty_usage_tokens())
            _add_usage(stage_usage, call.get("usage"))
        if row.get("status") == "success":
            item["successful_rows"] += 1

    for item in by_mode.values():
        count = item["successful_rows"]
        attempted = item["attempted_rows"]
        item["status_counts"] = dict(sorted(item["status_counts"].items()))
        item["model_calls_avg_per_success"] = {
            key: value / count if count else None
            for key, value in item["model_calls"].items()
        }
        item["model_calls_avg_per_attempt"] = {
            key: value / attempted if attempted else None
            for key, value in item["model_calls"].items()
        }
        item["by_stage"] = dict(sorted(item["by_stage"].items()))

    return {"by_mode": dict(sorted(by_mode.items()))}
