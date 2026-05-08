#!/usr/bin/env python3
"""Summarize MVP benchmark, artifact, token, and self-consistency metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.eval_summary import ScoreCell, expected_score_cells  # noqa: E402
from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.metrics import (  # noqa: E402
    score_and_negative_transfer_summary,
    self_consistency_summary,
    skill_artifact_summary,
    skill_induction_token_usage_summary,
    token_usage_summary,
)
from auto_skill.readiness import model_inventory  # noqa: E402


def load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    return load_jsonl(path)


def summarize_eval_file(
    path: Path,
    *,
    baseline_mode: str,
    expected_cells: list[ScoreCell] | None = None,
) -> dict[str, Any]:
    rows = load_existing_jsonl(path)
    status_counts: dict[str, int] = {}
    status_counts_by_mode: dict[str, dict[str, int]] = defaultdict(dict)
    observed_cells = set()
    success_cells = set()
    cell_statuses: dict[ScoreCell, list[str]] = defaultdict(list)
    for row in rows:
        status = str(row.get("status") or "unknown")
        mode = str(row.get("mode") or "unknown")
        cell = (
            str(row.get("pack_id") or ""),
            str(row.get("task_id") or ""),
            mode,
        )
        observed_cells.add(cell)
        cell_statuses[cell].append(status)
        if status == "success":
            success_cells.add(cell)
        status_counts[status] = status_counts.get(status, 0) + 1
        mode_counts = status_counts_by_mode.setdefault(mode, {})
        mode_counts[status] = mode_counts.get(status, 0) + 1
    by_evaluator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_evaluator[str(row.get("evaluator_kind") or "unknown")].append(row)
    coverage = coverage_report(
        expected_cells=expected_cells,
        observed_cells=observed_cells,
        success_cells=success_cells,
        cell_statuses=cell_statuses,
        rows=rows,
    )
    bucket = classify_score_bucket(path=path, evaluator_kinds=by_evaluator)
    return {
        "path": str(path),
        "score_bucket": bucket,
        "row_count": len(rows),
        "coverage": coverage
        | {
            "status_counts": dict(sorted(status_counts.items())),
            "status_counts_by_mode": {
                mode: dict(sorted(counts.items()))
                for mode, counts in sorted(status_counts_by_mode.items())
            },
        },
        "evaluators": {
            evaluator: score_and_negative_transfer_summary(
                evaluator_rows,
                evaluator_kind=evaluator,
                baseline_mode=baseline_mode,
                expected_cells=expected_cells,
            )
            for evaluator, evaluator_rows in sorted(by_evaluator.items())
        },
        "token_cost_at_heldout_inference": token_usage_summary(rows),
    }


def cell_to_json(cell: ScoreCell) -> dict[str, str]:
    pack_id, task_id, mode = cell
    return {"pack_id": pack_id, "task_id": task_id, "mode": mode}


def coverage_report(
    *,
    expected_cells: list[ScoreCell] | None,
    observed_cells: set[ScoreCell],
    success_cells: set[ScoreCell],
    cell_statuses: dict[ScoreCell, list[str]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not expected_cells:
        return {
            "expected_cells": None,
            "observed_cells": len(observed_cells),
            "successful_cells": len(success_cells),
            "non_success_rows": sum(1 for row in rows if row.get("status") != "success"),
            "missing_cells": [],
            "non_success_cells": [],
            "unexpected_cells": [],
            "complete": None,
            "coverage_basis": "observed_rows_only",
        }
    expected = set(expected_cells)
    missing = sorted(expected - observed_cells)
    non_success = []
    for cell in sorted(expected & observed_cells):
        if cell in success_cells:
            continue
        item = cell_to_json(cell)
        item["statuses"] = cell_statuses.get(cell) or ["missing"]
        non_success.append(item)
    return {
        "expected_cells": len(expected),
        "observed_cells": len(observed_cells & expected),
        "successful_cells": len(success_cells & expected),
        "non_success_rows": sum(1 for row in rows if row.get("status") != "success"),
        "missing_cells": [cell_to_json(cell) for cell in missing],
        "non_success_cells": non_success,
        "unexpected_cells": [cell_to_json(cell) for cell in sorted(observed_cells - expected)],
        "complete": not missing and not non_success,
        "coverage_basis": "expected_matrix",
    }


def classify_score_bucket(
    *,
    path: Path,
    evaluator_kinds: dict[str, list[dict[str, Any]]],
) -> str:
    text = " ".join([str(path), *evaluator_kinds]).lower()
    if "pattern_similarity" in text or "example_pattern_similarity" in text:
        return "debug_scores"
    if "surrogate" in text:
        return "surrogate_debug_scores"
    if "official" in text or "writingbench" in text:
        return "official_benchmark_scores"
    return "debug_scores"


def infer_eval_source(path: Path, rows: list[dict[str, Any]]) -> str | None:
    sources = {str(row.get("source")) for row in rows if row.get("source")}
    if len(sources) == 1:
        return next(iter(sources))
    text = " ".join(
        [
            str(path),
            *[str(row.get("evaluator_kind") or "") for row in rows[:20]],
            *[str(row.get("pack_id") or "") for row in rows[:20]],
        ]
    ).lower()
    if "writingbench" in text:
        return "WritingBench"
    if "presentbench" in text:
        return "PresentBench"
    return None


def expected_cells_for_eval(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    packs: list[dict[str, Any]],
    modes: list[str],
    limit_heldout: int | None,
) -> list[ScoreCell] | None:
    if not packs or not modes:
        return None
    source = infer_eval_source(path, rows)
    selected = [
        pack for pack in packs if source is None or str(pack.get("source")) == source
    ]
    return expected_score_cells(selected, modes, limit_heldout=limit_heldout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", type=Path, default=Path("runs/skill_mvp.qwen.mvp.jsonl"))
    parser.add_argument(
        "--eval",
        type=Path,
        action="append",
        default=[],
        help="Eval JSONL file. Repeat for official and surrogate eval artifacts.",
    )
    parser.add_argument(
        "--self-consistency",
        type=Path,
        action="append",
        default=[],
        help="Self-consistency JSONL output from run_self_consistency_metric.py.",
    )
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
        help="Example packs used to construct expected eval cells.",
    )
    parser.add_argument(
        "--modes",
        default=(
            "prompt_only,few_shot_examples_only,one_shot_skill_from_examples,"
            "ours_no_validation"
        ),
        help="Comma-separated heldout eval modes expected in each eval file.",
    )
    parser.add_argument(
        "--limit-heldout",
        type=int,
        help="Heldout-task limit used when the eval artifacts were generated.",
    )
    parser.add_argument("--baseline-mode", default="prompt_only")
    parser.add_argument(
        "--artifact-compare-modes",
        default="one_shot_skill_from_examples,auto_skill_feature_driven_no_validation",
        help="Comma-separated pair for artifact metric comparison.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/mvp_metrics.summary.json"))
    args = parser.parse_args()

    compare_modes = tuple(
        mode.strip() for mode in args.artifact_compare_modes.split(",") if mode.strip()
    )
    if len(compare_modes) != 2:
        print("error: --artifact-compare-modes must contain exactly two modes", file=sys.stderr)
        return 2

    skill_rows = load_existing_jsonl(args.skills)
    packs = load_existing_jsonl(args.packs) if args.packs and args.packs.exists() else []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    eval_summaries = []
    for path in args.eval:
        rows = load_existing_jsonl(path)
        expected_cells = expected_cells_for_eval(
            path=path,
            rows=rows,
            packs=packs,
            modes=modes,
            limit_heldout=args.limit_heldout,
        )
        eval_summaries.append(
            summarize_eval_file(
                path,
                baseline_mode=args.baseline_mode,
                expected_cells=expected_cells,
            )
        )
    self_consistency_rows = []
    for path in args.self_consistency:
        self_consistency_rows.extend(load_existing_jsonl(path))

    aggregated_eval_rows: list[dict[str, Any]] = []
    for path in args.eval:
        aggregated_eval_rows.extend(load_existing_jsonl(path))
    inventory = model_inventory(
        skill_rows=skill_rows,
        eval_rows=aggregated_eval_rows + self_consistency_rows,
    )

    score_buckets = {
        "official_benchmark_scores": [],
        "surrogate_debug_scores": [],
        "debug_scores": [],
    }
    for eval_summary in eval_summaries:
        score_buckets[eval_summary["score_bucket"]].append(eval_summary)

    summary = {
        "schema_version": "mvp-metrics-summary/v1",
        "inputs": {
            "skills": str(args.skills),
            "eval": [str(path) for path in args.eval],
            "self_consistency": [str(path) for path in args.self_consistency],
            "packs": str(args.packs) if args.packs else None,
            "modes": modes,
            "limit_heldout": args.limit_heldout,
            "baseline_mode": args.baseline_mode,
            "artifact_compare_modes": list(compare_modes),
        },
        "official_benchmark_scores": score_buckets["official_benchmark_scores"],
        "surrogate_debug_scores": score_buckets["surrogate_debug_scores"],
        "debug_scores": score_buckets["debug_scores"],
        "benchmark_score_and_paired_delta": score_buckets["official_benchmark_scores"],
        "all_score_summaries": eval_summaries,
        "coverage_and_non_success": [
            {"path": eval_summary["path"], **eval_summary["coverage"]}
            for eval_summary in eval_summaries
        ],
        "negative_transfer_rate": [
            {
                "path": eval_summary["path"],
                "evaluators": {
                    evaluator: summary["negative_transfer"]
                    for evaluator, summary in eval_summary["evaluators"].items()
                },
            }
            for eval_summary in eval_summaries
        ],
        "skill_artifact_metrics": skill_artifact_summary(
            skill_rows,
            compare_modes=(compare_modes[0], compare_modes[1]),
        ),
        "skill_induction_token_cost": skill_induction_token_usage_summary(skill_rows),
        "token_cost_at_heldout_inference": [
            {
                "path": eval_summary["path"],
                **eval_summary["token_cost_at_heldout_inference"],
            }
            for eval_summary in eval_summaries
        ],
        "self_consistency_metric": self_consistency_summary(self_consistency_rows)
        if self_consistency_rows
        else None,
        "model_inventory": inventory,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote MVP metrics summary to {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
