#!/usr/bin/env python3
"""Summarize upstream PresentBench score YAMLs for auto-skill modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.eval_summary import expected_score_cells, summarize_score_rows  # noqa: E402
from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.mvp import user_examples_from_pack  # noqa: E402
from auto_skill.presentbench_eval import (  # noqa: E402
    check_presentbench_official_eval_readiness,
    load_presentbench_score,
)
from scripts.eval.run_heldout_eval import select_packs  # noqa: E402

EVALUATOR_KIND = "presentbench_official_score_yaml"


def parse_score_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--score-root must be MODE=PATH, got: {value}")
        mode, raw_path = value.split("=", 1)
        mode = mode.strip()
        if not mode:
            raise ValueError(f"--score-root mode is empty: {value}")
        roots[mode] = Path(raw_path).expanduser()
    return roots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/PresentBench_repo"))
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path("data/PresentBench_code"),
        help="Local checkout of the official PresentBench evaluation code.",
    )
    parser.add_argument(
        "--score-root",
        action="append",
        required=True,
        help=(
            "Mode-to-result-root mapping, e.g. "
            "prompt_only=../PresentBench/results/prompt_only. Repeat for each mode."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/presentbench_official_scores.jsonl"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("runs/presentbench_official_scores.summary.json"),
    )
    parser.add_argument(
        "--judge-model",
        help="Optional upstream judge model prefix used to select *_score.yaml files.",
    )
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even when some score rows are missing. Default is fail-closed.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 when filters select no PresentBench tasks. Default is fail-closed.",
    )
    args = parser.parse_args()

    try:
        score_roots = parse_score_roots(args.score_root)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    packs = load_jsonl(args.packs)
    selected = select_packs(
        [pack for pack in packs if pack.get("source") == "PresentBench"],
        pack_ids=set(args.pack_id) if args.pack_id else None,
        source=None,
        limit=args.limit_packs,
    )

    rows = []
    for pack in selected:
        if not user_examples_from_pack(pack):
            continue
        heldout_tasks = pack.get("heldout_tasks", [])
        if args.limit_heldout is not None:
            heldout_tasks = heldout_tasks[: args.limit_heldout]
        for task in heldout_tasks:
            task_id = str(task["task_id"])
            for mode, result_root in score_roots.items():
                readiness = check_presentbench_official_eval_readiness(
                    task=task,
                    data_root=args.data_root,
                    result_root=result_root,
                    code_root=args.code_root,
                    judge_model=args.judge_model,
                ).to_json()
                row = {
                    "schema_version": "presentbench-official-score/v1",
                    "pack_id": pack["pack_id"],
                    "task_id": task_id,
                    "source_task_id": task.get("source_task_id"),
                    "mode": mode,
                    "evaluator_kind": EVALUATOR_KIND,
                    "readiness": readiness,
                    "status": "missing_score_artifact",
                    "overall_score": None,
                    "score": None,
                }
                if readiness["status"] == "missing_official_eval_artifacts":
                    row["status"] = "missing_official_eval_artifacts"
                elif readiness["status"] == "ambiguous_score_artifacts":
                    row["status"] = "ambiguous_score_artifacts"
                    row["error"] = readiness.get("score_selection_error")
                elif readiness.get("score_artifact"):
                    try:
                        score = load_presentbench_score(Path(readiness["score_artifact"]))
                    except (OSError, ValueError) as exc:
                        row["status"] = "score_parse_error"
                        row["error"] = str(exc)
                    else:
                        row["status"] = "success"
                        row["overall_score"] = score["score_percent"]
                        row["score"] = score
                rows.append(row)

    if not rows and not args.allow_empty:
        print("error: no PresentBench score rows selected", file=sys.stderr)
        return 3

    write_jsonl(args.out, rows)
    summary = summarize_score_rows(
        rows,
        evaluator_kind=EVALUATOR_KIND,
        score_summary_key="mean_presentbench_percent",
        expected_cells=expected_score_cells(
            selected,
            score_roots,
            limit_heldout=args.limit_heldout,
        ),
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} PresentBench official score rows to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(row.get("status") != "success" for row in rows) and not args.allow_partial:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
