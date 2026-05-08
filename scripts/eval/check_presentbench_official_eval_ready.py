#!/usr/bin/env python3
"""Check whether PresentBench heldout tasks have slide artifacts for official eval."""

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

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.mvp import user_examples_from_pack  # noqa: E402
from auto_skill.presentbench_eval import check_presentbench_official_eval_readiness  # noqa: E402
from scripts.eval.run_heldout_eval import select_packs  # noqa: E402


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
        help=(
            "Local checkout of https://github.com/PresentBench/PresentBench "
            "containing judge_all.py."
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("../PresentBench/results/auto_skill"),
        help=(
            "Result root expected by PresentBench. Each task needs "
            "<result-root>/<source_task_id>/generation_task/results/slides.pdf or slides.pptx."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/presentbench_official_ready.jsonl"))
    parser.add_argument(
        "--judge-model",
        help="Optional upstream judge model prefix used to select *_score.yaml files.",
    )
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 even when slide artifacts are missing; useful for local smoke checks.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 when filters select no PresentBench heldout tasks. Default is fail-closed.",
    )
    args = parser.parse_args()

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
            readiness = check_presentbench_official_eval_readiness(
                task=task,
                data_root=args.data_root,
                result_root=args.result_root,
                code_root=args.code_root,
                judge_model=args.judge_model,
            ).to_json()
            readiness["pack_id"] = pack["pack_id"]
            readiness["source"] = "PresentBench"
            rows.append(readiness)

    write_jsonl(args.out, rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote {len(rows)} readiness rows to {args.out}")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    if not rows and not args.allow_empty:
        return 3
    if args.allow_missing:
        return 0
    ready_statuses = {"ready_for_official_judge", "ready_for_zero_score", "scored"}
    return 0 if all(row["status"] in ready_statuses for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
