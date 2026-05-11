#!/usr/bin/env python3
"""Extract paired judge score/reason text for losing WritingBench tasks.

This is a deterministic post-processor. It does not call any LLM and does not
read candidate generations beyond the already-evaluated score rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.judge_reason_extraction import (  # noqa: E402
    extract_losing_task_judge_reasons,
)


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing eval file: {path}")
        rows.extend(load_jsonl(path))
    return rows


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _diagnosis_loss_count(path: Path) -> int | None:
    if not path.exists():
        raise FileNotFoundError(f"missing diagnosis file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    paired = data.get("paired_overall") if isinstance(data, dict) else None
    if not isinstance(paired, dict):
        return None
    value = paired.get("loss_count")
    return int(value) if isinstance(value, int) else None


def _summary_payload(args: argparse.Namespace, extraction: Any) -> dict[str, Any]:
    return {
        "eval_files": [str(p) for p in args.eval],
        "mode": args.mode,
        "baseline_mode": args.baseline_mode,
        "paired_success_cells": extraction.paired_success_cells,
        "loss_count": extraction.loss_count,
        "win_count": extraction.win_count,
        "tie_count": extraction.tie_count,
        "output_rows": len(extraction.rows),
        "missing_cells": [
            {
                "pack_id": item.pack_id,
                "task_id": item.task_id,
                "missing_modes": list(item.missing_modes),
            }
            for item in extraction.missing_cells
        ],
        "mismatched_criteria": [
            {
                "pack_id": item.pack_id,
                "task_id": item.task_id,
                "baseline_only": list(item.baseline_only),
                "mode_only": list(item.mode_only),
            }
            for item in extraction.mismatched_criteria
        ],
        "duplicate_cells": [
            {
                "mode": item.mode,
                "pack_id": item.pack_id,
                "task_id": item.task_id,
                "count": item.count,
            }
            for item in extraction.duplicate_cells
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        action="append",
        required=True,
        help="WritingBench eval JSONL. Repeat to merge multiple files.",
    )
    parser.add_argument("--mode", default="ours_no_validation")
    parser.add_argument("--baseline-mode", default="few_shot_examples_only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--diagnosis",
        type=Path,
        default=None,
        help="Optional per-criterion diagnosis JSON used only to verify loss count.",
    )
    parser.add_argument(
        "--expected-loss-count",
        type=int,
        default=None,
        help="Fail if the paired successful rows do not contain this many losses.",
    )
    parser.add_argument(
        "--allow-duplicate-cells",
        action="store_true",
        help="Exclude duplicate successful cells instead of failing closed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing --out.",
    )
    args = parser.parse_args()

    try:
        rows = _load_rows(args.eval)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    extraction = extract_losing_task_judge_reasons(
        rows,
        mode=args.mode,
        baseline_mode=args.baseline_mode,
        losing_only=True,
    )
    if extraction.duplicate_cells and not args.allow_duplicate_cells:
        print(
            f"error: found {len(extraction.duplicate_cells)} duplicate successful cells",
            file=sys.stderr,
        )
        return 2
    expected_loss_count = args.expected_loss_count
    if args.diagnosis is not None:
        try:
            diagnosis_loss_count = _diagnosis_loss_count(args.diagnosis)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if diagnosis_loss_count is not None:
            expected_loss_count = diagnosis_loss_count
    if expected_loss_count is not None and extraction.loss_count != expected_loss_count:
        print(
            (
                f"error: loss_count={extraction.loss_count} does not match "
                f"expected {expected_loss_count}"
            ),
            file=sys.stderr,
        )
        return 2

    summary = _summary_payload(args, extraction)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    _write_jsonl(args.out, extraction.rows)
    print(f"wrote {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
