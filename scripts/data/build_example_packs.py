#!/usr/bin/env python3
"""Construct user-visible example packs from few-shot benchmark splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import (  # noqa: E402
    build_example_pack_artifacts,
    load_jsonl,
    summarize_packs,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("artifacts/splits/fewshot_splits.jsonl"),
    )
    parser.add_argument(
        "--packs-out",
        type=Path,
        default=Path("artifacts/packs/example_packs.jsonl"),
    )
    parser.add_argument(
        "--private-out",
        type=Path,
        default=Path("artifacts/private/example_private_eval.jsonl"),
    )
    parser.add_argument(
        "--jobs-out",
        type=Path,
        default=Path("artifacts/jobs/example_generation_jobs.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/reports/example_pack_summary.md"),
    )
    parser.add_argument(
        "--max-material-chars",
        type=int,
        default=16_000,
        help="Maximum extracted material characters included in each generation prompt.",
    )
    args = parser.parse_args()

    splits = load_jsonl(args.splits)
    packs, private_rows, jobs = build_example_pack_artifacts(
        splits,
        max_material_chars=args.max_material_chars,
    )

    write_jsonl(args.packs_out, packs)
    write_jsonl(args.private_out, private_rows)
    write_jsonl(args.jobs_out, jobs)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(summarize_packs(packs, jobs), encoding="utf-8")

    print(f"Wrote {len(packs)} clean example packs to {args.packs_out}")
    print(f"Wrote {len(jobs)} desired-output generation jobs to {args.jobs_out}")
    print(f"Wrote private eval metadata to {args.private_out}")
    print(f"Wrote summary to {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
