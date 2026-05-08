#!/usr/bin/env python3
"""Apply generated desired outputs to clean example packs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.generated_outputs import (  # noqa: E402
    apply_outputs_to_pack,
    index_successful_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, default=Path("artifacts/packs/example_packs.jsonl"))
    parser.add_argument(
        "--generations",
        type=Path,
        default=Path("artifacts/jobs/generated_desired_outputs.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/packs/example_packs.v1.jsonl"))
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 when some desired outputs are missing. Default is fail-closed.",
    )
    args = parser.parse_args()

    packs = load_jsonl(args.packs)
    generations = load_jsonl(args.generations)
    outputs_by_job_id = index_successful_outputs(generations)

    updated_packs = []
    total_applied = 0
    total_missing = 0
    total_rejected = 0
    for pack in packs:
        updated, applied, missing, rejected = apply_outputs_to_pack(pack, outputs_by_job_id)
        updated_packs.append(updated)
        total_applied += applied
        total_missing += missing
        total_rejected += rejected

    write_jsonl(args.out, updated_packs)
    print(f"Wrote {len(updated_packs)} packs to {args.out}")
    print(f"Applied desired outputs: {total_applied}")
    print(f"Missing desired outputs: {total_missing}")
    print(f"Rejected desired outputs: {total_rejected}")
    if total_rejected or (total_missing and not args.allow_missing):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
