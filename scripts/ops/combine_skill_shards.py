#!/usr/bin/env python3
"""Combine successful per-pack skill rows from shard outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write available successful rows even when shard files or packs are missing.",
    )
    args = parser.parse_args()
    if args.shards <= 0:
        print("error: --shards must be positive", file=sys.stderr)
        return 2

    by_cell = {}
    missing_shards = []
    for index in range(args.shards):
        path = Path(f"{args.prefix}.shard{index}.jsonl")
        if not path.exists():
            missing_shards.append(str(path))
            continue
        for row in load_jsonl(path):
            if row.get("status") == "success" and row.get("pack_id") and row.get("mode"):
                key = (str(row["pack_id"]), str(row["mode"]))
                if key in by_cell:
                    print(f"error: duplicate successful skill row for {key}", file=sys.stderr)
                    return 3
                by_cell[key] = row

    packs = load_jsonl(args.packs)
    mode_order = [
        "one_shot_skill_from_examples",
        "auto_skill_minimal",
        "auto_skill_feature_driven_no_validation",
        "auto_skill_ours_full",
    ]
    ordered = []
    missing = []
    for pack in packs:
        pack_id = str(pack.get("pack_id"))
        pack_rows = [by_cell[(pack_id, mode)] for mode in mode_order if (pack_id, mode) in by_cell]
        pack_rows.extend(
            row
            for (row_pack_id, mode), row in sorted(by_cell.items())
            if row_pack_id == pack_id and mode not in mode_order
        )
        if pack_rows:
            ordered.extend(pack_rows)
        else:
            missing.append(pack_id)
    if (missing_shards or missing) and not args.allow_partial:
        if missing_shards:
            print("error: missing shard files: " + ",".join(missing_shards), file=sys.stderr)
        if missing:
            print("error: missing successful pack rows: " + ",".join(missing), file=sys.stderr)
        return 4
    write_jsonl(args.out, ordered)
    print(f"combined_success_rows={len(ordered)} missing_packs={len(missing)} out={args.out}")
    if missing:
        print("missing_pack_ids=" + ",".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
