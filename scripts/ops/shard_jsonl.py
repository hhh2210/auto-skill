#!/usr/bin/env python3
"""Split a JSONL file into deterministic modulo shards."""

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()

    if args.shards <= 0:
        print("error: --shards must be positive", file=sys.stderr)
        return 2
    rows = load_jsonl(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.shards):
        shard = [row for row_index, row in enumerate(rows) if row_index % args.shards == index]
        write_jsonl(args.out_dir / f"{args.prefix}.shard{index}.packs.jsonl", shard)
    print(f"sharded_rows={len(rows)} shards={args.shards} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
