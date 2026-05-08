#!/usr/bin/env python3
"""Update public example-derived extraction memory from skill rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.extraction_memory import (  # noqa: E402
    entries_from_skill_rows,
    load_memory_entries,
    summarize_memory_entries,
    write_memory_entries,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skills",
        type=Path,
        action="append",
        default=[Path("runs/skill_mvp.qwen.mvp.jsonl")],
        help="Skill induction JSONL. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/memory/extraction_memory.v1.jsonl"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge with existing --out entries instead of replacing the file.",
    )
    args = parser.parse_args()

    rows = []
    for path in args.skills:
        rows.extend(load_jsonl(path))
    entries = entries_from_skill_rows(rows)
    if args.append and args.out.exists():
        entries = load_memory_entries(args.out) + entries
    write_memory_entries(args.out, entries)
    summary = summarize_memory_entries(entries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
