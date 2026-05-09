#!/usr/bin/env python3
"""Export latest successful rows from an append-only generation JSONL log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.generated_outputs import (  # noqa: E402
    latest_successful_generation_rows,
    require_object_rows,
)
from auto_skill.schemas import SchemaValidationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations",
        type=Path,
        required=True,
        help="Append-only generated desired outputs JSONL.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL containing latest rows that are success + finish_reason=stop.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional summary JSON path.",
    )
    parser.add_argument(
        "--expect-successes",
        type=int,
        help="Fail unless exactly this many latest successful rows are exported.",
    )
    parser.add_argument(
        "--allow-incomplete-latest",
        action="store_true",
        help=(
            "Exit 0 even if some latest rows are non-success. By default the "
            "command fails closed when latest status counts include failures."
        ),
    )
    args = parser.parse_args()

    rows = load_jsonl(args.generations)
    try:
        require_object_rows(rows, label=str(args.generations))
    except SchemaValidationError as exc:
        print(f"error: invalid generation log: {exc}", file=sys.stderr)
        return 2
    successful, latest_status_counts = latest_successful_generation_rows(rows)
    latest_rows = sum(latest_status_counts.values())
    summary = {
        "source": str(args.generations),
        "out": str(args.out),
        "input_rows": len(rows),
        "latest_rows": latest_rows,
        "exported_success_rows": len(successful),
        "non_exported_latest_rows": latest_rows - len(successful),
        "latest_status_counts": latest_status_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.expect_successes is not None and len(successful) != args.expect_successes:
        print(
            f"error: expected {args.expect_successes} successful latest rows, "
            f"got {len(successful)}",
            file=sys.stderr,
        )
        return 3
    non_exported_latest = latest_rows - len(successful)
    if non_exported_latest and not args.allow_incomplete_latest:
        print(
            f"error: latest generation log still has {non_exported_latest} "
            "non-exportable rows",
            file=sys.stderr,
        )
        return 4
    write_jsonl(args.out, successful)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
