#!/usr/bin/env python3
"""Validate few-shot split JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.data_cleaning import (  # noqa: E402
    ValidationError,
    load_jsonl,
    summarize_splits,
    validate_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to few-shot split JSONL")
    args = parser.parse_args()

    try:
        rows = load_jsonl(args.path)
        validate_splits(rows)
    except (OSError, ValidationError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summarize_splits(rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
