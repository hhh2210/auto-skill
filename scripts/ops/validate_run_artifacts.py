#!/usr/bin/env python3
"""Validate generated-output, skill-induction, and eval JSONL artifacts."""

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
from auto_skill.schemas import SchemaValidationError, validate_artifact_rows  # noqa: E402


def validate_path(path: Path, *, kind: str) -> dict[str, Any]:
    rows = load_jsonl(path)
    validate_artifact_rows(rows, kind=kind, label=str(path))
    return {"path": str(path), "kind": kind, "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-outputs", type=Path, action="append", default=[])
    parser.add_argument("--skills", type=Path, action="append", default=[])
    parser.add_argument("--eval", type=Path, action="append", default=[])
    args = parser.parse_args()

    targets: list[tuple[Path, str]] = []
    targets.extend((path, "generated_outputs") for path in args.generated_outputs)
    targets.extend((path, "skills") for path in args.skills)
    targets.extend((path, "eval") for path in args.eval)
    if not targets:
        print("error: no artifacts provided", file=sys.stderr)
        return 2

    summaries = []
    try:
        for path, kind in targets:
            summaries.append(validate_path(path, kind=kind))
    except (OSError, SchemaValidationError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"validated": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
