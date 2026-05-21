#!/usr/bin/env python3
"""Audit cleaned benchmark artifacts against docs/benchmark_flow.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.benchmark_flow import audit_benchmark_flow  # noqa: E402
from auto_skill.example_packs import load_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("artifacts/splits/fewshot_splits.jsonl"),
    )
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument(
        "--private-eval",
        type=Path,
        default=Path("artifacts/private/example_private_eval.jsonl"),
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        help="Optional example-generation job JSONL to verify generation provenance.",
    )
    parser.add_argument(
        "--generated-outputs",
        type=Path,
        help="Optional generated desired-output JSONL to verify frozen output provenance.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = audit_benchmark_flow(
        splits=load_jsonl(args.splits),
        packs=load_jsonl(args.packs),
        private_rows=load_jsonl(args.private_eval),
        generation_jobs=load_jsonl(args.jobs) if args.jobs else None,
        generated_rows=load_jsonl(args.generated_outputs) if args.generated_outputs else None,
    )
    report = {
        "schema_version": "benchmark-flow-audit/v1",
        "status": "ok" if result.ok else "failed",
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
