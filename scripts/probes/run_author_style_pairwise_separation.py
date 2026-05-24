#!/usr/bin/env python3
"""Run within/cross author-style pairwise separation over probe jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_skill.probes.author_style.pairwise_separation import (
    PairwiseSeparationConfig,
    run_pairwise_separation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--private-debug-out", type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--auth", type=Path, default=Path("~/.codex/auth.json").expanduser())
    parser.add_argument("--service-tier", default="priority")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=5000)
    parser.add_argument("--debug-sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run_pairwise_separation(PairwiseSeparationConfig(**vars(parse_args())))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
