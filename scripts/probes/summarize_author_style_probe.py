#!/usr/bin/env python3
"""Summarize the T2 author-style probe into JSON plus a short Markdown note."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_skill.probes.author_style.probe_summary import (
    ProbeSummaryConfig,
    write_probe_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-oracle", type=Path, required=True)
    parser.add_argument("--qwen-crosscheck", type=Path)
    parser.add_argument("--mimo-crosscheck", type=Path)
    parser.add_argument("--pairwise-summary", type=Path)
    parser.add_argument("--legacy-strict", type=Path, action="append", default=[])
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    summary = write_probe_summary(ProbeSummaryConfig(**vars(parse_args())))
    print(json.dumps(summary["main_oracle"]["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
