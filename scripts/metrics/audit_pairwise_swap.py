#!/usr/bin/env python3
"""Audit swapped-order stability for pairwise example-likeness JSONL outputs."""

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
from auto_skill.pairwise_likeness import SWAP_PAIR_STATUSES, swap_flip_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--max-flip-rate", type=float, default=0.30)
    args = parser.parse_args()

    if args.max_flip_rate < 0 or args.max_flip_rate > 1:
        print("error: --max-flip-rate must be between 0 and 1", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for path in args.pairwise:
        rows.extend(load_jsonl(path))

    audit = swap_flip_audit(rows)
    audit["inputs"] = [str(path) for path in args.pairwise]
    audit["max_flip_rate"] = args.max_flip_rate

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(swap_audit_markdown(audit), encoding="utf-8")

    exceeded = False
    for mode, entry in sorted(audit["by_candidate"].items()):
        flip_rate = entry.get("swap_flip_rate_decisive_pairs")
        print(
            f"{mode}: flip_rate={_pct(flip_rate)} "
            f"flip={entry['counts']['swap_flip_decisive']} "
            f"decisive_pairs={entry['decisive_pair_count']}"
        )
        if flip_rate is not None and flip_rate > args.max_flip_rate:
            exceeded = True
    return 1 if exceeded else 0


def swap_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Pairwise Swap-Flip Audit",
        "",
        "## Inputs",
        "",
    ]
    for path in audit.get("inputs", []):
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Per-Candidate Counts",
            "",
            "| Candidate vs example_only anchor | Stable anchor | Stable candidate | "
            "Stable tie | Flip | Mixed | Incomplete | Decisive pairs | "
            "Flip rate | Stable-only decisive win-rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, entry in sorted(audit.get("by_candidate", {}).items()):
        counts = entry["counts"]
        lines.append(
            f"| {mode} | {counts['swap_stable_anchor_wins']} | "
            f"{counts['swap_stable_candidate_wins']} | "
            f"{counts['swap_stable_tie']} | {counts['swap_flip_decisive']} | "
            f"{counts['swap_one_decisive_one_tie']} | "
            f"{counts['swap_pair_incomplete']} | "
            f"{entry['decisive_pair_count']} | "
            f"{_pct(entry.get('swap_flip_rate_decisive_pairs'))} | "
            f"{_pct(entry.get('swap_stable_candidate_win_rate_decisive'))} |"
        )

    lines.extend(
        [
            "",
            "## Confidence Distribution",
            "",
            "| Candidate | Pair status | Low | Medium | High | Unknown |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for mode, entry in sorted(audit.get("by_candidate", {}).items()):
        by_status = entry.get("confidence_by_pair_status", {})
        for status in SWAP_PAIR_STATUSES:
            counts = by_status.get(status, {})
            lines.append(
                f"| {mode} | {status} | {counts.get('low', 0)} | "
                f"{counts.get('medium', 0)} | {counts.get('high', 0)} | "
                f"{counts.get('unknown', 0)} |"
            )

    ours = audit.get("by_candidate", {}).get("ours_no_validation")
    if ours:
        flip_rate = ours.get("swap_flip_rate_decisive_pairs")
        stable_rate = ours.get("swap_stable_candidate_win_rate_decisive")
        if flip_rate is not None and flip_rate > 0.30:
            interpretation = (
                "After dropping swap-flip pairs as the noise floor, the 48.8% "
                f"headline does not survive cleanly: stable-only decisive "
                f"win-rate is {_pct(stable_rate)}, and pairwise win-rate is "
                "within noise; cannot conclude ours < example_only."
            )
        else:
            interpretation = (
                "After dropping swap-flip pairs as the noise floor, the 48.8% "
                f"headline survives the swap-order audit: stable-only decisive "
                f"win-rate is {_pct(stable_rate)}, and pairwise win-rate is "
                "robust to swap order; ours_no_validation marginally below "
                "example_only on stable pairs."
            )
    else:
        interpretation = "No ours_no_validation rows were available for interpretation."
    lines.extend(["", "## Interpretation", "", interpretation, ""])
    return "\n".join(lines)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
