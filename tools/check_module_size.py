"""Fail when new Python modules exceed their tiered size budget.

Limits are tiered by path so that library code stays tight while CLI
orchestrators in ``scripts/`` are allowed to aggregate more glue. Use
``--limit`` only to force a single override (debug / one-off scans);
default runs apply the per-tier table below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# (path prefix, soft limit). Longest matching prefix wins.
DEFAULT_TIERS: tuple[tuple[str, int], ...] = (
    ("src", 400),
    ("scripts", 700),
    ("tools", 400),
)
DEFAULT_ROOTS = tuple(prefix for prefix, _ in DEFAULT_TIERS)

# Existing debt. Files here may shrink, but new over-limit modules should not be
# added without an explicit refactor or a conscious update to this list. Each
# entry is a known-large module that pre-dates the current tiered budget.
GRANDFATHERED = {
    # scripts/ over 700
    "scripts/eval/run_heldout_eval.py",
    "scripts/eval/run_writingbench_official_eval.py",
    "scripts/skills/run_skill_mvp.py",
    # src/ over 400
    "src/auto_skill/cleaning/author_style/audit.py",
    "src/auto_skill/cleaning/author_style/quality/report.py",
    "src/auto_skill/cleaning/author_style/selection.py",
    "src/auto_skill/cleaning/author_style/sources.py",
    "src/auto_skill/cleaning/author_style/stratified_eval.py",
    "src/auto_skill/cleaning/benchmark_flow.py",
    "src/auto_skill/diagnostics/coding_style.py",
    "src/auto_skill/diagnostics/compression.py",
    "src/auto_skill/diagnostics/criterion.py",
    "src/auto_skill/diagnostics/negative_transfer.py",
    "src/auto_skill/metrics/author_style_reference.py",
    "src/auto_skill/metrics/readiness.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Force a single limit for every scanned file (overrides tiers).",
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Scan only these roots. Defaults to the tiered roots.",
    )
    return parser.parse_args()


def module_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    return sorted(root.rglob("*.py")) if root.exists() else []


def limit_for(rel: str, override: int | None) -> int | None:
    if override is not None:
        return override
    best: tuple[int, int] | None = None
    for prefix, limit in DEFAULT_TIERS:
        prefix_with_sep = prefix.rstrip("/") + "/"
        if rel == prefix or rel.startswith(prefix_with_sep):
            score = len(prefix_with_sep)
            if best is None or score > best[0]:
                best = (score, limit)
    return best[1] if best else None


def main() -> int:
    args = parse_args()
    roots = [Path(root) for root in (args.roots or DEFAULT_ROOTS)]
    failures: list[tuple[int, int, str]] = []
    for path in [item for root in roots for item in iter_python_files(root)]:
        rel = path.as_posix()
        if rel in GRANDFATHERED:
            continue
        limit = limit_for(rel, args.limit)
        if limit is None:
            continue
        lines = module_lines(path)
        if lines > limit:
            failures.append((lines, limit, rel))
    if not failures:
        return 0
    print("Python modules over their tier budget and not grandfathered:")
    for lines, limit, rel in sorted(failures, reverse=True):
        print(f"{lines:5d} (limit {limit}) {rel}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
