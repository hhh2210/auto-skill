"""Fail when new Python modules exceed the local size budget."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_LIMIT = 300
DEFAULT_ROOTS = ("src", "scripts", "tools")

# Existing debt. Files here may shrink, but new over-limit modules should not be
# added without an explicit refactor or a conscious update to this list.
GRANDFATHERED = {
    "scripts/data/build_fewshot_splits.py",
    "scripts/data/inspect_benchmarks.py",
    "scripts/data/run_generation_jobs.py",
    "scripts/eval/audit_train_examples.py",
    "scripts/eval/run_heldout_eval.py",
    "scripts/eval/run_presentbench_official_judge.py",
    "scripts/eval/run_writingbench_official_eval.py",
    "scripts/metrics/aggregate_failure_modes.py",
    "scripts/metrics/export_judge_disagreements.py",
    "scripts/metrics/run_author_style_reference_retrieval.py",
    "scripts/metrics/run_grounding_eval.py",
    "scripts/metrics/run_pairwise_likeness.py",
    "scripts/metrics/run_pattern_similarity_eval.py",
    "scripts/metrics/run_self_consistency_metric.py",
    "scripts/metrics/run_skill_quality_eval.py",
    "scripts/metrics/summarize_mvp_metrics.py",
    "scripts/metrics/summarize_per_criterion_delta.py",
    "scripts/ops/prepare_three_metric_ablation.py",
    "scripts/ops/report_expanded_cleaning_status.py",
    "scripts/skills/run_skill_mvp.py",
    "src/auto_skill/cleaning/author_style/audit.py",
    "src/auto_skill/cleaning/author_style/cluster_profile.py",
    "src/auto_skill/cleaning/author_style/packs.py",
    "src/auto_skill/cleaning/author_style/pipeline.py",
    "src/auto_skill/cleaning/author_style/quality/report.py",
    "src/auto_skill/cleaning/author_style/selection.py",
    "src/auto_skill/cleaning/author_style/sources.py",
    "src/auto_skill/cleaning/author_style/stratified_eval.py",
    "src/auto_skill/cleaning/benchmark_flow.py",
    "src/auto_skill/cleaning/packs.py",
    "src/auto_skill/diagnostics/coding_style.py",
    "src/auto_skill/diagnostics/compression.py",
    "src/auto_skill/diagnostics/criterion.py",
    "src/auto_skill/diagnostics/negative_transfer.py",
    "src/auto_skill/memory/extraction.py",
    "src/auto_skill/metrics/author_style_reference.py",
    "src/auto_skill/metrics/readiness.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--root", action="append", dest="roots")
    return parser.parse_args()


def module_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    return sorted(root.rglob("*.py")) if root.exists() else []


def main() -> int:
    args = parse_args()
    roots = [Path(root) for root in (args.roots or DEFAULT_ROOTS)]
    failures = []
    for path in [item for root in roots for item in iter_python_files(root)]:
        rel = path.as_posix()
        lines = module_lines(path)
        if lines > args.limit and rel not in GRANDFATHERED:
            failures.append((lines, rel))
    if not failures:
        return 0
    print(f"Python modules over {args.limit} lines and not grandfathered:")
    for lines, rel in sorted(failures, reverse=True):
        print(f"{lines:5d} {rel}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
