#!/usr/bin/env python3
"""Summarize per-criterion deltas between two WritingBench eval modes.

Reads one or more WritingBench official-judge eval JSONL files and asks: in
the cells where ``--mode`` (e.g. ``ours_no_validation``) loses to
``--baseline-mode`` (e.g. ``few_shot_examples_only``), *which checklist
criterion* did it lose on?

The point is to recover task-specific diagnostic signal that the per-task
``overall_score`` (mean of 5 criteria) collapses away. WritingBench rubric
already encodes fine-grained dimensions per task, so we reuse those rather
than inventing task-agnostic constraint sub-scores.

Outputs:
  - JSON summary at ``--out`` (full per-task and bucket aggregates).
  - Optional human-readable Markdown at ``--report-md``.

This script does not call any LLM. It only post-processes existing rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.criterion_diagnosis import diagnose  # noqa: E402
from auto_skill.example_packs import load_jsonl  # noqa: E402


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing eval file: {path}")
        rows.extend(load_jsonl(path))
    return rows


def _format_bucket_table(buckets: list[dict[str, Any]]) -> str:
    if not buckets:
        return "_no buckets observed_\n"
    lines = [
        (
            "| Bucket | N | Mean Δ | Wins | Losses | Ties | "
            "Worst-pick on losing tasks | Worst-pick all | "
            "Worst-pick multi (losing) | Best-pick on winning tasks |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in buckets:
        mean_delta = bucket.get("mean_delta")
        mean_str = f"{mean_delta:+.2f}" if isinstance(mean_delta, (int, float)) else "n/a"
        lines.append(
            f"| {bucket['bucket']} | {bucket['appearances']} | {mean_str} | "
            f"{bucket['win_count']} | {bucket['loss_count']} | {bucket['tie_count']} | "
            f"**{bucket['worst_pick_count_on_losing_tasks']}** | "
            f"{bucket['worst_pick_count_all']} | "
            f"{bucket['worst_pick_count_on_losing_tasks_multilabel']} | "
            f"{bucket['best_pick_count_on_winning_tasks']} |"
        )
    return "\n".join(lines) + "\n"


def _format_task_table(tasks: list[dict[str, Any]], *, header: str) -> str:
    if not tasks:
        return f"_{header}: none_\n"
    lines = [
        f"### {header}",
        "",
        (
            "| Pack / task | Baseline | Mode | Δ | "
            "Worst criterion (Δ) | Primary bucket | Other matched buckets |"
        ),
        "|---|---:|---:|---:|---|---|---|",
    ]
    for task in tasks:
        worst = task.get("worst_criterion") or {}
        worst_name = worst.get("criterion") or "—"
        worst_delta = worst.get("delta")
        worst_str = (
            f"{worst_name} ({worst_delta:+.1f})"
            if isinstance(worst_delta, (int, float))
            else worst_name
        )
        primary = worst.get("primary_bucket") or "—"
        all_matched = worst.get("matched_buckets") or []
        other = [b for b in all_matched if b != primary]
        other_str = ", ".join(other) if other else "—"
        lines.append(
            f"| `{task['pack_id']}` / `{task['task_id']}` | "
            f"{task['baseline_overall']:.2f} | {task['mode_overall']:.2f} | "
            f"{task['overall_delta']:+.2f} | {worst_str} | "
            f"{primary} | {other_str} |"
        )
    return "\n".join(lines) + "\n"


def _format_criterion_table(
    criteria: list[dict[str, Any]], *, top_n: int = 15
) -> str:
    if not criteria:
        return "_no criteria observed_\n"
    head = criteria[:top_n]
    lines = [
        (
            "| Criterion | Primary bucket | Other matched | "
            "N | Mean Δ | Min Δ | Max Δ | W | L | T |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in head:
        primary = item["primary_bucket"]
        all_matched = item.get("matched_buckets") or []
        other = [b for b in all_matched if b != primary]
        other_str = ", ".join(other) if other else "—"
        lines.append(
            f"| {item['criterion']} | {primary} | {other_str} | "
            f"{item['appearances']} | {item['mean_delta']:+.2f} | "
            f"{item['min_delta']:+.1f} | {item['max_delta']:+.1f} | "
            f"{item['win_count']} | {item['loss_count']} | {item['tie_count']} |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(diagnosis: dict[str, Any]) -> str:
    paired = diagnosis["paired_overall"]
    skipped = diagnosis["skipped"]
    skipped_count = skipped.get("skipped_mismatched_criteria_count", 0)
    skipped_lines: list[str] = []
    if skipped_count:
        skipped_lines.append(
            f"- skipped tasks (mismatched criterion sets): **{skipped_count}**"
        )
        for entry in skipped["skipped_mismatched_criteria"][:8]:
            skipped_lines.append(
                f"  - `{entry['pack_id']}` / `{entry['task_id']}`: "
                f"baseline_only={entry['baseline_only']} mode_only={entry['mode_only']}"
            )
    else:
        skipped_lines.append("- skipped tasks (mismatched criterion sets): 0")
    duplicate_count = skipped.get("duplicate_success_cells_count", 0)
    if duplicate_count:
        skipped_lines.append(f"- duplicate success cells excluded: **{duplicate_count}**")
        for entry in skipped["duplicate_success_cells"][:8]:
            skipped_lines.append(
                f"  - `{entry['mode']}` / `{entry['pack_id']}` / `{entry['task_id']}`: "
                f"count={entry['count']} judge_models={entry['judge_models']}"
            )
    else:
        skipped_lines.append("- duplicate success cells excluded: 0")

    parts: list[str] = [
        f"# Per-criterion delta diagnosis: `{diagnosis['mode']}` vs `{diagnosis['baseline_mode']}`",
        "",
        "## Overall paired summary",
        "",
        f"- common cells: **{paired['common_cells']}**",
        (
            "- mean overall Δ: **"
            + (
                f"{paired['mean_overall_delta']:+.3f}"
                if isinstance(paired["mean_overall_delta"], (int, float))
                else "n/a"
            )
            + "**"
        ),
        (
            f"- W / L / T: **{paired['win_count']} / "
            f"{paired['loss_count']} / {paired['tie_count']}**"
        ),
        (
            "- min / max overall Δ: "
            + (
                f"{paired['min_overall_delta']:+.2f} / {paired['max_overall_delta']:+.2f}"
                if isinstance(paired.get("min_overall_delta"), (int, float))
                else "n/a"
            )
        ),
        *skipped_lines,
        "",
        "## Bucket diagnosis",
        "",
        (
            "Buckets are heuristic substring-keyword groupings of free-form "
            "criterion names; they are not a taxonomy. The primary bucket is "
            "first-match in `_KEYWORD_BUCKETS`; multi-label columns count "
            "every matched bucket. Cite **Worst-pick on losing tasks** "
            "(bolded column) for systemic-failure claims; the all-tasks "
            "column is a sanity check that includes tasks the candidate mode "
            "actually won."
        ),
        "",
        _format_bucket_table(diagnosis["bucket_diagnosis"]["buckets"]),
        "",
        f"_Disclaimer: {diagnosis['bucket_disclaimer']}_",
        "",
        "## Top per-criterion (worst mean delta first)",
        "",
        _format_criterion_table(diagnosis["criterion_aggregate"], top_n=15),
        "",
        _format_task_table(
            diagnosis["top_regressions"], header="Top regressions (mode loses most)"
        ),
        "",
        _format_task_table(
            diagnosis["top_improvements"], header="Top improvements (mode gains most)"
        ),
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        action="append",
        required=True,
        help="WritingBench official eval JSONL. Repeat to merge multiple files.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="Candidate mode (e.g. ours_no_validation).",
    )
    parser.add_argument(
        "--baseline-mode",
        required=True,
        help="Anchor mode to compare against (e.g. few_shot_examples_only).",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=-0.5,
        help="Per-criterion delta threshold to count as a loss (default -0.5).",
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.5,
        help="Per-criterion delta threshold to count as a win (default +0.5).",
    )
    parser.add_argument(
        "--top-n-regressions",
        type=int,
        default=10,
        help="Number of worst-regressed tasks to list (default 10).",
    )
    parser.add_argument(
        "--top-n-improvements",
        type=int,
        default=5,
        help="Number of best-improved tasks to list (default 5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSON output path for the structured diagnosis.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Optional Markdown report path for human inspection.",
    )
    parser.add_argument(
        "--allow-duplicate-cells",
        action="store_true",
        help=(
            "Allow repeated successful (mode, pack_id, task_id) cells by excluding "
            "them from pairing. Default is fail-closed to avoid mixing judge runs."
        ),
    )
    args = parser.parse_args()

    rows = _load_rows(args.eval)
    if not rows:
        print("error: no rows loaded from --eval inputs", file=sys.stderr)
        return 2

    mode_counts = Counter(str(row.get("mode") or "") for row in rows)
    if args.mode not in mode_counts:
        print(
            f"error: mode {args.mode!r} not present; observed modes: {sorted(mode_counts)}",
            file=sys.stderr,
        )
        return 2
    if args.baseline_mode not in mode_counts:
        print(
            (
                f"error: baseline mode {args.baseline_mode!r} not present; "
                f"observed modes: {sorted(mode_counts)}"
            ),
            file=sys.stderr,
        )
        return 2

    diagnosis = diagnose(
        rows,
        mode=args.mode,
        baseline_mode=args.baseline_mode,
        loss_threshold=args.loss_threshold,
        win_threshold=args.win_threshold,
        top_n_regressions=args.top_n_regressions,
        top_n_improvements=args.top_n_improvements,
    )
    duplicate_count = diagnosis["skipped"]["duplicate_success_cells_count"]
    if duplicate_count and not args.allow_duplicate_cells:
        print(
            (
                f"error: found {duplicate_count} duplicate successful score cells; "
                "pass separate eval files one at a time or use --allow-duplicate-cells "
                "to exclude duplicates from this diagnosis"
            ),
            file=sys.stderr,
        )
        for entry in diagnosis["skipped"]["duplicate_success_cells"][:8]:
            print(
                (
                    f"  duplicate: mode={entry['mode']} pack_id={entry['pack_id']} "
                    f"task_id={entry['task_id']} count={entry['count']} "
                    f"judge_models={entry['judge_models']}"
                ),
                file=sys.stderr,
            )
        return 2
    diagnosis["inputs"] = {
        "eval_files": [str(p) for p in args.eval],
        "mode_counts": dict(sorted(mode_counts.items())),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")

    if args.report_md is not None:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(_render_markdown(diagnosis), encoding="utf-8")
        print(f"wrote {args.report_md}")

    paired = diagnosis["paired_overall"]
    print(
        f"common_cells={paired['common_cells']} "
        f"mean_delta={paired['mean_overall_delta']} "
        f"W/L/T={paired['win_count']}/{paired['loss_count']}/{paired['tie_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
