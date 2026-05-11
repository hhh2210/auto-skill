#!/usr/bin/env python3
"""Prepare aliases, inputs, and reports for three-metric ablations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402

ScoreMap = dict[str, dict[tuple[str, str], float]]
StatusMap = dict[str, Counter[str]]


@dataclass(frozen=True)
class ModeSpec:
    label: str
    task_source: str | None
    task_mode: str | None
    skill_source: str | None
    skill_mode: str | None
    note: str = "-"


MODE_SPECS = (
    ModeSpec(
        label="prompt_only",
        task_source="base_eval",
        task_mode="prompt_only",
        skill_source=None,
        skill_mode=None,
        note="no skill artifact",
    ),
    ModeSpec(
        label="example_only",
        task_source="base_eval",
        task_mode="few_shot_examples_only",
        skill_source=None,
        skill_mode=None,
        note="anchor; no skill artifact",
    ),
    ModeSpec(
        label="one_shot_skill",
        task_source="base_eval",
        task_mode="one_shot_skill_from_examples",
        skill_source="baseline_skills",
        skill_mode="one_shot_skill_from_examples",
    ),
    ModeSpec(
        label="minimal_no_memory",
        task_source="base_eval",
        task_mode="ours_no_validation",
        skill_source="no_memory_skills",
        skill_mode="auto_skill_minimal",
    ),
    ModeSpec(
        label="minimal_mem_within",
        task_source="within_eval",
        task_mode="ours_no_validation",
        skill_source="within_skills",
        skill_mode="auto_skill_minimal",
    ),
    ModeSpec(
        label="minimal_mem_xpack_holdout",
        task_source="xpack_eval",
        task_mode="ours_no_validation",
        skill_source="xpack_skills",
        skill_mode="auto_skill_minimal",
    ),
    ModeSpec(
        label="feature_loo_full",
        task_source="feature_eval",
        task_mode="auto_skill",
        skill_source="feature_skills",
        skill_mode="auto_skill_ours_full",
        note="main-branch feature extraction + LOO + validation-aware merge",
    ),
)


def spec_by_label() -> dict[str, ModeSpec]:
    return {spec.label: spec for spec in MODE_SPECS}


def require_jsonl(path: Path, *, role: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required {role}: {path}")
    return load_jsonl(path)


def require_text(path: Path, *, role: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing required {role}: {path}")
    return path.read_text(encoding="utf-8")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def fmt(value: object, digits: int = 3) -> str:
    if not isinstance(value, int | float):
        return "-"
    return f"{float(value):.{digits}f}"


def cmd_minimal_alias(args: argparse.Namespace) -> int:
    rows = []
    seen_pack_ids: set[str] = set()
    skipped = Counter()
    for row in require_jsonl(args.skills, role="minimal skill rows"):
        pack_id = str(row.get("pack_id") or "")
        if row.get("status") != "success":
            skipped["non_success"] += 1
            continue
        if row.get("mode") != "auto_skill_minimal":
            skipped["wrong_mode"] += 1
            continue
        if not isinstance(row.get("skill_md"), str) or not str(row.get("skill_md")).strip():
            skipped["missing_skill_md"] += 1
            continue
        if not pack_id:
            skipped["missing_pack_id"] += 1
            continue
        if pack_id in seen_pack_ids:
            print(f"error: duplicate minimal skill row for pack_id={pack_id}", file=sys.stderr)
            return 3
        seen_pack_ids.add(pack_id)
        copied = dict(row)
        copied["mode"] = "auto_skill_feature_driven_no_validation"
        metadata = dict(copied.get("metadata") or {})
        metadata["minimal_eval_alias_for"] = "auto_skill_minimal"
        metadata["minimal_memory_variant"] = args.variant
        copied["metadata"] = metadata
        rows.append(copied)
    write_jsonl(args.out, rows)
    skipped_text = ",".join(f"{key}:{value}" for key, value in sorted(skipped.items())) or "-"
    print(f"alias_rows={len(rows)} skipped={skipped_text} out={args.out}")
    return 0


def _path_for_source(args: argparse.Namespace, source: str) -> Path:
    return getattr(args, source.replace("-", "_"))


def add_eval_alias(rows: list[dict[str, Any]], args: argparse.Namespace, spec: ModeSpec) -> None:
    if spec.task_source is None or spec.task_mode is None:
        return
    path = _path_for_source(args, spec.task_source)
    matched = 0
    success = 0
    seen_success: set[tuple[str, str]] = set()
    for row in require_jsonl(path, role=f"{spec.label} task eval"):
        if row.get("mode") == spec.task_mode:
            matched += 1
            if row.get("status") == "success":
                key = (str(row.get("pack_id") or ""), str(row.get("task_id") or ""))
                if not all(key):
                    raise ValueError(f"{spec.label} eval row missing pack_id/task_id in {path}")
                if key in seen_success:
                    raise ValueError(f"duplicate {spec.label} eval cell {key} in {path}")
                seen_success.add(key)
                success += 1
            copied = dict(row)
            copied["mode"] = spec.label
            rows.append(copied)
    if matched == 0:
        raise ValueError(f"missing mode {spec.task_mode!r} for {spec.label} in {path}")
    if success == 0:
        raise ValueError(f"mode {spec.task_mode!r} for {spec.label} has no success rows in {path}")


def cmd_pairwise_input(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for spec in MODE_SPECS:
        add_eval_alias(rows, args, spec)
    write_jsonl(args.out, rows)
    print(f"pairwise_input_rows={len(rows)} out={args.out}")
    return 0


def add_skill_alias(rows: list[dict[str, Any]], args: argparse.Namespace, spec: ModeSpec) -> None:
    if spec.skill_source is None or spec.skill_mode is None:
        return
    path = _path_for_source(args, spec.skill_source)
    matched = 0
    seen_pack_ids: set[str] = set()
    for row in require_jsonl(path, role=f"{spec.label} skill rows"):
        if row.get("mode") == spec.skill_mode and row.get("status") == "success":
            pack_id = str(row.get("pack_id") or "")
            skill_md = row.get("skill_md")
            if not pack_id:
                raise ValueError(f"{spec.label} skill row missing pack_id in {path}")
            if pack_id in seen_pack_ids:
                raise ValueError(
                    f"duplicate {spec.label} skill row for pack_id={pack_id} in {path}"
                )
            if not isinstance(skill_md, str) or not skill_md.strip():
                raise ValueError(f"{spec.label} skill row missing skill_md for pack_id={pack_id}")
            seen_pack_ids.add(pack_id)
            matched += 1
            copied = dict(row)
            copied["mode"] = spec.label
            rows.append(copied)
    if matched == 0:
        raise ValueError(
            f"missing successful skill mode {spec.skill_mode!r} for {spec.label} in {path}"
        )


def cmd_skill_quality_input(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for spec in MODE_SPECS:
        add_skill_alias(rows, args, spec)
    write_jsonl(args.out, rows)
    print(f"skill_quality_input_rows={len(rows)} out={args.out}")
    return 0


def load_scores(sources: dict[str, tuple[Path, str]]) -> tuple[ScoreMap, StatusMap]:
    scores: ScoreMap = {}
    statuses: StatusMap = {}
    for label, (path, mode) in sources.items():
        scores[label] = {}
        statuses[label] = Counter()
        matched = 0
        for row in require_jsonl(path, role=f"{label} task eval"):
            if row.get("mode") != mode:
                continue
            matched += 1
            statuses[label][str(row.get("status") or "unknown")] += 1
            score = row.get("overall_score")
            if row.get("status") == "success":
                if isinstance(score, bool) or not isinstance(score, int | float):
                    raise ValueError(
                        f"successful {label} row missing numeric overall_score in {path}"
                    )
                if not math.isfinite(float(score)):
                    raise ValueError(
                        f"successful {label} row has non-finite overall_score in {path}"
                    )
                key = (str(row.get("pack_id") or ""), str(row.get("task_id") or ""))
                if key in scores[label]:
                    raise ValueError(f"duplicate successful {label} task eval cell {key} in {path}")
                scores[label][key] = float(score)
        if matched == 0:
            raise ValueError(f"missing mode {mode!r} for {label} in {path}")
        if not scores[label]:
            raise ValueError(f"mode {mode!r} for {label} has no successful scored rows in {path}")
    return scores, statuses


def paired_delta(
    scores: dict[str, dict[tuple[str, str], float]], candidate: str, baseline: str
) -> dict[str, object]:
    keys = sorted(set(scores[candidate]) & set(scores[baseline]))
    deltas = [scores[candidate][key] - scores[baseline][key] for key in keys]
    return {
        "n": len(keys),
        "mean_delta": mean(deltas),
        "wins": sum(1 for delta in deltas if delta > 0),
        "losses": sum(1 for delta in deltas if delta < 0),
        "ties": sum(1 for delta in deltas if delta == 0),
    }


def memory_stats(path_by_label: dict[str, Path]) -> list[dict[str, object]]:
    base_tokens: dict[str, float] = {}
    rows = []
    for label, path in path_by_label.items():
        entries: list[float] = []
        positives: list[float] = []
        negatives: list[float] = []
        tokens: list[float] = []
        ratios: list[float] = []
        skill_rows = require_jsonl(path, role=f"{label} skill rows")
        for row in skill_rows:
            report = row.get("memory_report")
            report = report if isinstance(report, dict) else {}
            token_count = report.get("extraction_prompt_tokens") or row.get(
                "extraction_prompt_tokens"
            )
            pack_id = str(row.get("pack_id") or "")
            if isinstance(token_count, int | float):
                tokens.append(float(token_count))
                if label == "minimal_no_memory":
                    base_tokens[pack_id] = float(token_count)
                elif pack_id in base_tokens and base_tokens[pack_id]:
                    ratios.append(float(token_count) / base_tokens[pack_id])
            entries.append(float(report.get("entry_count") or 0))
            positives.append(float(report.get("positive_count") or 0))
            negatives.append(float(report.get("negative_count") or 0))
        rows.append(
            {
                "label": label,
                "rows": len(skill_rows),
                "entry_median": median(entries),
                "entry_p90": percentile(entries, 0.9),
                "positive_median": median(positives),
                "negative_median": median(negatives),
                "token_median": median(tokens),
                "token_p90": percentile(tokens, 0.9),
                "ratio_median": median(ratios) if ratios else 1.0,
                "ratio_p90": percentile(ratios, 0.9) if ratios else 1.0,
            }
        )
    return rows


def cmd_report(args: argparse.Namespace) -> int:
    mode_sources = {
        spec.label: (_path_for_source(args, spec.task_source), spec.task_mode)
        for spec in MODE_SPECS
        if spec.task_source is not None and spec.task_mode is not None
    }
    mode_order = [spec.label for spec in MODE_SPECS]
    specs = spec_by_label()
    scores, statuses = load_scores(mode_sources)
    pairwise = json.loads(require_text(args.pairwise_summary, role="pairwise summary"))
    paired = (
        pairwise.get("paired_task_level", {})
        if isinstance(pairwise.get("paired_task_level"), dict)
        else {}
    )
    call_level = (
        pairwise.get("call_level", {})
        if isinstance(pairwise.get("call_level"), dict)
        else {}
    )
    quality = json.loads(require_text(args.skill_quality_summary, role="skill-quality summary"))
    quality_by_mode = quality.get("by_mode", {}) if isinstance(quality.get("by_mode"), dict) else {}

    lines = [
        "# Auto-Skills Three-Metric Ablation (30WB)",
        "",
        "## Metric Registry",
        "",
        "| Metric type | Question | Implementation | Primary readout | Use |",
        "|---|---|---|---|---|",
        "| Benchmark task quality | Did the agent complete the heldout task well? | "
        "WritingBench official prompt, split solver/judge, MIMO judge | "
        "Mean score and paired delta | Main task-completion quality metric |",
        "| Pairwise example-likeness | Which output better matches reusable patterns "
        "from train examples? | Blind MIMO pairwise judge, anchor=`example_only`, "
        "swapped A/B order | Decisive win-rate vs anchor | Example-style transfer "
        "diagnostic |",
        "| Skill artifact quality | Is the induced `skill_md` faithful, reusable, "
        "effective, clear, and concise? | Ctx2Skill-style five-dimension MIMO judge "
        "over public train examples + `skill_md` | Mean 0-100 score | "
        "Skill artifact diagnostic |",
        "",
        "## Clean Summary Table",
        "",
        "| Mode | Task mean | Delta vs example_only | Pairwise decisive win-rate | "
        "Skill quality | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label in mode_order:
        delta = paired_delta(scores, label, "example_only") if label != "example_only" else None
        quality_value = quality_by_mode.get(label, {}).get("mean_skill_quality")
        notes = specs[label].note
        lines.append(
            f"| `{label}` | {fmt(mean(list(scores[label].values())))} | "
            f"{'-' if delta is None else fmt(delta['mean_delta'])} | "
            f"{fmt(paired.get(label, {}).get('candidate_win_rate_decisive_tasks'))} | "
            f"{'N/A' if quality_value is None else fmt(quality_value, 2)} | {notes} |"
        )

    lines += [
        "",
        "## Metric 1: Benchmark Task Quality",
        "",
        "| Mode | Cells | Success | Mean | Delta vs example_only | W/L/T | Non-success |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label in mode_order:
        delta = paired_delta(scores, label, "example_only") if label != "example_only" else None
        non_success = {key: value for key, value in statuses[label].items() if key != "success"}
        non_success_text = (
            ", ".join(f"{key}:{value}" for key, value in sorted(non_success.items())) or "-"
        )
        delta_text = "-"
        wlt_text = "-"
        if delta is not None:
            delta_text = f"{fmt(delta['mean_delta'])} (n={delta['n']})"
            wlt_text = f"{delta['wins']}/{delta['losses']}/{delta['ties']}"
        lines.append(
            f"| `{label}` | {sum(statuses[label].values())} | {len(scores[label])} | "
            f"{fmt(mean(list(scores[label].values())))} | "
            f"{delta_text} | {wlt_text} | {non_success_text} |"
        )

    lines += [
        "",
        "## Metric 2: Pairwise Example-Likeness",
        "",
        "| Candidate vs example_only | Paired tasks | Candidate wins | Anchor wins | "
        "Ties/splits | Decisive win-rate | All-task win-rate | "
        "Call-level decisive win-rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in [item for item in mode_order if item != "example_only"]:
        p = paired.get(label, {})
        c = call_level.get(label, {})
        if not p:
            lines.append(f"| `{label}` | not run | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| `{label}` | {p.get('paired_tasks', 0)} | {p.get('candidate_wins', 0)} | "
            f"{p.get('anchor_wins', 0)} | {p.get('ties_or_splits', 0)} | "
            f"{fmt(p.get('candidate_win_rate_decisive_tasks'))} | "
            f"{fmt(p.get('candidate_win_rate_all_tasks'))} | "
            f"{fmt(c.get('candidate_win_rate_decisive_calls'))} |"
        )

    lines += [
        "",
        "## Metric 3: Skill Artifact Quality",
        "",
        "| Mode | Skill rows | Success | Mean skill quality | Non-success |",
        "|---|---:|---:|---:|---|",
    ]
    for label in mode_order:
        item = quality_by_mode.get(label)
        if not item:
            reason = "no skill artifact" if label in {"prompt_only", "example_only"} else "not run"
            lines.append(f"| `{label}` | N/A | N/A | N/A | {reason} |")
            continue
        status_counts = item.get("status_counts", {})
        non_success = {key: value for key, value in status_counts.items() if key != "success"}
        non_success_text = (
            ", ".join(f"{key}:{value}" for key, value in sorted(non_success.items())) or "-"
        )
        lines.append(
            f"| `{label}` | {item.get('rows', 0)} | {item.get('success', 0)} | "
            f"{fmt(item.get('mean_skill_quality'), 2)} | {non_success_text} |"
        )

    lines += [
        "",
        "## Memory Load",
        "",
        "| Variant | Skill rows | Memory entries median / p90 | Positive median | "
        "Negative median | Extraction prompt tokens median / p90 | "
        "Token ratio median / p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in memory_stats(
        {
            "minimal_no_memory": args.no_memory_skills,
            "minimal_mem_within": args.within_skills,
            "minimal_mem_xpack_holdout": args.xpack_skills,
        }
    ):
        lines.append(
            f"| `{row['label']}` | {row['rows']} | "
            f"{fmt(row['entry_median'], 1)} / {fmt(row['entry_p90'], 1)} | "
            f"{fmt(row['positive_median'], 1)} | {fmt(row['negative_median'], 1)} | "
            f"{fmt(row['token_median'], 1)} / {fmt(row['token_p90'], 1)} | "
            f"{fmt(row['ratio_median'], 2)} / {fmt(row['ratio_p90'], 2)} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    alias = sub.add_parser("minimal-alias")
    alias.add_argument("--skills", type=Path, required=True)
    alias.add_argument("--out", type=Path, required=True)
    alias.add_argument("--variant", required=True)
    alias.set_defaults(func=cmd_minimal_alias)

    pairwise = sub.add_parser("pairwise-input")
    pairwise.add_argument("--base-eval", type=Path, required=True)
    pairwise.add_argument("--within-eval", type=Path, required=True)
    pairwise.add_argument("--xpack-eval", type=Path, required=True)
    pairwise.add_argument("--feature-eval", type=Path, required=True)
    pairwise.add_argument("--out", type=Path, required=True)
    pairwise.set_defaults(func=cmd_pairwise_input)

    quality = sub.add_parser("skill-quality-input")
    quality.add_argument("--baseline-skills", type=Path, required=True)
    quality.add_argument("--no-memory-skills", type=Path, required=True)
    quality.add_argument("--within-skills", type=Path, required=True)
    quality.add_argument("--xpack-skills", type=Path, required=True)
    quality.add_argument("--feature-skills", type=Path, required=True)
    quality.add_argument("--out", type=Path, required=True)
    quality.set_defaults(func=cmd_skill_quality_input)

    report = sub.add_parser("report")
    report.add_argument("--base-eval", type=Path, required=True)
    report.add_argument("--within-eval", type=Path, required=True)
    report.add_argument("--xpack-eval", type=Path, required=True)
    report.add_argument("--feature-eval", type=Path, required=True)
    report.add_argument("--no-memory-skills", type=Path, required=True)
    report.add_argument("--within-skills", type=Path, required=True)
    report.add_argument("--xpack-skills", type=Path, required=True)
    report.add_argument("--pairwise-summary", type=Path, required=True)
    report.add_argument("--skill-quality-summary", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.set_defaults(func=cmd_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
