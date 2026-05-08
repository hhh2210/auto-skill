#!/usr/bin/env python3
"""Build few-shot train/heldout splits for example-driven skill induction."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from inspect_benchmarks import (
        DEFAULT_PRESENTBENCH,
        DEFAULT_WRITINGBENCH,
        is_lfs_pointer,
        load_json,
        load_jsonl,
        load_yaml,
        presentbench_case_dirs,
        read_text,
    )
except ModuleNotFoundError:
    from scripts.data.inspect_benchmarks import (
        DEFAULT_PRESENTBENCH,
        DEFAULT_WRITINGBENCH,
        is_lfs_pointer,
        load_json,
        load_jsonl,
        load_yaml,
        presentbench_case_dirs,
        read_text,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWN_BAD_PRESENTBENCH_CASES = {
    # instructions.md asks for Chapter 10 System-Level I/O, but judge_prompt.json
    # checks Chapter 8 Exceptional Control Flow. This makes heldout scoring invalid.
    "education/CSAPP-Lectures_2015Fall/Lecture15",
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        pass
    try:
        path.resolve().relative_to(REPO_ROOT.parent.resolve())
        return os.path.relpath(path.resolve(), REPO_ROOT.resolve())
    except ValueError:
        return str(path)


def writing_task(row: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "source": "WritingBench",
        "source_id": row["index"],
        "domain": {
            "primary": row.get("domain1"),
            "secondary": row.get("domain2"),
            "language": row.get("lang"),
        },
        "task_input": row["query"],
        "supervision": {
            "type": "instance_specific_rubric",
            "items": row.get("checklist", []),
        },
        "judge": {
            "type": "llm_score_1_to_10_per_criterion",
            "prompt_file": repo_relative_path(root / "prompt.py"),
        },
        "expected_artifacts": ["written_response"],
    }


def present_task(case_dir: Path, root: Path) -> dict[str, Any]:
    rel_case = case_dir.relative_to(root)
    category = rel_case.parts[0]
    generation_dir = case_dir / "generation_task"
    judge_prompt_path = generation_dir / "judge_prompt.json"
    statistics_path = generation_dir / "statistics.yaml"
    materials = sorted(
        p for p in case_dir.iterdir() if p.is_file() and p.name.startswith("material")
    )
    judge_prompt = load_json(judge_prompt_path)
    checklists = {
        key: value
        for key, value in judge_prompt.items()
        if isinstance(value, list)
        and (
            key.startswith("material_independent_checklist_")
            or key.startswith("material_dependent_checklist_")
        )
    }
    checklist_counts = {key: len(value) for key, value in checklists.items()}
    prompt_prefixes = {
        key: value
        for key, value in judge_prompt.items()
        if isinstance(value, str) and key.endswith("_prefix")
    }
    common_prompt_path = root / category / "common_judge_prompt.json"
    weights_path = root / category / "judge_weights.yaml"
    return {
        "source": "PresentBench",
        "source_id": str(rel_case),
        "domain": {
            "primary": category,
            "secondary": "/".join(rel_case.parts[1:-1]) or rel_case.name,
        },
        "task_input": read_text(generation_dir / "instructions.md"),
        "supervision": {
            "type": "material_independent_and_dependent_checklists",
            "checklist_counts": checklist_counts,
            "checklists": checklists,
            "prompt_prefixes": prompt_prefixes,
            "judge_prompt_file": repo_relative_path(judge_prompt_path),
        },
        "judge": {
            "type": "binary_yes_no_checklist_with_boxed_answer",
            "domain_common_prompt_file": repo_relative_path(common_prompt_path),
            "domain_common_prompt": (
                load_json(common_prompt_path) if common_prompt_path.exists() else None
            ),
            "weights_file": repo_relative_path(weights_path),
            "weights": load_yaml(weights_path) if weights_path.exists() else None,
        },
        "materials": [
            {
                "path": repo_relative_path(p),
                "bytes": p.stat().st_size,
                "is_lfs_pointer": is_lfs_pointer(p),
            }
            for p in materials
        ],
        "statistics": load_yaml(statistics_path) if statistics_path.exists() else None,
        "expected_artifacts": ["slide_deck"],
    }


def writing_groups(root: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    rows = load_jsonl(root / "benchmark_query" / "benchmark_all.jsonl")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("domain1", "unknown"),
            row.get("domain2", "unknown"),
            row.get("lang", "unknown"),
        )
        groups[key].append(row)
    return groups


def present_groups(
    root: Path,
    *,
    include_known_bad: bool = False,
) -> dict[tuple[str, str], list[Path]]:
    case_dirs = []
    for case_dir in presentbench_case_dirs(root):
        if (
            not include_known_bad
            and str(case_dir.relative_to(root)) in KNOWN_BAD_PRESENTBENCH_CASES
        ):
            continue
        case_dirs.append(case_dir)
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for case_dir in case_dirs:
        rel_parts = case_dir.relative_to(root).parts
        category = rel_parts[0]
        collection = rel_parts[1] if len(rel_parts) > 2 else rel_parts[-1]
        groups[(category, collection)].append(case_dir)
    return groups


def summarize_group_selection(
    groups: dict[tuple[Any, ...], list[Any]],
    *,
    train_size: int,
    heldout_size: int,
    max_groups: int,
    selected_groups: int,
) -> dict[str, Any]:
    minimum_group_size = train_size + heldout_size
    group_sizes = sorted((len(items) for items in groups.values()), reverse=True)
    eligible_groups = sum(1 for size in group_sizes if size >= minimum_group_size)
    return {
        "total_groups": len(groups),
        "eligible_groups": eligible_groups,
        "selected_groups": selected_groups,
        "skipped_too_small": len(groups) - eligible_groups,
        "truncated_by_max_groups": max(0, eligible_groups - selected_groups),
        "minimum_group_size": minimum_group_size,
        "max_groups": max_groups,
        "largest_group_sizes": group_sizes[:10],
    }


def build_writing_splits(
    root: Path,
    train_size: int,
    heldout_size: int,
    max_groups: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    groups = writing_groups(root)

    splits: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(items) < train_size + heldout_size:
            continue
        selected = list(items)
        rng.shuffle(selected)
        train_rows = selected[:train_size]
        heldout_rows = selected[train_size : train_size + heldout_size]
        split_id = f"writingbench::{key[0]}::{key[1]}::{key[2]}".replace(" ", "_")
        splits.append(
            {
                "split_id": split_id,
                "source": "WritingBench",
                "learning_problem": "few_shot_skill_induction",
                "domain": {"primary": key[0], "secondary": key[1], "language": key[2]},
                "train_examples": [writing_task(row, root) for row in train_rows],
                "heldout_tasks": [writing_task(row, root) for row in heldout_rows],
                "prototype_goal": (
                    "infer a reusable writing skill from train examples and improve "
                    "heldout rubric scores"
                ),
            }
        )
        if len(splits) >= max_groups:
            break
    return splits


def build_present_splits(
    root: Path,
    train_size: int,
    heldout_size: int,
    max_groups: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    groups = present_groups(root)

    splits: list[dict[str, Any]] = []
    for (category, collection), items in sorted(
        groups.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        if len(items) < train_size + heldout_size:
            continue
        selected = list(items)
        rng.shuffle(selected)
        train_cases = selected[:train_size]
        heldout_cases = selected[train_size : train_size + heldout_size]
        splits.append(
            {
                "split_id": f"presentbench::{category}::{collection}",
                "source": "PresentBench",
                "learning_problem": "few_shot_skill_induction",
                "domain": {"primary": category, "secondary": collection},
                "train_examples": [present_task(path, root) for path in train_cases],
                "heldout_tasks": [present_task(path, root) for path in heldout_cases],
                "prototype_goal": (
                    "infer a reusable slide-generation skill from train examples and "
                    "improve heldout checklist pass rate"
                ),
            }
        )
        if len(splits) >= max_groups:
            break
    return splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writingbench-root", type=Path, default=DEFAULT_WRITINGBENCH)
    parser.add_argument("--presentbench-root", type=Path, default=DEFAULT_PRESENTBENCH)
    parser.add_argument("--train-size", type=int, default=3)
    parser.add_argument("--heldout-size", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=5)
    parser.add_argument(
        "--max-writing-groups",
        type=int,
        help="Override --max-groups for WritingBench only.",
    )
    parser.add_argument(
        "--max-present-groups",
        type=int,
        help="Override --max-groups for PresentBench only.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/splits/fewshot_splits.jsonl"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/splits/fewshot_split_summary.json"),
    )
    args = parser.parse_args()
    max_writing_groups = args.max_writing_groups or args.max_groups
    max_present_groups = args.max_present_groups or args.max_groups

    rng = random.Random(args.seed)
    writing_splits = (
        build_writing_splits(
            args.writingbench_root,
            args.train_size,
            args.heldout_size,
            max_writing_groups,
            rng,
        )
    )
    present_splits = (
        build_present_splits(
            args.presentbench_root,
            args.train_size,
            args.heldout_size,
            max_present_groups,
            rng,
        )
    )
    splits = [*writing_splits, *present_splits]

    write_jsonl(args.out, splits)
    summary = {
        "schema_version": "fewshot-split-build-summary/v1",
        "parameters": {
            "train_size": args.train_size,
            "heldout_size": args.heldout_size,
            "max_groups": args.max_groups,
            "max_writing_groups": max_writing_groups,
            "max_present_groups": max_present_groups,
            "seed": args.seed,
        },
        "sources": {
            "WritingBench": summarize_group_selection(
                writing_groups(args.writingbench_root),
                train_size=args.train_size,
                heldout_size=args.heldout_size,
                max_groups=max_writing_groups,
                selected_groups=len(writing_splits),
            ),
            "PresentBench": summarize_group_selection(
                present_groups(args.presentbench_root),
                train_size=args.train_size,
                heldout_size=args.heldout_size,
                max_groups=max_present_groups,
                selected_groups=len(present_splits),
            )
            | {
                "raw_total_groups_before_exclusions": len(
                    present_groups(args.presentbench_root, include_known_bad=True)
                ),
                "raw_case_count_before_exclusions": len(
                    list(presentbench_case_dirs(args.presentbench_root))
                ),
                "case_count_after_exclusions": sum(
                    len(items) for items in present_groups(args.presentbench_root).values()
                ),
                "excluded_known_bad_cases": sorted(KNOWN_BAD_PRESENTBENCH_CASES),
                "excluded_known_bad_case_count": len(KNOWN_BAD_PRESENTBENCH_CASES),
            },
        },
        "total_splits": len(splits),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(splits)} few-shot splits to {args.out}")
    print(f"Wrote split selection summary to {args.summary_out}")
    for split in splits[:5]:
        print(
            json.dumps(
                {
                    "split_id": split["split_id"],
                    "source": split["source"],
                    "train": len(split["train_examples"]),
                    "heldout": len(split["heldout_tasks"]),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
