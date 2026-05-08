#!/usr/bin/env python3
"""Inspect WritingBench and PresentBench and export few-shot skill seed examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WRITINGBENCH = Path(os.getenv("WRITINGBENCH_ROOT", REPO_ROOT / "data" / "WritingBench"))
DEFAULT_PRESENTBENCH = Path(
    os.getenv("PRESENTBENCH_ROOT", REPO_ROOT / "data" / "PresentBench_repo")
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Any:
    if yaml is None:
        return {"raw": read_text(path)}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 512:
        return False
    head = path.read_text(encoding="utf-8", errors="ignore")
    return head.startswith("version https://git-lfs.github.com/spec/")


def checklist_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def normalize_writingbench(root: Path, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark_file = root / "benchmark_query" / "benchmark_all.jsonl"
    rows = load_jsonl(benchmark_file)
    by_domain: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for row in rows:
        domain = f"{row.get('domain1', 'unknown')} / {row.get('domain2', 'unknown')}"
        by_domain[domain] = by_domain.get(domain, 0) + 1
        lang = row.get("lang", "unknown")
        by_lang[lang] = by_lang.get(lang, 0) + 1
        if len(examples) < limit:
            examples.append(
                {
                    "source": "WritingBench",
                    "source_id": row["index"],
                    "domain": {
                        "primary": row.get("domain1"),
                        "secondary": row.get("domain2"),
                        "language": lang,
                    },
                    "task_input": row["query"],
                    "judge": {
                        "type": "llm_score_1_to_10_per_criterion",
                        "prompt_file": str(root / "prompt.py"),
                    },
                    "supervision": {
                        "type": "instance_specific_rubric",
                        "items": row.get("checklist", []),
                    },
                    "learning_problem": {
                        "mode": "few_shot_skill_induction",
                        "source_examples": (
                            "sample several user-visible solved examples from the same "
                            "domain or requirement type"
                        ),
                        "target": (
                            "extract a reusable skill module that improves heldout "
                            "writing tasks"
                        ),
                    },
                    "materials": [],
                    "derived_fields": {
                        "baseline_output": None,
                        "desired_output": None,
                        "benchmark_trace_eval_only": None,
                        "extracted_skill": None,
                        "heldout_output_with_skill": None,
                        "gap_diagnosis_optional": None,
                    },
                }
            )

    summary = {
        "source": "WritingBench",
        "root": str(root),
        "benchmark_file": str(benchmark_file),
        "num_examples": len(rows),
        "num_domains": len(by_domain),
        "languages": by_lang,
        "top_domains": sorted(by_domain.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "schema": ["index", "domain1", "domain2", "lang", "query", "checklist"],
    }
    return summary, examples


def require_path(path: Path, *, description: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"{description} does not exist: {path}. "
        "Prepare local benchmark data or pass the corresponding --*-root argument."
    )


def presentbench_case_dirs(root: Path) -> list[Path]:
    return sorted({path.parents[1] for path in root.glob("**/generation_task/instructions.md")})


def normalize_presentbench(root: Path, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_dirs = presentbench_case_dirs(root)
    by_category: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    lfs_pointer_count = 0
    material_count = 0

    for case_dir in case_dirs:
        rel_case = case_dir.relative_to(root)
        category = rel_case.parts[0]
        by_category[category] = by_category.get(category, 0) + 1

        generation_dir = case_dir / "generation_task"
        judge_prompt_path = generation_dir / "judge_prompt.json"
        statistics_path = generation_dir / "statistics.yaml"
        materials = sorted(
            p
            for p in case_dir.iterdir()
            if p.is_file() and p.name.startswith("material")
        )
        material_count += len(materials)
        lfs_pointer_count += sum(1 for p in materials if is_lfs_pointer(p))

        if len(examples) < limit:
            judge_prompt = load_json(judge_prompt_path)
            checklist_keys = {
                key: checklist_count(value)
                for key, value in judge_prompt.items()
                if key.startswith("material_independent_checklist_")
                or key.startswith("material_dependent_checklist_")
            }
            examples.append(
                {
                    "source": "PresentBench",
                    "source_id": str(rel_case),
                    "domain": {
                        "primary": category,
                        "secondary": "/".join(rel_case.parts[1:-1]) or rel_case.name,
                    },
                    "task_input": read_text(generation_dir / "instructions.md"),
                    "judge": {
                        "type": "binary_yes_no_checklist_with_boxed_answer",
                        "judge_prompt_file": str(judge_prompt_path),
                        "domain_common_prompt_file": str(
                            root / category / "common_judge_prompt.json"
                        ),
                        "weights_file": str(root / category / "judge_weights.yaml"),
                    },
                    "supervision": {
                        "type": "material_independent_and_dependent_checklists",
                        "checklist_counts": checklist_keys,
                        "judge_prompt": judge_prompt,
                    },
                    "learning_problem": {
                        "mode": "few_shot_skill_induction",
                        "source_examples": (
                            "sample several slide-generation cases with similar "
                            "domain/checklist patterns"
                        ),
                        "target": (
                            "extract a reusable slide-generation skill module that "
                            "improves heldout decks"
                        ),
                    },
                    "materials": [
                        {
                            "path": str(p),
                            "bytes": p.stat().st_size,
                            "is_lfs_pointer": is_lfs_pointer(p),
                        }
                        for p in materials
                    ],
                    "statistics": load_yaml(statistics_path) if statistics_path.exists() else None,
                    "derived_fields": {
                        "baseline_output": None,
                        "desired_output": None,
                        "benchmark_trace_eval_only": None,
                        "extracted_skill": None,
                        "heldout_output_with_skill": None,
                        "gap_diagnosis_optional": None,
                    },
                }
            )

    summary = {
        "source": "PresentBench",
        "root": str(root),
        "num_examples": len(case_dirs),
        "categories": by_category,
        "material_files": material_count,
        "material_lfs_pointers": lfs_pointer_count,
        "schema": [
            "case_path",
            "generation_task/instructions.md",
            "generation_task/judge_prompt.json",
            "generation_task/statistics.yaml",
            "material.*",
        ],
    }
    return summary, examples


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writingbench-root", type=Path, default=DEFAULT_WRITINGBENCH)
    parser.add_argument("--presentbench-root", type=Path, default=DEFAULT_PRESENTBENCH)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    try:
        require_path(args.writingbench_root, description="WritingBench root")
        require_path(args.presentbench_root, description="PresentBench root")
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2

    writing_summary, writing_examples = normalize_writingbench(args.writingbench_root, args.limit)
    present_summary, present_examples = normalize_presentbench(args.presentbench_root, args.limit)

    report = {
        "summaries": [writing_summary, present_summary],
        "next_code_units": [
            "seed_ingest: full export to unified JSONL",
            "example_sampler: group same-domain examples into few-shot train/heldout splits",
            (
                "example_builder: create user-visible solved examples when benchmark "
                "lacks gold examples"
            ),
            "skill_extraction: synthesize SKILL.md/templates/tests from user examples only",
            (
                "downstream_eval: rerun heldout tasks with generated skill modules "
                "and compare rubric scores"
            ),
            "gap_diagnosis: optional analysis only, not the primary benchmark target",
        ],
    }
    examples = writing_examples + present_examples

    write_json(args.out_dir / "benchmark_inspection.json", report)
    write_jsonl(args.out_dir / "seed_examples.jsonl", examples)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {len(examples)} example seeds to {args.out_dir / 'seed_examples.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
