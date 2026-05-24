#!/usr/bin/env python3
"""Run the author-style stylometric cosine baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.io.jsonl import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.probes.author_style.stylometric import (  # noqa: E402
    function_words_for_jobs,
    jobs_from_probe_dir,
    pack_metadata,
    run_baseline,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--gpt55-rows", type=Path)
    parser.add_argument("--qwen-rows", type=Path)
    parser.add_argument("--mimo-rows", type=Path)
    parser.add_argument("--results-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packs = load_jsonl(args.probe_dir / "oracle" / "packs.jsonl")
    jobs = jobs_from_probe_dir(args.probe_dir)
    function_words = function_words_for_jobs(jobs)
    rows = run_baseline(
        jobs,
        pack_metadata=pack_metadata(packs),
        function_words=function_words,
    )
    write_jsonl(args.out, rows)
    summary = summarize(
        rows,
        function_words=function_words,
        judge_rows={
            name: load_jsonl(path)
            for name, path in {
                "gpt5_5": args.gpt55_rows,
                "qwen": args.qwen_rows,
                "mimo": args.mimo_rows,
            }.items()
            if path
        },
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.results_md:
        append_results(args.results_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def append_results(path: Path, summary: dict[str, object]) -> None:
    overall = summary["overall"]
    assert isinstance(overall, dict)
    lines = [
        "",
        "## Stylometric Baseline",
        "",
        f"- Top-1 accuracy: {pct(overall['accuracy'])} "
        f"({overall['correct']}/{overall['rows']}), "
        f"Wilson 95% CI [{pct(overall['ci95_low'])}, {pct(overall['ci95_high'])}].",
        "- Random reference: requested 25%; actual uniform 5-way chance is 20%.",
    ]
    by_source = summary["by_source"]
    assert isinstance(by_source, dict)
    for source, row in sorted(by_source.items()):
        assert isinstance(row, dict)
        lines.append(f"- {source}: {pct(row['accuracy'])} ({row['correct']}/{row['rows']}).")
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section = "\n".join(lines)
    marker = "## Stylometric Baseline"
    if marker in existing:
        prefix, _, _ = existing.partition(marker)
        path.write_text(prefix.rstrip() + section + "\n", encoding="utf-8")
    else:
        path.write_text(existing.rstrip() + "\n" + section + "\n", encoding="utf-8")


def pct(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
