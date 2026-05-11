#!/usr/bin/env python3
"""Run skill artifact quality / clarity judging over generated SKILL.md rows."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import PromptRunResult, parse_json_object  # noqa: E402
from scripts.metrics.demo_skill_quality_clarity import (  # noqa: E402
    build_skill_quality_prompt,
    compute_skill_quality_avg,
    public_examples_from_row,
)

SCHEMA_VERSION = "skill-quality-eval/v1"
EVALUATOR_KIND = "ctx2skill_style_skill_quality"
DEFAULT_PARSE_MAX_ATTEMPTS = 3


def skill_index(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            if row.get("status") != "success":
                continue
            pack_id = str(row.get("pack_id") or "")
            mode = str(row.get("mode") or "")
            skill_md = row.get("skill_md")
            if pack_id and mode and isinstance(skill_md, str) and skill_md.strip():
                rows[(pack_id, mode)] = row
    return rows


def pack_examples_text(pack: dict[str, Any]) -> str:
    examples = public_examples_from_row(pack)
    return json.dumps(examples, ensure_ascii=False, indent=2)


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("pack_id") or ""), str(row.get("mode") or "")


def call_judge(
    *,
    client: ChatCompletionClient,
    prompt: str,
    max_tokens: int,
) -> PromptRunResult:
    result = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of reusable SKILL.md artifacts. "
                    "Return only strict JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return PromptRunResult(
        text=result.text,
        model=result.model,
        usage=result.usage,
        finish_reason=result.finish_reason,
        request_id=result.request_id,
    )


def parse_quality_report(text: str) -> dict[str, Any]:
    report = parse_json_object(text)
    if "parse_error" in report:
        return report
    try:
        compute_skill_quality_avg(report)
    except ValueError as exc:
        report["parse_error"] = str(exc)
    return report


def model_error_row(
    *,
    pack: dict[str, Any],
    skill_row: dict[str, Any],
    mode: str,
    judge_model: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_kind": EVALUATOR_KIND,
        "pack_id": str(pack.get("pack_id") or ""),
        "mode": mode,
        "status": "model_error",
        "skill_model": skill_row.get("solver_model"),
        "judge_model": judge_model,
        "judge": None,
        "judge_parse_attempts": [],
        "skill_quality_report": None,
        "overall_skill_quality": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def evaluate_one(
    *,
    pack: dict[str, Any],
    skill_row: dict[str, Any],
    mode: str,
    config: ChatCompletionConfig,
    max_tokens: int,
    parse_max_attempts: int,
) -> dict[str, Any]:
    pack_id = str(pack.get("pack_id") or "")
    client = ChatCompletionClient(config)
    prompt = build_skill_quality_prompt(
        examples_text=pack_examples_text(pack),
        skill_md=str(skill_row["skill_md"]),
    )
    attempts = []
    last_judge: PromptRunResult | None = None
    last_report: dict[str, Any] = {"parse_error": "judge_not_called"}
    status = "judge_parse_error"
    for attempt in range(1, max(1, parse_max_attempts) + 1):
        judge = call_judge(client=client, prompt=prompt, max_tokens=max_tokens)
        report = parse_quality_report(judge.text)
        status = "success" if judge.finish_reason == "stop" and "parse_error" not in report else (
            "judge_incomplete" if judge.finish_reason != "stop" else "judge_parse_error"
        )
        attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "finish_reason": judge.finish_reason,
                "model": judge.model,
                "request_id": judge.request_id,
                "usage": judge.usage,
                "parse_error": report.get("parse_error"),
            }
        )
        last_judge = judge
        last_report = report
        if status == "success":
            break
    assert last_judge is not None
    overall_skill_quality = (
        compute_skill_quality_avg(last_report) if status == "success" else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_kind": EVALUATOR_KIND,
        "pack_id": pack_id,
        "mode": mode,
        "status": status,
        "skill_model": skill_row.get("solver_model"),
        "judge_model": last_judge.model or config.model,
        "judge": last_judge.to_json(),
        "judge_parse_attempts": attempts,
        "skill_quality_report": last_report,
        "overall_skill_quality": overall_skill_quality,
    }


def missing_skill_row(pack: dict[str, Any], mode: str, judge_model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_kind": EVALUATOR_KIND,
        "pack_id": str(pack.get("pack_id") or ""),
        "mode": mode,
        "status": "missing_skill",
        "skill_model": None,
        "judge_model": judge_model,
        "judge": None,
        "judge_parse_attempts": [],
        "skill_quality_report": None,
        "overall_skill_quality": None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        mode = str(row.get("mode") or "")
        entry = by_mode.setdefault(
            mode,
            {
                "rows": 0,
                "success": 0,
                "mean_skill_quality": None,
                "status_counts": {},
            },
        )
        entry["rows"] += 1
        entry["status_counts"][status] = entry["status_counts"].get(status, 0) + 1
        if status == "success":
            entry["success"] += 1
    for mode, entry in by_mode.items():
        values = [
            float(row["overall_skill_quality"])
            for row in rows
            if row.get("mode") == mode
            and row.get("status") == "success"
            and isinstance(row.get("overall_skill_quality"), int | float)
        ]
        entry["mean_skill_quality"] = sum(values) / len(values) if values else None
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_kind": EVALUATOR_KIND,
        "rows": len(rows),
        "status_counts": status_counts,
        "by_mode": by_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--skills", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--mode-alias", action="append", default=[])
    parser.add_argument("--modes", required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--judge-config-prefix", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--parse-max-attempts", type=int, default=DEFAULT_PARSE_MAX_ATTEMPTS)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    aliases = {}
    for item in args.mode_alias:
        if "=" not in item:
            print(f"error: invalid --mode-alias {item!r}; expected output=input", file=sys.stderr)
            return 2
        output_mode, input_mode = item.split("=", 1)
        aliases[output_mode.strip()] = input_mode.strip()

    packs = load_jsonl(args.packs)
    skills = skill_index(args.skills)
    expected = {(str(pack.get("pack_id") or ""), mode) for pack in packs for mode in modes}
    if args.dry_run:
        available = sum(
            1 for pack_id, mode in expected if (pack_id, aliases.get(mode, mode)) in skills
        )
        print(
            json.dumps(
                {"expected_rows": len(expected), "available_skill_rows": available},
                ensure_ascii=False,
            )
        )
        return 0

    try:
        config = (
            ChatCompletionConfig.from_env(args.env_file, prefix=args.judge_config_prefix)
            if args.judge_config_prefix
            else ChatCompletionConfig.from_env(args.env_file)
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.timeout_seconds is not None:
        config = replace(config, timeout_seconds=args.timeout_seconds)
    if args.max_retries is not None:
        config = replace(config, max_retries=args.max_retries)
    if args.enable_thinking is not None:
        config = replace(config, enable_thinking=args.enable_thinking)
    if args.num_threads is not None:
        config = replace(config, num_threads=args.num_threads)

    rows = []
    if args.resume and args.out.exists():
        rows = [
            row
            for row in load_jsonl(args.out)
            if row_key(row) in expected and row.get("status") == "success"
        ]
    completed = {row_key(row) for row in rows}
    jobs = []
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        for mode in modes:
            key = (pack_id, mode)
            if key in completed:
                continue
            source_mode = aliases.get(mode, mode)
            skill_row = skills.get((pack_id, source_mode))
            if skill_row is None:
                rows.append(missing_skill_row(pack, mode, config.model))
                write_jsonl(args.out, rows)
                continue
            jobs.append((pack, skill_row, mode))

    print(
        f"skill-quality modes={modes} rows={len(expected)} jobs={len(jobs)} "
        f"judge_model={config.model} num_threads={config.num_threads}",
        flush=True,
    )
    if config.num_threads == 1:
        for pack, skill_row, mode in jobs:
            try:
                row = evaluate_one(
                    pack=pack,
                    skill_row=skill_row,
                    mode=mode,
                    config=config,
                    max_tokens=args.max_tokens,
                    parse_max_attempts=args.parse_max_attempts,
                )
            except Exception as exc:  # noqa: BLE001 - checkpoint and continue batch judging.
                row = model_error_row(
                    pack=pack,
                    skill_row=skill_row,
                    mode=mode,
                    judge_model=config.model,
                    exc=exc,
                )
            rows.append(row)
            write_jsonl(args.out, rows)
    else:
        with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
            future_jobs = {
                executor.submit(
                    evaluate_one,
                    pack=pack,
                    skill_row=skill_row,
                    mode=mode,
                    config=config,
                    max_tokens=args.max_tokens,
                    parse_max_attempts=args.parse_max_attempts,
                ): (pack, skill_row, mode)
                for pack, skill_row, mode in jobs
            }
            for future in as_completed(future_jobs):
                try:
                    rows.append(future.result())
                except Exception as exc:  # noqa: BLE001 - checkpoint and continue batch judging.
                    pack, skill_row, mode = future_jobs[future]
                    rows.append(
                        model_error_row(
                            pack=pack,
                            skill_row=skill_row,
                            mode=mode,
                            judge_model=config.model,
                            exc=exc,
                        )
                    )
                write_jsonl(args.out, rows)

    summary = summarize(rows)
    summary.update({"modes": modes, "judge_model": config.model, "expected_rows": len(expected)})
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
