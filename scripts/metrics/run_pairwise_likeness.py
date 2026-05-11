#!/usr/bin/env python3
"""Run anchor pairwise example-likeness judging over existing eval outputs."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
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
from auto_skill.mvp import PromptRunResult, user_examples_from_pack  # noqa: E402
from auto_skill.pairwise_likeness import (  # noqa: E402
    EVALUATOR_KIND,
    SCHEMA_VERSION,
    build_pairwise_likeness_prompt,
    pairwise_status,
    pairwise_summary_markdown,
    parse_pairwise_likeness_report,
    summarize_pairwise_rows,
    winner_mode,
)
from scripts.eval.run_heldout_eval import select_packs  # noqa: E402

DEFAULT_MODES = "prompt_only,one_shot_skill_from_examples,ours_no_validation"


@dataclass(frozen=True)
class PairwiseJob:
    pack: dict[str, Any]
    task: dict[str, Any]
    anchor_row: dict[str, Any]
    candidate_row: dict[str, Any]
    anchor_mode: str
    candidate_mode: str
    order: str


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("mode") or ""),
    )


def pairwise_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("candidate_mode") or ""),
        str(row.get("order") or ""),
    )


def candidate_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        generation = row.get("generation")
        if not isinstance(generation, dict) or not isinstance(generation.get("text"), str):
            continue
        key = row_key(row)
        if all(key):
            index[key] = row
    return index


def call_judge(
    client: ChatCompletionClient,
    *,
    prompt: str,
    max_tokens: int,
) -> PromptRunResult:
    result = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict pairwise evaluator of example-likeness. "
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


def judge_with_parse_retry(
    *,
    client: ChatCompletionClient,
    prompt: str,
    max_tokens: int,
    parse_max_attempts: int,
) -> tuple[PromptRunResult, dict[str, Any], str, list[dict[str, Any]]]:
    last_judge: PromptRunResult | None = None
    last_report: dict[str, Any] = {"parse_error": "judge_not_called"}
    last_status = "judge_parse_error"
    attempts = []
    for attempt in range(1, max(1, parse_max_attempts) + 1):
        judge = call_judge(client, prompt=prompt, max_tokens=max_tokens)
        report = parse_pairwise_likeness_report(judge.text)
        status = pairwise_status(finish_reason=judge.finish_reason, report=report)
        attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "finish_reason": judge.finish_reason,
                "model": judge.model,
                "request_id": judge.request_id,
                "usage": judge.usage,
                "parse_error": report.get("parse_error"),
                "winner": report.get("winner"),
            }
        )
        last_judge = judge
        last_report = report
        last_status = status
        if status != "judge_parse_error":
            break
    assert last_judge is not None
    return last_judge, last_report, last_status, attempts


def evaluate_job(
    job: PairwiseJob,
    *,
    config: ChatCompletionConfig,
    max_tokens: int,
    parse_max_attempts: int,
) -> tuple[tuple[str, str, str, str], dict[str, Any]]:
    pack_id = str(job.pack["pack_id"])
    task_id = str(job.task["task_id"])
    if job.order == "anchor_first":
        output_a = str(job.anchor_row["generation"]["text"])
        output_b = str(job.candidate_row["generation"]["text"])
        mode_a = job.anchor_mode
        mode_b = job.candidate_mode
    else:
        output_a = str(job.candidate_row["generation"]["text"])
        output_b = str(job.anchor_row["generation"]["text"])
        mode_a = job.candidate_mode
        mode_b = job.anchor_mode
    key = (pack_id, task_id, job.candidate_mode, job.order)
    client = ChatCompletionClient(config)
    try:
        prompt = build_pairwise_likeness_prompt(
            examples=user_examples_from_pack(job.pack),
            heldout_task=job.task,
            output_a=output_a,
            output_b=output_b,
        )
        judge, report, status, attempts = judge_with_parse_retry(
            client=client,
            prompt=prompt,
            max_tokens=max_tokens,
            parse_max_attempts=parse_max_attempts,
        )
        verdict = winner_mode(report, mode_a=mode_a, mode_b=mode_b) if status == "success" else None
        row = {
            "schema_version": SCHEMA_VERSION,
            "pack_id": pack_id,
            "task_id": task_id,
            "source": job.pack.get("source"),
            "source_task_id": job.task.get("source_task_id"),
            "evaluator_kind": EVALUATOR_KIND,
            "anchor_mode": job.anchor_mode,
            "candidate_mode": job.candidate_mode,
            "order": job.order,
            "status": status,
            "judge": judge.to_json(),
            "judge_parse_attempts": attempts,
            "pairwise_report": report,
            "winner_mode": verdict,
            "anchor_solver_model": _generation_model(job.anchor_row),
            "candidate_solver_model": _generation_model(job.candidate_row),
            "judge_model": judge.model or config.model,
        }
        return key, row
    except Exception as exc:  # noqa: BLE001 - checkpoint and continue batch judging.
        return key, {
            "schema_version": SCHEMA_VERSION,
            "pack_id": pack_id,
            "task_id": task_id,
            "source": job.pack.get("source"),
            "source_task_id": job.task.get("source_task_id"),
            "evaluator_kind": EVALUATOR_KIND,
            "anchor_mode": job.anchor_mode,
            "candidate_mode": job.candidate_mode,
            "order": job.order,
            "status": "model_error",
            "judge": None,
            "judge_parse_attempts": [],
            "pairwise_report": None,
            "winner_mode": None,
            "anchor_solver_model": _generation_model(job.anchor_row),
            "candidate_solver_model": _generation_model(job.candidate_row),
            "judge_model": config.model,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _generation_model(row: dict[str, Any]) -> str | None:
    generation = row.get("generation")
    if isinstance(generation, dict) and isinstance(generation.get("model"), str):
        return generation["model"]
    model = row.get("solver_model")
    return model if isinstance(model, str) else None


def missing_candidate_row(
    *,
    pack: dict[str, Any],
    task: dict[str, Any],
    anchor_mode: str,
    candidate_mode: str,
    order: str,
    judge_model: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": str(pack["pack_id"]),
        "task_id": str(task["task_id"]),
        "source": pack.get("source"),
        "source_task_id": task.get("source_task_id"),
        "evaluator_kind": EVALUATOR_KIND,
        "anchor_mode": anchor_mode,
        "candidate_mode": candidate_mode,
        "order": order,
        "status": "missing_candidate_output",
        "judge": None,
        "judge_parse_attempts": [],
        "pairwise_report": None,
        "winner_mode": None,
        "anchor_solver_model": None,
        "candidate_solver_model": None,
        "judge_model": judge_model,
    }


def append_row(out: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows.append(row)
    write_jsonl(out, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--candidate-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--judge-config-prefix", default=None)
    parser.add_argument("--anchor-mode", default="few_shot_examples_only")
    parser.add_argument("--candidate-modes", default=DEFAULT_MODES)
    parser.add_argument("--source", choices=["WritingBench", "PresentBench"])
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument(
        "--orders",
        choices=["both", "anchor_first", "candidate_first"],
        default="both",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--parse-max-attempts", type=int, default=3)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2
    candidate_modes = [
        mode.strip() for mode in args.candidate_modes.split(",") if mode.strip()
    ]
    if args.anchor_mode in candidate_modes:
        candidate_modes = [mode for mode in candidate_modes if mode != args.anchor_mode]
    orders = (
        ["anchor_first", "candidate_first"]
        if args.orders == "both"
        else [args.orders]
    )

    packs = load_jsonl(args.packs)
    selected = select_packs(
        packs,
        pack_ids=set(args.pack_id) if args.pack_id else None,
        source=args.source,
        limit=args.limit_packs,
    )
    candidates = candidate_index(load_jsonl(args.candidate_eval))
    expected_keys = {
        (str(pack["pack_id"]), str(task["task_id"]), mode, order)
        for pack in selected
        for task in (pack.get("heldout_tasks") or [])[: args.limit_heldout]
        for mode in candidate_modes
        for order in orders
    }
    if args.dry_run:
        available = 0
        for pack_id, task_id, mode, _ in expected_keys:
            if (
                (pack_id, task_id, args.anchor_mode) in candidates
                and (pack_id, task_id, mode) in candidates
            ):
                available += 1
        print(
            json.dumps(
                {
                    "selected_pack_ids": [pack["pack_id"] for pack in selected],
                    "anchor_mode": args.anchor_mode,
                    "candidate_modes": candidate_modes,
                    "orders": orders,
                    "expected_rows": len(expected_keys),
                    "available_rows": available,
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        config = ChatCompletionConfig.from_env(
            args.env_file,
            prefix=args.judge_config_prefix,
        ) if args.judge_config_prefix else ChatCompletionConfig.from_env(args.env_file)
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
    if config.num_threads <= 0:
        print("error: --num-threads must be positive", file=sys.stderr)
        return 2

    rows = []
    if args.resume and args.out.exists():
        rows = [
            row
            for row in load_jsonl(args.out)
            if row.get("status") == "success"
            and pairwise_key(row) in expected_keys
            and row.get("judge_model") == config.model
        ]
    completed = {pairwise_key(row) for row in rows}
    if rows:
        print(f"Loaded {len(rows)} successful pairwise rows from {args.out}", flush=True)
    print(
        f"pairwise anchor={args.anchor_mode} candidates={candidate_modes} "
        f"judge_model={config.model} rows={len(expected_keys)} "
        f"num_threads={config.num_threads}",
        flush=True,
    )

    jobs: list[PairwiseJob] = []
    for pack in selected:
        pack_id = str(pack["pack_id"])
        tasks = pack.get("heldout_tasks") or []
        if args.limit_heldout is not None:
            tasks = tasks[: args.limit_heldout]
        for task in tasks:
            task_id = str(task["task_id"])
            anchor = candidates.get((pack_id, task_id, args.anchor_mode))
            for mode in candidate_modes:
                candidate = candidates.get((pack_id, task_id, mode))
                for order in orders:
                    key = (pack_id, task_id, mode, order)
                    if key in completed:
                        continue
                    if anchor is None or candidate is None:
                        append_row(
                            args.out,
                            rows,
                            missing_candidate_row(
                                pack=pack,
                                task=task,
                                anchor_mode=args.anchor_mode,
                                candidate_mode=mode,
                                order=order,
                                judge_model=config.model,
                            ),
                        )
                        continue
                    jobs.append(
                        PairwiseJob(
                            pack=pack,
                            task=task,
                            anchor_row=anchor,
                            candidate_row=candidate,
                            anchor_mode=args.anchor_mode,
                            candidate_mode=mode,
                            order=order,
                        )
                    )

    print(f"Running {len(jobs)} pairwise judge calls", flush=True)
    if config.num_threads == 1:
        for job in jobs:
            key, row = evaluate_job(
                job,
                config=config,
                max_tokens=args.max_tokens,
                parse_max_attempts=args.parse_max_attempts,
            )
            append_row(args.out, rows, row)
            print(
                f"  {key[1]} {key[2]} {key[3]}: "
                f"{row['status']} {row.get('winner_mode')}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
            futures = [
                executor.submit(
                    evaluate_job,
                    job,
                    config=config,
                    max_tokens=args.max_tokens,
                    parse_max_attempts=args.parse_max_attempts,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                key, row = future.result()
                append_row(args.out, rows, row)
                print(
                    f"  {key[1]} {key[2]} {key[3]}: "
                    f"{row['status']} {row.get('winner_mode')}",
                    flush=True,
                )

    summary = summarize_pairwise_rows(rows)
    summary.update(
        {
            "anchor_mode": args.anchor_mode,
            "candidate_modes": candidate_modes,
            "orders": orders,
            "judge_model": config.model,
            "expected_rows": len(expected_keys),
        }
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(pairwise_summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
