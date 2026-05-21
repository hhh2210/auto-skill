#!/usr/bin/env python3
"""Run source-derived target/reference/hard-negative author-style oracle metric."""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from auto_skill.cleaning.author_style.gpt_transport import codex_response_text  # noqa: E402
from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.metrics.author_style_reference import (  # noqa: E402
    SCHEMA_VERSION,
    ReferenceRetrievalJob,
    build_error_row,
    build_reference_retrieval_jobs,
    build_reference_retrieval_prompt,
    build_success_row,
    parse_reference_retrieval_report,
    reference_retrieval_status,
    reference_retrieval_summary_markdown,
    summarize_reference_retrieval_rows,
)


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("pack_id") or ""), str(row.get("task_id") or "")


def judge_openai_compatible(
    *,
    config: ChatCompletionConfig,
    prompt: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    client = ChatCompletionClient(config)
    result = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of anonymous author-style similarity. "
                    "Return only strict JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return result.text, {
        "provider": "openai_compatible",
        "model": result.model or config.model,
        "usage": result.usage,
        "finish_reason": result.finish_reason,
        "request_id": result.request_id,
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def judge_codex_oauth(
    *,
    args: argparse.Namespace,
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    text, usage = codex_response_text(
        prompt,
        model=args.model,
        auth_path=args.auth,
        instructions=(
            "You are a strict evaluator of anonymous author-style similarity. "
            "Return only strict JSON."
        ),
        service_tier=args.service_tier,
        timeout_seconds=args.timeout_seconds,
    )
    return text, {
        "provider": "codex_oauth",
        "model": args.model,
        "usage": usage,
        "finish_reason": None,
        "request_id": None,
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def judge_with_parse_retry(
    *,
    job: ReferenceRetrievalJob,
    args: argparse.Namespace,
    config: ChatCompletionConfig | None,
) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]:
    prompt = build_reference_retrieval_prompt(
        target_text=job.target_text,
        candidates=job.candidates,
        max_target_chars=args.max_target_chars,
        max_candidate_chars=args.max_candidate_chars,
    )
    candidate_ids = {candidate.candidate_id for candidate in job.candidates}
    last_judge: dict[str, Any] = {"model": args.model if args.backend == "codex-oauth" else None}
    last_report: dict[str, Any] = {"parse_error": "judge_not_called"}
    last_status = "judge_parse_error"
    attempts = []
    for attempt in range(1, args.parse_max_attempts + 1):
        if args.backend == "codex-oauth":
            text, judge = judge_codex_oauth(args=args, prompt=prompt)
        else:
            assert config is not None
            text, judge = judge_openai_compatible(
                config=config,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
        report = parse_reference_retrieval_report(text, candidate_ids=candidate_ids)
        status = reference_retrieval_status(
            finish_reason=judge.get("finish_reason"),
            report=report,
        )
        attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "model": judge.get("model"),
                "usage": judge.get("usage"),
                "finish_reason": judge.get("finish_reason"),
                "request_id": judge.get("request_id"),
                "parse_error": report.get("parse_error"),
                "selected_candidate_id": report.get("most_similar_candidate_id"),
            }
        )
        last_judge = judge
        last_report = report
        last_status = status
        if status != "judge_parse_error":
            break
    return last_judge, last_report, last_status, attempts


def evaluate_job(
    job: ReferenceRetrievalJob,
    *,
    args: argparse.Namespace,
    config: ChatCompletionConfig | None,
) -> tuple[tuple[str, str], dict[str, Any]]:
    key = (job.pack_id, job.task_id)
    try:
        judge, report, status, attempts = judge_with_parse_retry(
            job=job,
            args=args,
            config=config,
        )
        return key, build_success_row(
            job=job,
            judge=judge,
            report=report,
            status=status,
            attempts=attempts,
        )
    except Exception as exc:  # noqa: BLE001 - checkpoint batch errors per row.
        judge_model = args.model if args.backend == "codex-oauth" else (
            config.model if config is not None else None
        )
        return key, build_error_row(
            job=job,
            status="model_error",
            error=f"{type(exc).__name__}: {exc}",
            judge_model=judge_model,
        )


def append_row(out: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows.append(row)
    write_jsonl(out, rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--private-eval", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument("--references-per-target", type=int, default=1)
    parser.add_argument("--negatives-per-target", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=("openai-compatible", "codex-oauth"),
        default="openai-compatible",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--judge-config-prefix", default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--auth", type=Path, default=Path("~/.codex/auth.json").expanduser())
    parser.add_argument("--service-tier", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--parse-max-attempts", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-target-chars", type=int, default=5000)
    parser.add_argument("--max-candidate-chars", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.references_per_target <= 0:
        print("error: --references-per-target must be positive", file=sys.stderr)
        return 2
    if args.negatives_per_target <= 0:
        print("error: --negatives-per-target must be positive", file=sys.stderr)
        return 2
    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2
    if args.num_threads <= 0:
        print("error: --num-threads must be positive", file=sys.stderr)
        return 2

    packs = load_jsonl(args.packs)
    if args.limit_packs is not None:
        packs = packs[: args.limit_packs]
    if args.limit_heldout is not None:
        packs = [
            {**pack, "heldout_tasks": (pack.get("heldout_tasks") or [])[: args.limit_heldout]}
            for pack in packs
        ]
    private_eval = load_jsonl(args.private_eval)
    hard_negatives = load_jsonl(args.hard_negatives)
    jobs, skipped = build_reference_retrieval_jobs(
        packs=packs,
        private_eval=private_eval,
        hard_negatives=hard_negatives,
        references_per_target=args.references_per_target,
        negatives_per_target=args.negatives_per_target,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "expected_rows": len(jobs),
                    "skipped": skipped,
                    "pack_ids": [pack.get("pack_id") for pack in packs],
                    "backend": args.backend,
                    "model": args.model if args.backend == "codex-oauth" else None,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    config = None
    if args.backend == "openai-compatible":
        try:
            config = (
                ChatCompletionConfig.from_env(args.env_file, prefix=args.judge_config_prefix)
                if args.judge_config_prefix
                else ChatCompletionConfig.from_env(args.env_file)
            )
        except ConfigError as exc:
            print(f"configuration error: {exc}", file=sys.stderr)
            return 2
        config = replace(config, timeout_seconds=args.timeout_seconds)
        if args.max_retries is not None:
            config = replace(config, max_retries=args.max_retries)

    rows = []
    if args.resume and args.out.exists():
        rows = [
            row
            for row in load_jsonl(args.out)
            if row.get("status") == "success" and row_key(row) in {job_key(job) for job in jobs}
        ]
    completed = {row_key(row) for row in rows}
    pending = [job for job in jobs if job_key(job) not in completed]
    model_label = (
        args.model if args.backend == "codex-oauth" else (config.model if config else None)
    )
    print(
        f"reference retrieval backend={args.backend} model={model_label} "
        f"rows={len(jobs)} pending={len(pending)} num_threads={args.num_threads}",
        flush=True,
    )

    if args.num_threads == 1:
        for job in pending:
            key, row = evaluate_job(job, args=args, config=config)
            append_row(args.out, rows, row)
            print(f"  {key[1]}: {row['status']} {row.get('oracle_correct')}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(evaluate_job, job, args=args, config=config)
                for job in pending
            ]
            for future in as_completed(futures):
                key, row = future.result()
                append_row(args.out, rows, row)
                print(f"  {key[1]}: {row['status']} {row.get('oracle_correct')}", flush=True)

    summary = summarize_reference_retrieval_rows(rows, skipped=skipped)
    summary.update(
        {
            "backend": args.backend,
            "judge_model": model_label,
            "expected_rows": len(jobs),
            "references_per_target": args.references_per_target,
            "negatives_per_target": args.negatives_per_target,
        }
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(reference_retrieval_summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def job_key(job: ReferenceRetrievalJob) -> tuple[str, str]:
    return job.pack_id, job.task_id


if __name__ == "__main__":
    raise SystemExit(main())
