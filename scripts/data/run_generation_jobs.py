#!/usr/bin/env python3
"""Run desired-output generation jobs with a Bailian/OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, load_jsonl_lenient_final_line  # noqa: E402
from auto_skill.generated_outputs import private_leak_matches  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402

SYSTEM_PROMPT = """You generate high-quality final outputs for user-visible examples.
Follow the task and material excerpts exactly.
Return only the final desired output that should be visible to a future user.
Do not include meta-commentary about dataset construction."""

_THREAD_LOCAL = threading.local()


def read_jsonl_or_exit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"error: JSONL file does not exist: {path}", file=sys.stderr)
        raise SystemExit(2)
    try:
        return load_jsonl(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: failed to read JSONL file {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def read_existing_successes(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    completed: dict[str, set[str]] = {}
    try:
        rows, warnings = load_jsonl_lenient_final_line(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: failed to read existing output file {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for row in rows:
        if not isinstance(row, dict):
            print(
                f"error: malformed JSONL row in {path}: expected object, "
                f"got {type(row).__name__}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if (
            row.get("status") == "success"
            and row.get("job_id")
            and row.get("finish_reason") == "stop"
        ):
            completed.setdefault(str(row["job_id"]), set()).add(str(row.get("prompt_sha256")))
    return completed


def read_existing_latest_statuses(path: Path) -> dict[tuple[str, str], str]:
    """Return the latest status for each ``(job_id, prompt_sha256)`` pair."""

    if not path.exists():
        return {}
    latest: dict[tuple[str, str], str] = {}
    try:
        rows, warnings = load_jsonl_lenient_final_line(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: failed to read existing output file {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for row in rows:
        if not isinstance(row, dict):
            print(
                f"error: malformed JSONL row in {path}: expected object, "
                f"got {type(row).__name__}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        job_id = row.get("job_id")
        prompt_sha = row.get("prompt_sha256")
        status = row.get("status")
        if job_id and prompt_sha and status:
            latest[(str(job_id), str(prompt_sha))] = str(status)
    return latest


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value!r}")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def validate_job(job: dict[str, Any], index: int) -> list[str]:
    errors = []
    for field in ("job_id", "pack_id", "example_id", "prompt", "prompt_sha256"):
        if not isinstance(job.get(field), str) or not job.get(field):
            errors.append(f"job[{index}] missing string field {field!r}")
    return errors


def validate_jobs_or_exit(jobs: list[dict[str, Any]]) -> None:
    errors = []
    for index, job in enumerate(jobs):
        errors.extend(validate_job(job, index))
    if errors:
        print("error: invalid generation jobs:", file=sys.stderr)
        for error in errors[:20]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more errors", file=sys.stderr)
        raise SystemExit(2)


def select_jobs(
    jobs: list[dict[str, Any]],
    *,
    pack_id: str | None,
    source: str | None,
    retry_keys: set[tuple[str, str]] | None,
    limit: int | None,
    completed: dict[str, set[str]],
    resume: bool,
) -> list[dict[str, Any]]:
    selected = []
    for job in jobs:
        if pack_id is not None and job.get("pack_id") != pack_id:
            continue
        if source is not None and job.get("source") != source:
            continue
        job_id = str(job.get("job_id"))
        prompt_sha = str(job.get("prompt_sha256"))
        if retry_keys is not None and (job_id, prompt_sha) not in retry_keys:
            continue
        if resume and prompt_sha in completed.get(job_id, set()):
            continue
        selected.append(job)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def count_statuses(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def has_failed_generation_rows(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("status") != "success" for row in rows)


def build_messages(job: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": job["prompt"]},
    ]


def safe_error(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:500],
        "status_code": getattr(exc, "status_code", None),
        "code": getattr(exc, "code", None),
        "request_id": getattr(exc, "request_id", None),
    }


def thread_client(config: ChatCompletionConfig) -> ChatCompletionClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = ChatCompletionClient(config)
        _THREAD_LOCAL.client = client
    return client


def build_output_row(
    job: dict[str, Any],
    *,
    config: ChatCompletionConfig,
    temperature: float | None,
    max_tokens: int,
    enable_thinking: bool | None,
    thinking_budget: int | None,
    print_lock: threading.Lock,
    index: int,
    total: int,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    prompt_sha = str(job["prompt_sha256"])
    started = time.monotonic()
    with print_lock:
        print(
            f"[{index}/{total}] starting {job_id} "
            f"(prompt_chars={len(job['prompt'])}, prompt_sha={prompt_sha[:12]})",
            flush=True,
        )
    try:
        result = thread_client(config).complete(
            build_messages(job),
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        leak_matches = private_leak_matches(result.text)
        if leak_matches:
            row = {
                "schema_version": "generated-desired-output/v1",
                "status": "rejected_private_leak",
                "job_id": job_id,
                "pack_id": job.get("pack_id"),
                "example_id": job.get("example_id"),
                "source": job.get("source"),
                "source_task_id": job.get("source_task_id"),
                "prompt_sha256": prompt_sha,
                "prompt_template_version": job.get("prompt_template_version"),
                "builder_version": job.get("builder_version"),
                "model": result.model,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "leak_matches": leak_matches,
            }
            row["duration_seconds"] = round(time.monotonic() - started, 3)
            return row
        if result.finish_reason != "stop":
            row = {
                "schema_version": "generated-desired-output/v1",
                "status": "rejected_incomplete_generation",
                "job_id": job_id,
                "pack_id": job.get("pack_id"),
                "example_id": job.get("example_id"),
                "source": job.get("source"),
                "source_task_id": job.get("source_task_id"),
                "prompt_sha256": prompt_sha,
                "prompt_template_version": job.get("prompt_template_version"),
                "builder_version": job.get("builder_version"),
                "model": result.model,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "generation_params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "enable_thinking": enable_thinking,
                    "thinking_budget": thinking_budget,
                    "configured_model": config.model,
                    "base_url": config.base_url,
                },
            }
            row["duration_seconds"] = round(time.monotonic() - started, 3)
            return row
        row = {
            "schema_version": "generated-desired-output/v1",
            "status": "success",
            "job_id": job_id,
            "pack_id": job.get("pack_id"),
            "example_id": job.get("example_id"),
            "source": job.get("source"),
            "source_task_id": job.get("source_task_id"),
            "prompt_sha256": prompt_sha,
            "prompt_template_version": job.get("prompt_template_version"),
            "builder_version": job.get("builder_version"),
            "generation_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "thinking_budget": thinking_budget,
                "configured_model": config.model,
                "base_url": config.base_url,
            },
            "model": result.model,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
            "desired_output": result.text,
        }
        row["duration_seconds"] = round(time.monotonic() - started, 3)
        return row
    except Exception as exc:  # noqa: BLE001 - keep batch jobs resumable.
        row = {
            "schema_version": "generated-desired-output/v1",
            "status": "error",
            "job_id": job_id,
            "pack_id": job.get("pack_id"),
            "example_id": job.get("example_id"),
            "source": job.get("source"),
            "source_task_id": job.get("source_task_id"),
            "prompt_sha256": prompt_sha,
            "prompt_template_version": job.get("prompt_template_version"),
            "builder_version": job.get("builder_version"),
            "error": safe_error(exc),
        }
        row["duration_seconds"] = round(time.monotonic() - started, 3)
        return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("artifacts/jobs/example_generation_jobs.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/jobs/generated_desired_outputs.jsonl"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--config-prefix",
        help=(
            "Optional environment prefix for model config, e.g. MIMO or QWEN. "
            "When set, {PREFIX}_MODEL is required; {PREFIX}_BASE_URL/API_KEY "
            "fall back to BAILIAN/OpenAI if unset."
        ),
    )
    parser.add_argument("--pack-id")
    parser.add_argument(
        "--source",
        choices=("WritingBench", "PresentBench"),
        help="Only run generation jobs from one benchmark source.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature. Defaults to BAILIAN_TEMPERATURE if set; "
            "otherwise omitted so the provider/model default applies."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking-budget", type=positive_int)
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=None,
        help="Override request timeout from BAILIAN_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--max-retries",
        type=non_negative_int,
        default=None,
        help="Override SDK retry count from BAILIAN_MAX_RETRIES.",
    )
    parser.add_argument(
        "--num-threads",
        type=positive_int,
        default=None,
        help=(
            "Number of concurrent API requests. Defaults to BAILIAN_NUM_THREADS "
            "if set, otherwise 1."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-existing-failures-only",
        action="store_true",
        help=(
            "Only select jobs whose latest row in --out is non-success. This is "
            "useful for retrying API errors or length rejections without expanding "
            "the sample window."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even when selected generation jobs fail. Default is fail-closed.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 when filters select no generation jobs. Default is fail-closed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = read_jsonl_or_exit(args.jobs)
    validate_jobs_or_exit(jobs)
    completed = read_existing_successes(args.out)
    retry_keys = None
    if args.retry_existing_failures_only:
        latest_statuses = read_existing_latest_statuses(args.out)
        retry_keys = {
            key for key, status in latest_statuses.items() if status != "success"
        }
    selected = select_jobs(
        jobs,
        pack_id=args.pack_id,
        source=args.source,
        retry_keys=retry_keys,
        limit=args.limit,
        completed=completed,
        resume=args.resume,
    )

    if args.dry_run:
        print(f"Selected {len(selected)} jobs from {args.jobs}")
        if selected:
            first = selected[0]
            print(json.dumps({"job_id": first["job_id"], "prompt": first["prompt"][:1200]}))
        return 0
    if not selected and not args.allow_empty:
        print("error: no generation jobs selected", file=sys.stderr)
        return 3
    if not selected and args.allow_empty:
        print("Running 0 jobs with model=(not loaded) base_url=(not loaded)")
        print(json.dumps({"status_counts": {}}, ensure_ascii=False))
        return 0

    try:
        config = ChatCompletionConfig.from_env(args.env_file, prefix=args.config_prefix)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.timeout_seconds is not None:
        config = replace(config, timeout_seconds=args.timeout_seconds)
    if args.max_retries is not None:
        config = replace(config, max_retries=args.max_retries)
    num_threads = args.num_threads if args.num_threads is not None else config.num_threads
    temperature = args.temperature
    if temperature is None:
        temperature = config.temperature
    enable_thinking = args.enable_thinking
    if enable_thinking is None:
        enable_thinking = config.enable_thinking
    thinking_budget = args.thinking_budget
    if thinking_budget is None:
        thinking_budget = config.thinking_budget

    print(
        f"Running {len(selected)} jobs with model={config.model} "
        f"base_url={config.base_url} num_threads={num_threads} "
        f"timeout_seconds={config.timeout_seconds} max_retries={config.max_retries} "
        f"enable_thinking={enable_thinking} thinking_budget={thinking_budget}"
    )
    started_all = time.monotonic()
    print_lock = threading.Lock()
    output_rows = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_job = {
            executor.submit(
                build_output_row,
                job,
                config=config,
                temperature=temperature,
                max_tokens=args.max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                print_lock=print_lock,
                index=index,
                total=len(selected),
            ): job
            for index, job in enumerate(selected, start=1)
        }
        completed_count = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            row = future.result()
            output_rows.append(row)
            append_jsonl(args.out, row)
            completed_count += 1
            duration = row.get("duration_seconds")
            suffix = f" in {duration}s" if duration is not None else ""
            print(
                f"[done {completed_count}/{len(selected)}] "
                f"{job['job_id']}: {row['status']}{suffix}"
            )
    elapsed = time.monotonic() - started_all
    per_minute = (len(selected) / elapsed * 60) if elapsed > 0 else 0.0
    print(
        f"Finished {len(selected)} selected jobs in {elapsed:.1f}s "
        f"({per_minute:.2f} jobs/min)"
    )
    status_counts = count_statuses(output_rows)
    print(json.dumps({"status_counts": status_counts}, ensure_ascii=False))
    if has_failed_generation_rows(output_rows) and not args.allow_partial:
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
