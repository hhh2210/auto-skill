#!/usr/bin/env python3
"""Evaluate whether heldout outputs are grounded in current task materials."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from auto_skill.grounding import (  # noqa: E402
    EVALUATOR_KIND,
    SCHEMA_VERSION,
    build_grounding_prompt,
    grounding_status,
    parse_grounding_report,
    summarize_grounding_rows,
)
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import PromptRunResult  # noqa: E402


@dataclass(frozen=True)
class GroundingJob:
    pack: dict[str, Any]
    task: dict[str, Any]
    candidate_row: dict[str, Any]


@dataclass(frozen=True)
class SkippedCandidate:
    cell: tuple[str, str, str]
    reason: str


def call_model(
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
                    "You are a strict factual-grounding evaluator. "
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


def generation_text(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if not isinstance(generation, dict):
        return ""
    text = generation.get("text")
    return text if isinstance(text, str) else ""


def pack_index(packs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(pack.get("pack_id") or ""): pack
        for pack in packs
        if isinstance(pack.get("pack_id"), str)
    }


def task_index(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(task.get("task_id") or ""): task
        for task in pack.get("heldout_tasks") or []
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }


def candidate_model(row: dict[str, Any]) -> str | None:
    model = row.get("solver_model")
    if isinstance(model, str) and model:
        return model
    generation = row.get("generation")
    if isinstance(generation, dict):
        model = generation.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def row_cell(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("mode") or ""),
    )


def load_resume_success_rows(
    out: Path, expected_cells: set[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    if not out.exists():
        return []
    rows = []
    for row in load_jsonl(out):
        if row.get("status") == "success" and row_cell(row) in expected_cells:
            rows.append(row)
    return rows


def judge_with_parse_retry(
    *,
    client: ChatCompletionClient,
    prompt: str,
    max_tokens: int,
    parse_max_attempts: int,
) -> tuple[PromptRunResult, dict[str, Any], str, int]:
    if parse_max_attempts <= 0:
        raise ValueError("parse_max_attempts must be positive")

    last_judge: PromptRunResult | None = None
    last_report: dict[str, Any] = {"parse_error": "judge_not_called"}
    last_status = "judge_parse_error"
    for attempt in range(1, parse_max_attempts + 1):
        judge = call_model(client, prompt=prompt, max_tokens=max_tokens)
        report = parse_grounding_report(judge.text)
        status = grounding_status(report, judge.finish_reason)
        last_judge = judge
        last_report = report
        last_status = status
        if status != "judge_parse_error":
            return judge, report, status, attempt
        if attempt < parse_max_attempts:
            print(
                f"  grounding parse error attempt {attempt}/{parse_max_attempts}: "
                f"{report.get('parse_error')}; retrying judge",
                file=sys.stderr,
            )
    assert last_judge is not None
    return last_judge, last_report, last_status, parse_max_attempts


def evaluate_job(
    job: GroundingJob,
    *,
    config: ChatCompletionConfig,
    max_tokens: int,
    parse_max_attempts: int,
    max_evidence_chars: int,
    max_candidate_chars: int,
) -> tuple[tuple[str, str, str], dict[str, Any]]:
    candidate_row = job.candidate_row
    pack_id, task_id, mode = row_cell(candidate_row)
    cell = (pack_id, task_id, mode)
    candidate_output = generation_text(candidate_row)
    client = ChatCompletionClient(config)
    try:
        prompt = build_grounding_prompt(
            heldout_task=job.task,
            candidate_output=candidate_output,
            mode=mode,
            max_evidence_chars=max_evidence_chars,
            max_candidate_chars=max_candidate_chars,
        )
        judge, report, status, attempts = judge_with_parse_retry(
            client=client,
            prompt=prompt,
            max_tokens=max_tokens,
            parse_max_attempts=parse_max_attempts,
        )
        return cell, {
            "schema_version": SCHEMA_VERSION,
            "pack_id": pack_id,
            "task_id": task_id,
            "source": candidate_row.get("source") or job.pack.get("source"),
            "source_task_id": candidate_row.get("source_task_id") or job.task.get("source_task_id"),
            "mode": mode,
            "evaluator_kind": EVALUATOR_KIND,
            "status": status,
            "candidate_eval_status": candidate_row.get("status"),
            "candidate_output_chars": len(candidate_output),
            "grounding_report": report,
            "judge": judge.to_json(),
            "judge_parse_attempts": attempts,
            "solver_model": candidate_model(candidate_row),
            "judge_model": judge.model or config.model,
        }
    except Exception as exc:  # noqa: BLE001 - keep batch runs fail-soft per cell.
        return cell, {
            "schema_version": SCHEMA_VERSION,
            "pack_id": pack_id,
            "task_id": task_id,
            "source": candidate_row.get("source") or job.pack.get("source"),
            "source_task_id": candidate_row.get("source_task_id") or job.task.get("source_task_id"),
            "mode": mode,
            "evaluator_kind": EVALUATOR_KIND,
            "status": "model_error",
            "candidate_eval_status": candidate_row.get("status"),
            "candidate_output_chars": len(candidate_output),
            "grounding_report": None,
            "error": str(exc),
            "judge": None,
            "judge_parse_attempts": 0,
            "solver_model": candidate_model(candidate_row),
            "judge_model": config.model,
        }


def build_jobs(
    *,
    packs: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    modes: set[str] | None,
    pack_ids: set[str] | None,
) -> tuple[list[GroundingJob], list[SkippedCandidate]]:
    packs_by_id = pack_index(packs)
    jobs: list[GroundingJob] = []
    skipped: list[SkippedCandidate] = []
    for row in candidate_rows:
        if row.get("status") != "success":
            continue
        pack_id, task_id, mode = row_cell(row)
        if modes is not None and mode not in modes:
            continue
        if pack_ids is not None and pack_id not in pack_ids:
            continue
        cell = (pack_id, task_id, mode)
        if not generation_text(row).strip():
            skipped.append(SkippedCandidate(cell=cell, reason="empty_generation"))
            continue
        pack = packs_by_id.get(pack_id)
        if pack is None:
            skipped.append(SkippedCandidate(cell=cell, reason="missing_pack"))
            continue
        task = task_index(pack).get(task_id)
        if task is None:
            skipped.append(SkippedCandidate(cell=cell, reason="missing_heldout_task"))
            continue
        jobs.append(GroundingJob(pack=pack, task=task, candidate_row=row))
    return jobs, skipped


def parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--judge-config-prefix")
    parser.add_argument("--modes")
    parser.add_argument("--pack-id", action="append", dest="pack_ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--parse-max-attempts", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=60000)
    parser.add_argument("--max-candidate-chars", type=int, default=60000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    if args.num_threads <= 0:
        print("error: --num-threads must be positive", file=sys.stderr)
        return 2
    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2

    try:
        config = ChatCompletionConfig.from_env(
            args.env_file, prefix=args.judge_config_prefix
        )
    except ConfigError as exc:
        print(f"judge configuration error: {exc}", file=sys.stderr)
        return 2

    print(
        f"grounding_judge_model={config.model} "
        f"(prefix={args.judge_config_prefix!r})"
    )

    jobs, skipped = build_jobs(
        packs=load_jsonl(args.packs),
        candidate_rows=load_jsonl(args.eval),
        modes=parse_csv_set(args.modes),
        pack_ids=set(args.pack_ids) if args.pack_ids else None,
    )
    if skipped and not args.allow_partial:
        print("error: selected candidate rows could not be grounded:", file=sys.stderr)
        for item in skipped[:20]:
            print(f"  {item.cell}: {item.reason}", file=sys.stderr)
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more", file=sys.stderr)
        print("Use --allow-partial to write rows for the resolvable subset.", file=sys.stderr)
        return 1
    if not jobs and not args.allow_empty:
        print(
            "error: no grounding jobs selected; check --eval, --packs, --modes, "
            "and --pack-id filters, or pass --allow-empty",
            file=sys.stderr,
        )
        return 1
    if skipped:
        print(f"warning: skipped {len(skipped)} selected rows due to missing context")
    expected_cells = {row_cell(job.candidate_row) for job in jobs}
    rows = load_resume_success_rows(args.out, expected_cells) if args.resume else []
    completed = {row_cell(row) for row in rows}
    pending = [job for job in jobs if row_cell(job.candidate_row) not in completed]

    print(f"Running {len(pending)} grounding jobs; resume reused {len(rows)} rows")
    results: dict[tuple[str, str, str], dict[str, Any]] = {
        row_cell(row): row for row in rows
    }
    if args.num_threads == 1:
        for index, job in enumerate(pending, start=1):
            cell, row = evaluate_job(
                job,
                config=config,
                max_tokens=args.max_tokens,
                parse_max_attempts=args.parse_max_attempts,
                max_evidence_chars=args.max_evidence_chars,
                max_candidate_chars=args.max_candidate_chars,
            )
            results[cell] = row
            print(f"[{index}/{len(pending)}] {cell[0]} {cell[2]}: {row['status']}")
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(
                    evaluate_job,
                    job,
                    config=config,
                    max_tokens=args.max_tokens,
                    parse_max_attempts=args.parse_max_attempts,
                    max_evidence_chars=args.max_evidence_chars,
                    max_candidate_chars=args.max_candidate_chars,
                )
                for job in pending
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                cell, row = future.result()
                results[cell] = row
                print(f"[{index}/{len(pending)}] {cell[0]} {cell[2]}: {row['status']}")

    ordered = [results[cell] for cell in sorted(results)]
    write_jsonl(args.out, ordered)
    summary = summarize_grounding_rows(ordered)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
