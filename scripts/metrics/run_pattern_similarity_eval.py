#!/usr/bin/env python3
"""Score whether heldout outputs follow user-example patterns."""

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

from auto_skill.eval_summary import ScoreCell, expected_score_cells  # noqa: E402
from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import PromptRunResult, user_examples_from_pack  # noqa: E402
from auto_skill.pattern_similarity import (  # noqa: E402
    EVALUATOR_KIND,
    SCHEMA_VERSION,
    SKILL_AWARE_EVALUATOR_KIND,
    build_pattern_similarity_prompt,
    parse_pattern_similarity_report,
    pattern_similarity_status,
    summarize_pattern_similarity_rows,
)
from scripts.eval.run_heldout_eval import select_packs, skill_index  # noqa: E402

DEFAULT_PARSE_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class PatternJob:
    pack: dict[str, Any]
    task: dict[str, Any]
    mode: str
    candidate_row: dict[str, Any]
    skill_md: str | None


def call_model(
    client: ChatCompletionClient,
    *,
    user_prompt: str,
    max_tokens: int,
) -> PromptRunResult:
    result = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of example-pattern fidelity. "
                    "Return only strict JSON."
                ),
            },
            {"role": "user", "content": user_prompt},
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


def candidate_index(rows: list[dict[str, Any]]) -> dict[ScoreCell, dict[str, Any]]:
    index = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        generation = row.get("generation")
        if not isinstance(generation, dict) or not isinstance(generation.get("text"), str):
            continue
        cell = (
            str(row.get("pack_id") or ""),
            str(row.get("task_id") or ""),
            str(row.get("mode") or ""),
        )
        if all(cell):
            index[cell] = row
    return index


def mode_skill(mode: str, skills: dict[tuple[str, str], str], pack_id: str) -> str | None:
    if mode == "one_shot_skill_from_examples":
        return skills.get((pack_id, "one_shot_skill_from_examples"))
    if mode == "ours_no_validation":
        return skills.get((pack_id, "auto_skill_feature_driven_no_validation"))
    if mode == "auto_skill":
        return skills.get((pack_id, "auto_skill_ours_full"))
    return None


def row_score_cell(row: dict[str, Any]) -> ScoreCell:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("mode") or ""),
    )


def load_resume_success_rows(out: Path, expected_cells: list[ScoreCell]) -> list[dict[str, Any]]:
    if not out.exists():
        return []
    expected = set(expected_cells)
    rows = []
    for row in load_jsonl(out):
        if row.get("status") == "success" and row_score_cell(row) in expected:
            rows.append(row)
    return rows


def eval_missing_candidate_row(
    pack: dict[str, Any],
    task: dict[str, Any],
    mode: str,
    *,
    evaluator_kind: str = EVALUATOR_KIND,
    skill_aware: bool = False,
    judge_model: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": str(pack["pack_id"]),
        "task_id": str(task["task_id"]),
        "source": pack.get("source"),
        "source_task_id": task.get("source_task_id"),
        "mode": mode,
        "evaluator_kind": evaluator_kind,
        "skill_aware": skill_aware,
        "status": "missing_candidate_output",
        "judge": None,
        "pattern_report": None,
        "overall_score": None,
        "solver_model": None,
        "judge_model": judge_model,
    }


def evaluate_job(
    job: PatternJob,
    *,
    config: ChatCompletionConfig,
    max_tokens: int,
    skill_aware: bool,
    parse_max_attempts: int,
) -> tuple[ScoreCell, dict[str, Any]]:
    pack = job.pack
    task = job.task
    pack_id = str(pack["pack_id"])
    task_id = str(task["task_id"])
    cell = (pack_id, task_id, job.mode)
    candidate_output = str(job.candidate_row["generation"]["text"])
    evaluator_kind = SKILL_AWARE_EVALUATOR_KIND if skill_aware else EVALUATOR_KIND
    candidate_solver_model = (
        job.candidate_row.get("solver_model")
        or _candidate_generation_model(job.candidate_row)
    )
    judge_model_id = config.model
    client = ChatCompletionClient(config)
    try:
        prompt = build_pattern_similarity_prompt(
            examples=user_examples_from_pack(pack),
            heldout_task=task,
            mode=job.mode,
            candidate_output=candidate_output,
            skill_md=job.skill_md if skill_aware else None,
        )
        judge, report, status, parse_attempts = judge_with_parse_retry(
            client=client,
            prompt=prompt,
            max_tokens=max_tokens,
            parse_max_attempts=parse_max_attempts,
        )
        score = report.get("pattern_similarity_score") if status == "success" else None
        row = {
            "schema_version": SCHEMA_VERSION,
            "pack_id": pack_id,
            "task_id": task_id,
            "source": pack.get("source"),
            "source_task_id": task.get("source_task_id"),
            "mode": job.mode,
            "evaluator_kind": evaluator_kind,
            "skill_aware": skill_aware,
            "status": status,
            "candidate_eval_status": job.candidate_row.get("status"),
            "judge": judge.to_json(),
            "judge_parse_attempts": parse_attempts,
            "pattern_report": report,
            "overall_score": score,
            "solver_model": candidate_solver_model,
            "judge_model": judge.model or judge_model_id,
        }
        return cell, row
    except Exception as exc:  # noqa: BLE001 - one bad call should not kill the run.
        return cell, {
            "schema_version": SCHEMA_VERSION,
            "pack_id": pack_id,
            "task_id": task_id,
            "source": pack.get("source"),
            "source_task_id": task.get("source_task_id"),
            "mode": job.mode,
            "evaluator_kind": evaluator_kind,
            "skill_aware": skill_aware,
            "status": "model_error",
            "candidate_eval_status": job.candidate_row.get("status"),
            "judge": None,
            "pattern_report": None,
            "overall_score": None,
            "solver_model": candidate_solver_model,
            "judge_model": judge_model_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def judge_with_parse_retry(
    *,
    client: ChatCompletionClient,
    prompt: str,
    max_tokens: int,
    parse_max_attempts: int,
) -> tuple[PromptRunResult, dict[str, Any], str, list[dict[str, Any]]]:
    attempts = max(1, parse_max_attempts)
    parse_attempts: list[dict[str, Any]] = []
    last_judge: PromptRunResult | None = None
    last_report: dict[str, Any] = {"parse_error": "judge_not_called"}
    last_status = "judge_parse_error"

    for attempt in range(1, attempts + 1):
        judge = call_model(client, user_prompt=prompt, max_tokens=max_tokens)
        report = parse_pattern_similarity_report(judge.text)
        status = pattern_similarity_status(
            finish_reason=judge.finish_reason,
            report=report,
        )
        parse_attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "finish_reason": judge.finish_reason,
                "model": judge.model,
                "request_id": judge.request_id,
                "usage": judge.usage,
                "parse_error": report.get("parse_error"),
                "pattern_similarity_score": report.get("pattern_similarity_score"),
            }
        )
        last_judge = judge
        last_report = report
        last_status = status
        if status != "judge_parse_error":
            break

    assert last_judge is not None
    return last_judge, last_report, last_status, parse_attempts


def _candidate_generation_model(candidate_row: dict[str, Any]) -> str | None:
    generation = candidate_row.get("generation")
    if isinstance(generation, dict):
        model = generation.get("model")
        if isinstance(model, str) and model.strip():
            return model
    return None


def append_row(out: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows.append(row)
    write_jsonl(out, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument("--skills", type=Path, default=Path("runs/skill_mvp.qwen.mvp.jsonl"))
    parser.add_argument("--candidate-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--judge-config-prefix",
        default=None,
        help=(
            "If set (e.g. 'JUDGE'), load judge-side config from <PREFIX>_*. "
            "<PREFIX>_MODEL required; URL/key fall back to BAILIAN_*."
        ),
    )
    parser.add_argument("--source", choices=["WritingBench", "PresentBench"])
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument(
        "--modes",
        default=(
            "prompt_only,few_shot_examples_only,one_shot_skill_from_examples,"
            "ours_no_validation,auto_skill"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument(
        "--parse-max-attempts",
        type=int,
        default=DEFAULT_PARSE_MAX_ATTEMPTS,
        help=(
            "Retry judge calls when the provider returns complete but unparseable "
            "JSON. Provider/network retries remain controlled by --max-retries."
        ),
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument(
        "--skill-aware",
        action="store_true",
        help="Include candidate skill text in the judge prompt; debug-only, not a blind metric.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2

    packs = load_jsonl(args.packs)
    selected = select_packs(
        packs,
        pack_ids=set(args.pack_id) if args.pack_id else None,
        source=args.source,
        limit=args.limit_packs,
    )
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    expected_cells = expected_score_cells(selected, modes, limit_heldout=args.limit_heldout)
    candidates = candidate_index(load_jsonl(args.candidate_eval))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected_pack_ids": [pack["pack_id"] for pack in selected],
                    "modes": modes,
                    "candidate_success_cells": len(set(candidates) & set(expected_cells)),
                    "expected_cells": len(expected_cells),
                    "evaluator_kind": SKILL_AWARE_EVALUATOR_KIND
                    if args.skill_aware
                    else EVALUATOR_KIND,
                    "skill_aware": args.skill_aware,
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        if args.judge_config_prefix:
            config = ChatCompletionConfig.from_env(
                args.env_file, prefix=args.judge_config_prefix
            )
        else:
            config = ChatCompletionConfig.from_env(args.env_file)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.timeout_seconds is not None:
        config = replace(config, timeout_seconds=args.timeout_seconds)
    if args.max_retries is not None:
        config = replace(config, max_retries=args.max_retries)
    if args.enable_thinking is not None:
        config = replace(config, enable_thinking=args.enable_thinking)
    if args.stream is not None:
        config = replace(config, stream=args.stream)
    num_threads = args.num_threads if args.num_threads is not None else config.num_threads
    if num_threads <= 0:
        print("error: --num-threads must be positive", file=sys.stderr)
        return 2

    skills = skill_index(load_jsonl(args.skills))
    evaluator_kind = SKILL_AWARE_EVALUATOR_KIND if args.skill_aware else EVALUATOR_KIND
    print(
        f"evaluator_kind={evaluator_kind} skill_aware={args.skill_aware} "
        f"judge_model={config.model}"
        + (f" (prefix={args.judge_config_prefix!r})" if args.judge_config_prefix else ""),
        flush=True,
    )
    rows = load_resume_success_rows(args.out, expected_cells) if args.resume else []
    completed = {row_score_cell(row) for row in rows}
    if args.resume and rows:
        print(f"Loaded {len(rows)} successful pattern rows from {args.out}", flush=True)

    jobs: list[PatternJob] = []
    for pack in selected:
        pack_id = str(pack["pack_id"])
        heldout_tasks = pack.get("heldout_tasks") or []
        if args.limit_heldout is not None:
            heldout_tasks = heldout_tasks[: args.limit_heldout]
        for task in heldout_tasks:
            task_id = str(task["task_id"])
            for mode in modes:
                cell = (pack_id, task_id, mode)
                if cell in completed:
                    continue
                candidate = candidates.get(cell)
                if candidate is None:
                    append_row(
                        args.out,
                        rows,
                        eval_missing_candidate_row(
                            pack,
                            task,
                            mode,
                            evaluator_kind=evaluator_kind,
                            skill_aware=args.skill_aware,
                            judge_model=config.model,
                        ),
                    )
                    continue
                jobs.append(
                    PatternJob(
                        pack=pack,
                        task=task,
                        mode=mode,
                        candidate_row=candidate,
                        skill_md=mode_skill(mode, skills, pack_id),
                    )
                )

    print(f"Running {len(jobs)} pattern-similarity cells with num_threads={num_threads}")
    if num_threads == 1:
        for job in jobs:
            cell, row = evaluate_job(
                job,
                config=config,
                max_tokens=args.max_tokens,
                skill_aware=args.skill_aware,
                parse_max_attempts=args.parse_max_attempts,
            )
            append_row(args.out, rows, row)
            print(f"  {cell[1]} {cell[2]}: {row['status']} {row.get('overall_score')}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    evaluate_job,
                    job,
                    config=config,
                    max_tokens=args.max_tokens,
                    skill_aware=args.skill_aware,
                    parse_max_attempts=args.parse_max_attempts,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                cell, row = future.result()
                append_row(args.out, rows, row)
                print(
                    f"  {cell[1]} {cell[2]}: {row['status']} {row.get('overall_score')}",
                    flush=True,
                )

    summary = summarize_pattern_similarity_rows(
        rows,
        expected_cells=expected_cells,
        evaluator_kind=evaluator_kind,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} pattern rows to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(row.get("status") != "success" for row in rows) and not args.allow_partial:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
