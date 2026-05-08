#!/usr/bin/env python3
"""Generate and score heldout outputs for prompt/skill baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.eval_summary import (  # noqa: E402
    ScoreCell,
    expected_score_cells,
    summarize_score_rows,
)
from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import (  # noqa: E402
    PromptRunResult,
    build_heldout_generation_prompt,
    build_judge_prompt,
    evaluation_criteria,
    extract_overall_score,
    parse_json_object,
    user_examples_from_pack,
)

GENERATION_SYSTEM_PROMPT = """You complete heldout tasks for an auto-skill benchmark.
Use only user-visible examples, reusable skills, task input, and visible materials."""

JUDGE_SYSTEM_PROMPT = """You are a strict benchmark evaluator.
Use the provided rubric/checklist only for scoring. Return strict JSON."""
REFUSAL_FINISH_REASONS = {"content_filter", "safety", "refusal"}


@dataclass(frozen=True)
class HeldoutEvalJob:
    pack: dict[str, Any]
    task: dict[str, Any]
    mode: str
    examples: list[Any]
    private_eval: dict[str, Any]
    skill_md: str | None
    evaluator_kind: str


def call_model(
    client: ChatCompletionClient,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
    max_tokens: int,
) -> PromptRunResult:
    result = client.complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return PromptRunResult(
        text=result.text,
        model=result.model,
        usage=result.usage,
        finish_reason=result.finish_reason,
        request_id=result.request_id,
    )


def private_eval_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for pack in rows:
        pack_id = str(pack.get("pack_id"))
        for item in pack.get("heldout_private", []):
            index[(pack_id, str(item.get("task_ref")))] = item
    return index


def skill_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    index = {}
    for row in rows:
        pack_id = row.get("pack_id")
        mode = row.get("mode")
        skill_md = row.get("skill_md")
        if pack_id and mode and isinstance(skill_md, str) and skill_md.strip():
            index[(str(pack_id), str(mode))] = skill_md
    return index


def load_skill_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def select_packs(
    packs: list[dict[str, Any]],
    *,
    pack_ids: set[str] | None,
    source: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for pack in packs:
        if pack_ids and pack.get("pack_id") not in pack_ids:
            continue
        if source and pack.get("source") != source:
            continue
        if not user_examples_from_pack(pack):
            continue
        selected.append(pack)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def mode_skill(mode: str, skills: dict[tuple[str, str], str], pack_id: str) -> str | None:
    if mode == "one_shot_skill_from_examples":
        return skills.get((pack_id, "one_shot_skill_from_examples"))
    if mode == "ours_no_validation":
        return skills.get((pack_id, "auto_skill_feature_driven_no_validation"))
    if mode == "auto_skill":
        return skills.get((pack_id, "auto_skill_ours_full"))
    return None


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    expected_cells: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    return summarize_score_rows(
        rows,
        evaluator_kind="qwen_llm_rubric_surrogate",
        score_summary_key="mean_overall_score",
        expected_cells=expected_cells,
    )


def has_non_success_rows(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("status") != "success" for row in rows)


def row_score_cell(row: dict[str, Any]) -> ScoreCell:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("mode") or ""),
    )


def successful_score_cells(rows: list[dict[str, Any]]) -> set[ScoreCell]:
    cells: set[ScoreCell] = set()
    for row in rows:
        cell = row_score_cell(row)
        if row.get("status") == "success" and all(cell):
            cells.add(cell)
    return cells


def load_resume_success_rows(out: Path, expected_cells: list[ScoreCell]) -> list[dict[str, Any]]:
    if not out.exists():
        return []
    expected = set(expected_cells)
    rows = []
    for row in load_jsonl(out):
        if row.get("status") != "success":
            continue
        if row_score_cell(row) in expected:
            rows.append(row)
    return rows


def append_checkpoint_row(out: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows.append(row)
    write_jsonl(out, rows)


def write_empty_eval_result(out: Path, summary_out: Path) -> None:
    write_jsonl(out, [])
    summary = summarize_rows([])
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote 0 eval rows to {out}")
    print(f"Wrote summary to {summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def is_valid_overall_score(score: float | None) -> bool:
    if isinstance(score, bool):
        return False
    return isinstance(score, (int, float)) and math.isfinite(float(score)) and 1 <= score <= 10


def score_row_status(
    *,
    generation: PromptRunResult,
    judge: PromptRunResult,
    judge_report: dict[str, Any],
    overall_score: float | None,
) -> str:
    """Return the fail-closed status for one generated-and-judged row."""

    if generation.finish_reason != "stop":
        return "generation_incomplete"
    if judge.finish_reason in REFUSAL_FINISH_REASONS:
        return "judge_refusal"
    if judge.finish_reason != "stop":
        return "judge_incomplete"
    if "parse_error" in judge_report or overall_score is None:
        return "judge_parse_error"
    if not is_valid_overall_score(overall_score):
        return "judge_invalid_score"
    return "success"


def judge_with_parse_retry(
    *,
    judge_client: ChatCompletionClient,
    judge_prompt: str,
    generation: PromptRunResult,
    judge_max_tokens: int,
    parse_max_attempts: int,
) -> tuple[PromptRunResult, dict[str, Any], float | None, list[dict[str, Any]]]:
    attempts = max(1, parse_max_attempts)
    parse_attempts: list[dict[str, Any]] = []
    last_judge: PromptRunResult | None = None
    last_report: dict[str, Any] = {"parse_error": "judge_not_called"}
    last_score: float | None = None

    for attempt in range(1, attempts + 1):
        judge = call_model(
            judge_client,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=judge_prompt,
            temperature=0.0,
            max_tokens=judge_max_tokens,
        )
        judge_report = parse_json_object(judge.text)
        overall_score = extract_overall_score(judge_report)
        status = score_row_status(
            generation=generation,
            judge=judge,
            judge_report=judge_report,
            overall_score=overall_score,
        )
        parse_attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "finish_reason": judge.finish_reason,
                "model": judge.model,
                "request_id": judge.request_id,
                "usage": judge.usage,
                "parse_error": judge_report.get("parse_error"),
                "overall_score": overall_score,
            }
        )
        last_judge = judge
        last_report = judge_report
        last_score = overall_score
        if status in {"success", "judge_incomplete", "judge_refusal"}:
            break
        if status == "judge_invalid_score" and attempt >= attempts:
            break
        if status == "judge_parse_error" and attempt >= attempts:
            break

    assert last_judge is not None
    return last_judge, last_report, last_score, parse_attempts


def eval_failure_row(
    *,
    pack: dict[str, Any],
    task: dict[str, Any],
    mode: str,
    status: str,
    evaluator_kind: str,
    solver_model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "heldout-eval/v1",
        "pack_id": str(pack["pack_id"]),
        "task_id": str(task["task_id"]),
        "source": pack.get("source"),
        "source_task_id": task.get("source_task_id"),
        "mode": mode,
        "evaluator_kind": evaluator_kind,
        "status": status,
        "generation": None,
        "judge": None,
        "judge_report": None,
        "overall_score": None,
        "solver_model": solver_model,
        "judge_model": judge_model,
    }


def eval_model_error_row(
    *,
    pack: dict[str, Any],
    task: dict[str, Any],
    mode: str,
    evaluator_kind: str,
    error: Exception,
    solver_model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    row = eval_failure_row(
        pack=pack,
        task=task,
        mode=mode,
        status="model_error",
        evaluator_kind=evaluator_kind,
        solver_model=solver_model,
        judge_model=judge_model,
    )
    row["error"] = f"{type(error).__name__}: {error}"
    return row


def evaluate_heldout_job(
    job: HeldoutEvalJob,
    *,
    config: ChatCompletionConfig,
    temperature: float | None,
    max_tokens: int,
    judge_max_tokens: int,
    max_material_chars: int,
    judge_config: ChatCompletionConfig | None = None,
    parse_max_attempts: int = 1,
) -> tuple[ScoreCell, dict[str, Any], str]:
    pack = job.pack
    task = job.task
    pack_id = str(pack["pack_id"])
    task_id = str(task["task_id"])
    cell = (pack_id, task_id, job.mode)
    client = ChatCompletionClient(config)
    judge_client = ChatCompletionClient(judge_config) if judge_config is not None else client
    solver_model_id = config.model
    judge_model_id = (judge_config or config).model
    try:
        prompt = build_heldout_generation_prompt(
            task=task,
            mode=job.mode,
            examples=job.examples,
            skill_md=job.skill_md,
            max_material_chars=max_material_chars,
        )
        generation = call_model(
            client,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if generation.finish_reason != "stop":
            row = {
                "schema_version": "heldout-eval/v1",
                "pack_id": pack_id,
                "task_id": task_id,
                "source": pack.get("source"),
                "mode": job.mode,
                "evaluator_kind": job.evaluator_kind,
                "status": "generation_incomplete",
                "generation": generation.to_json(),
                "judge": None,
                "judge_report": None,
                "overall_score": None,
                "solver_model": generation.model or solver_model_id,
                "judge_model": judge_model_id,
            }
            return cell, row, f"generation_incomplete ({generation.finish_reason})"
        judge_prompt = build_judge_prompt(
            task=task,
            candidate_output=generation.text,
            private_eval=job.private_eval,
        )
        judge, judge_report, overall_score, judge_parse_attempts = judge_with_parse_retry(
            judge_client=judge_client,
            judge_prompt=judge_prompt,
            generation=generation,
            judge_max_tokens=judge_max_tokens,
            parse_max_attempts=parse_max_attempts,
        )
        status = score_row_status(
            generation=generation,
            judge=judge,
            judge_report=judge_report,
            overall_score=overall_score,
        )
        row = {
            "schema_version": "heldout-eval/v1",
            "pack_id": pack_id,
            "task_id": task_id,
            "source": pack.get("source"),
            "mode": job.mode,
            "evaluator_kind": job.evaluator_kind,
            "status": status,
            "generation": generation.to_json(),
            "judge": judge.to_json(),
            "judge_parse_attempts": judge_parse_attempts,
            "judge_report": judge_report,
            "overall_score": overall_score,
            "solver_model": generation.model or solver_model_id,
            "judge_model": judge.model or judge_model_id,
        }
        return cell, row, str(overall_score)
    except Exception as exc:  # noqa: BLE001 - provider failures should not kill whole eval.
        row = eval_model_error_row(
            pack=pack,
            task=task,
            mode=job.mode,
            evaluator_kind=job.evaluator_kind,
            error=exc,
            solver_model=solver_model_id,
            judge_model=judge_model_id,
        )
        return cell, row, row["error"]


def run_eval_jobs(
    jobs: list[HeldoutEvalJob],
    *,
    config: ChatCompletionConfig,
    temperature: float | None,
    max_tokens: int,
    judge_max_tokens: int,
    max_material_chars: int,
    num_threads: int,
    out: Path,
    rows: list[dict[str, Any]],
    completed_cells: set[ScoreCell],
    judge_config: ChatCompletionConfig | None = None,
    parse_max_attempts: int = 1,
) -> None:
    def run_one(job: HeldoutEvalJob) -> tuple[ScoreCell, dict[str, Any], str]:
        return evaluate_heldout_job(
            job,
            config=config,
            temperature=temperature,
            max_tokens=max_tokens,
            judge_max_tokens=judge_max_tokens,
            max_material_chars=max_material_chars,
            judge_config=judge_config,
            parse_max_attempts=parse_max_attempts,
        )

    if num_threads == 1:
        for job in jobs:
            cell, row, message = run_one(job)
            append_checkpoint_row(out, rows, row)
            if row.get("status") == "success":
                completed_cells.add(cell)
            print(f"  {cell[1]} {cell[2]}: {message}", flush=True)
        return

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_cell = {
            executor.submit(run_one, job): (
                str(job.pack["pack_id"]),
                str(job.task["task_id"]),
                job.mode,
            )
            for job in jobs
        }
        for future in as_completed(future_to_cell):
            cell, row, message = future.result()
            append_checkpoint_row(out, rows, row)
            if row.get("status") == "success":
                completed_cells.add(cell)
            print(f"  {cell[1]} {cell[2]}: {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument(
        "--skills",
        type=Path,
        action="append",
        help=(
            "Skill rows JSONL. Repeat to merge MVP and ours_full skill artifacts. "
            "Defaults to runs/skill_mvp.qwen.jsonl."
        ),
    )
    parser.add_argument(
        "--private-eval",
        type=Path,
        default=Path("artifacts/private/example_private_eval.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/heldout_eval.qwen.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("runs/heldout_eval.summary.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--judge-config-prefix",
        default=None,
        help=(
            "If set (e.g. 'JUDGE'), load surrogate-judge config from <PREFIX>_*. "
            "<PREFIX>_MODEL is required; URL/key fall back to BAILIAN_*."
        ),
    )
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--source", choices=["WritingBench", "PresentBench"])
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument(
        "--modes",
        default=(
            "prompt_only,few_shot_examples_only,one_shot_skill_from_examples,"
            "ours_no_validation,auto_skill"
        ),
        help="Comma-separated modes.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--max-material-chars", type=int, default=4000)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument(
        "--parse-max-attempts",
        type=int,
        default=1,
        help=(
            "Retry judge calls when a complete response cannot be parsed into "
            "a valid numeric score. Provider/network retries remain controlled "
            "by --max-retries."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        help=(
            "Concurrent pack/task/mode cells. Each cell still runs generation "
            "then judge sequentially."
        ),
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-log", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even when some eval rows fail. Default is fail-closed.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 when filters select no evaluable rows. Default is fail-closed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing success rows from --out and skip completed pack/task/mode cells.",
    )
    parser.add_argument(
        "--evaluator-kind",
        default="qwen_llm_rubric_surrogate",
        choices=["qwen_llm_rubric_surrogate"],
        help="Current scoring mode. This is not the official PresentBench visual evaluator.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    packs = load_jsonl(args.packs)
    selected = select_packs(
        packs,
        pack_ids=set(args.pack_id) if args.pack_id else None,
        source=args.source,
        limit=args.limit_packs,
    )
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2
    if not selected and not args.dry_run and not args.allow_empty:
        print("error: no packs selected for heldout evaluation", file=sys.stderr)
        return 3
    if not selected and not args.dry_run and args.allow_empty:
        write_empty_eval_result(args.out, args.summary_out)
        return 0

    if args.dry_run:
        task_count = sum(
            min(len(pack.get("heldout_tasks", [])), args.limit_heldout or 10**9)
            for pack in selected
        )
        print(
            json.dumps(
                {
                    "selected_pack_ids": [pack["pack_id"] for pack in selected],
                    "modes": modes,
                    "heldout_tasks": task_count,
                    "model_calls_estimate": task_count * len(modes) * 2,
                    "evaluator_kind": args.evaluator_kind,
                }
            )
        )
        return 0

    try:
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
    if args.thinking_budget is not None:
        config = replace(config, thinking_budget=args.thinking_budget)
    if args.stream or args.stream_log:
        config = replace(config, stream=True, stream_log=args.stream_log)
    num_threads = args.num_threads if args.num_threads is not None else config.num_threads
    if num_threads <= 0:
        print("error: --num-threads must be positive", file=sys.stderr)
        return 2
    if num_threads > 1 and config.stream_log:
        print("error: --stream-log is not supported with --num-threads > 1", file=sys.stderr)
        return 2
    judge_config: ChatCompletionConfig | None = None
    if args.judge_config_prefix:
        try:
            judge_config = ChatCompletionConfig.from_env(
                args.env_file, prefix=args.judge_config_prefix
            )
        except ConfigError as exc:
            print(f"judge configuration error: {exc}", file=sys.stderr)
            return 2
        print(
            f"solver_model={config.model}  judge_model={judge_config.model} "
            f"(prefix={args.judge_config_prefix!r})",
            flush=True,
        )
    else:
        print(
            f"solver_model={config.model}  judge_model={config.model} "
            "(monoculture; pass --judge-config-prefix to split)",
            flush=True,
        )
    solver_model = config.model
    judge_model = (judge_config or config).model
    temperature = args.temperature if args.temperature is not None else config.temperature
    private_index = private_eval_index(load_jsonl(args.private_eval))
    skill_paths = args.skills or [Path("runs/skill_mvp.qwen.jsonl")]
    skills = skill_index(load_skill_rows(skill_paths))

    expected_cells = expected_score_cells(
        selected,
        modes,
        limit_heldout=args.limit_heldout,
    )
    rows = load_resume_success_rows(args.out, expected_cells) if args.resume else []
    completed_cells = successful_score_cells(rows)
    if args.resume and rows:
        print(f"Loaded {len(rows)} successful checkpoint rows from {args.out}", flush=True)
    jobs: list[HeldoutEvalJob] = []
    for pack_index, pack in enumerate(selected, start=1):
        pack_id = str(pack["pack_id"])
        examples = user_examples_from_pack(pack)
        heldout_tasks = pack.get("heldout_tasks", [])
        if args.limit_heldout is not None:
            heldout_tasks = heldout_tasks[: args.limit_heldout]
        print(
            f"[{pack_index}/{len(selected)}] evaluating {pack_id} "
            f"({len(heldout_tasks)} heldout tasks)",
            flush=True,
        )
        for task in heldout_tasks:
            task_id = str(task["task_id"])
            private_eval = private_index.get((pack_id, task_id))
            for mode in modes:
                cell = (pack_id, task_id, mode)
                if cell in completed_cells:
                    print(f"  {task_id} {mode}: skipped checkpoint success", flush=True)
                    continue
                if private_eval is None:
                    append_checkpoint_row(
                        args.out,
                        rows,
                        eval_failure_row(
                            pack=pack,
                            task=task,
                            mode=mode,
                            status="missing_private_eval",
                            evaluator_kind=args.evaluator_kind,
                            solver_model=solver_model,
                            judge_model=judge_model,
                        )
                    )
                    continue
                if not evaluation_criteria(private_eval):
                    append_checkpoint_row(
                        args.out,
                        rows,
                        eval_failure_row(
                            pack=pack,
                            task=task,
                            mode=mode,
                            status="missing_eval_criteria",
                            evaluator_kind=args.evaluator_kind,
                            solver_model=solver_model,
                            judge_model=judge_model,
                        )
                    )
                    continue
                skill_md = mode_skill(mode, skills, pack_id)
                if (
                    mode in {"one_shot_skill_from_examples", "ours_no_validation", "auto_skill"}
                    and not skill_md
                ):
                    append_checkpoint_row(
                        args.out,
                        rows,
                        eval_failure_row(
                            pack=pack,
                            task=task,
                            mode=mode,
                            status=(
                                "missing_ours_full_skill"
                                if mode == "auto_skill"
                                else (
                                    "missing_no_validation_skill"
                                    if mode == "ours_no_validation"
                                    else "missing_skill"
                                )
                            ),
                            evaluator_kind=args.evaluator_kind,
                            solver_model=solver_model,
                            judge_model=judge_model,
                        )
                    )
                    continue
                jobs.append(
                    HeldoutEvalJob(
                        pack=pack,
                        task=task,
                        mode=mode,
                        examples=examples,
                        private_eval=private_eval,
                        skill_md=skill_md,
                        evaluator_kind=args.evaluator_kind,
                    )
                )

    print(f"Running {len(jobs)} eval cells with num_threads={num_threads}", flush=True)
    run_eval_jobs(
        jobs,
        config=config,
        temperature=temperature,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        max_material_chars=args.max_material_chars,
        num_threads=num_threads,
        out=args.out,
        rows=rows,
        completed_cells=completed_cells,
        judge_config=judge_config,
        parse_max_attempts=args.parse_max_attempts,
    )

    write_jsonl(args.out, rows)
    summary = summarize_rows(rows, expected_cells=expected_cells)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} eval rows to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not rows and not args.allow_empty:
        return 3
    if has_non_success_rows(rows) and not args.allow_partial:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
