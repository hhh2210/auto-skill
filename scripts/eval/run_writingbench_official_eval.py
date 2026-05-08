#!/usr/bin/env python3
"""Generate heldout outputs and score them with the WritingBench evaluator prompt."""

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

from auto_skill.eval_summary import (  # noqa: E402
    ScoreCell,
    expected_score_cells,
    summarize_score_rows,
)
from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import (  # noqa: E402
    HELDOUT_GENERATION_PROMPT_VERSION,
    PromptRunResult,
    build_heldout_generation_prompt,
    user_examples_from_pack,
)
from auto_skill.writingbench_eval import (  # noqa: E402
    average_writingbench_scores,
    build_writingbench_official_prompt,
    load_writingbench_prompt_templates,
    parse_writingbench_score,
)
from scripts.eval.run_heldout_eval import (  # noqa: E402
    GENERATION_SYSTEM_PROMPT,
    append_checkpoint_row,
    load_skill_rows,
    private_eval_index,
    select_packs,
    skill_index,
    successful_score_cells,
)

JUDGE_KIND = "writingbench_official_prompt_qwen_judge"
REFUSAL_FINISH_REASONS = {"content_filter", "safety", "refusal"}
SKILL_REQUIRED_MODES = {
    "one_shot_skill_from_examples",
    "ours_no_validation",
    "auto_skill",
    "examples_plus_one_shot_skill",
    "examples_plus_feature_skill",
    "slide_constrained_examples_plus_feature_skill",
}


@dataclass(frozen=True)
class WritingBenchEvalJob:
    pack: dict[str, Any]
    task: dict[str, Any]
    mode: str
    examples: list[Any]
    criteria: list[dict[str, Any]]
    skill_md: str | None
    reused_row: dict[str, Any] | None = None


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


def mode_skill(mode: str, skills: dict[tuple[str, str], str], pack_id: str) -> str | None:
    if mode in {"one_shot_skill_from_examples", "examples_plus_one_shot_skill"}:
        return skills.get((pack_id, "one_shot_skill_from_examples"))
    if mode in {
        "ours_no_validation",
        "examples_plus_feature_skill",
        "slide_constrained_examples_plus_feature_skill",
    }:
        return skills.get((pack_id, "auto_skill_feature_driven_no_validation"))
    if mode == "auto_skill":
        return skills.get((pack_id, "auto_skill_ours_full"))
    return None


def is_reusable_candidate_row(row: dict[str, Any] | None) -> bool:
    if row is None or row.get("status") != "success":
        return False
    generation = row.get("generation")
    return isinstance(generation, dict) and isinstance(generation.get("text"), str)


def mode_needs_skill(mode: str, reused_row: dict[str, Any] | None) -> bool:
    return mode in SKILL_REQUIRED_MODES and reused_row is None


def row_score_cell(row: dict[str, Any]) -> ScoreCell:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("mode") or ""),
    )


def runtime_metadata(
    *,
    config: ChatCompletionConfig,
    judge_config: ChatCompletionConfig | None,
    reuse_candidates_from: Path | None,
    temperature: float | None,
    max_tokens: int,
    judge_max_tokens: int,
    max_material_chars: int,
) -> dict[str, Any]:
    """Return metadata that defines whether checkpoint rows are compatible."""

    effective_judge_config = judge_config or config
    metadata = {
        "reused_candidates_from": str(reuse_candidates_from) if reuse_candidates_from else None,
        "judge_enable_thinking": effective_judge_config.enable_thinking,
        "judge_thinking_budget": effective_judge_config.thinking_budget,
        "judge_max_tokens": judge_max_tokens,
    }
    if reuse_candidates_from is None:
        metadata["heldout_generation_prompt_version"] = HELDOUT_GENERATION_PROMPT_VERSION
        metadata["solver_model"] = config.model
        metadata["solver_temperature"] = temperature
        metadata["solver_max_tokens"] = max_tokens
        metadata["max_material_chars"] = max_material_chars
        metadata["solver_enable_thinking"] = config.enable_thinking
        metadata["solver_thinking_budget"] = config.thinking_budget
    return metadata


def row_matches_runtime(
    row: dict[str, Any],
    *,
    expected_judge_model: str,
    metadata: dict[str, Any],
) -> bool:
    if row.get("judge_model") != expected_judge_model:
        return False
    for key, value in metadata.items():
        if key not in row:
            return False
        if row[key] != value:
            return False
    return True


def load_compatible_resume_success_rows(
    out: Path,
    expected_cells: list[ScoreCell],
    *,
    expected_judge_model: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if not out.exists():
        return []
    expected = set(expected_cells)
    rows = []
    for row in load_jsonl(out):
        if row.get("status") != "success":
            continue
        if row_score_cell(row) not in expected:
            continue
        if not row_matches_runtime(
            row,
            expected_judge_model=expected_judge_model,
            metadata=metadata,
        ):
            continue
        rows.append(row)
    return rows


def existing_rows_outside_expected_cells(
    out: Path, expected_cells: list[ScoreCell]
) -> list[ScoreCell]:
    if not out.exists():
        return []
    expected = set(expected_cells)
    outside: list[ScoreCell] = []
    seen: set[ScoreCell] = set()
    for row in load_jsonl(out):
        cell = row_score_cell(row)
        if cell not in expected and cell not in seen:
            outside.append(cell)
            seen.add(cell)
    return outside


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    expected_cells: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    return summarize_score_rows(
        rows,
        evaluator_kind=JUDGE_KIND,
        score_summary_key="mean_writingbench_score",
        expected_cells=expected_cells,
    )


def has_non_success_rows(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("status") != "success" for row in rows)


def write_empty_eval_result(out: Path, summary_out: Path) -> None:
    write_jsonl(out, [])
    summary = summarize_rows([])
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote 0 eval rows to {out}")
    print(f"Wrote summary to {summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def score_with_writingbench_prompt(
    *,
    client: ChatCompletionClient,
    system_prompt: str,
    prompt_template: str,
    query: str,
    response: str,
    criteria: list[dict[str, Any]],
    max_tokens: int,
    judge_client: ChatCompletionClient | None = None,
    parse_max_attempts: int = 1,
) -> tuple[str, dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    scores: dict[str, list[dict[str, Any]]] = {}
    judge_calls: list[dict[str, Any]] = []
    status = "success"
    judge_runner = judge_client or client

    for criterion in criteria:
        name = str(criterion.get("name") or f"criterion_{len(scores) + 1}")
        prompt = build_writingbench_official_prompt(
            template=prompt_template,
            query=query,
            response=response,
            criteria=criterion,
        )
        parsed: dict[str, Any] | None = None
        judge: PromptRunResult | None = None
        for attempt in range(1, max(1, parse_max_attempts) + 1):
            judge = call_model(
                judge_runner,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            parsed = parse_writingbench_score(judge.text)
            judge_calls.append(
                {
                    "criterion": name,
                    "attempt": attempt,
                    "finish_reason": judge.finish_reason,
                    "model": judge.model,
                    "usage": judge.usage,
                    "parse_error": parsed.get("parse_error"),
                    "score": parsed.get("score"),
                }
            )
            if judge.finish_reason != "stop" or "parse_error" not in parsed:
                break
        assert judge is not None
        assert parsed is not None
        scores.setdefault(name, []).append(parsed)
        if judge.finish_reason in REFUSAL_FINISH_REASONS:
            status = "judge_refusal"
            break
        if judge.finish_reason != "stop":
            status = "judge_incomplete"
            break
        if "parse_error" in parsed:
            status = "judge_parse_error"
            break
    return status, scores, judge_calls


def eval_failure_row(
    *,
    pack: dict[str, Any],
    task: dict[str, Any],
    mode: str,
    status: str,
    solver_model: str | None = None,
    judge_model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": "writingbench-official-eval/v1",
        "pack_id": str(pack["pack_id"]),
        "task_id": str(task["task_id"]),
        "source_task_id": task.get("source_task_id"),
        "mode": mode,
        "evaluator_kind": JUDGE_KIND,
        "status": status,
        "generation": None,
        "judge_calls": [],
        "scores": {},
        "overall_score": None,
        "solver_model": solver_model,
        "judge_model": judge_model,
    }
    if metadata:
        row.update(metadata)
    return row


def eval_model_error_row(
    *,
    pack: dict[str, Any],
    task: dict[str, Any],
    mode: str,
    error: Exception,
    solver_model: str | None = None,
    judge_model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = eval_failure_row(
        pack=pack,
        task=task,
        mode=mode,
        status="model_error",
        solver_model=solver_model,
        judge_model=judge_model,
        metadata=metadata,
    )
    row["error"] = f"{type(error).__name__}: {error}"
    return row


def evaluate_writingbench_job(
    job: WritingBenchEvalJob,
    *,
    config: ChatCompletionConfig,
    templates: Any,
    temperature: float | None,
    max_tokens: int,
    judge_max_tokens: int,
    max_material_chars: int,
    metadata: dict[str, Any],
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
        if job.reused_row is not None:
            generation = PromptRunResult(**job.reused_row["generation"])
        else:
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
                "schema_version": "writingbench-official-eval/v1",
                "pack_id": pack_id,
                "task_id": task_id,
                "source_task_id": task.get("source_task_id"),
                "mode": job.mode,
                "evaluator_kind": JUDGE_KIND,
                "status": "generation_incomplete",
                "generation": generation.to_json(),
                "scores": {},
                "overall_score": None,
                "solver_model": generation.model or solver_model_id,
                "judge_model": judge_model_id,
            }
            row.update(metadata)
            return cell, row, "generation_incomplete"

        status, scores, judge_calls = score_with_writingbench_prompt(
            client=client,
            system_prompt=templates.evaluate_system,
            prompt_template=templates.evaluate_prompt,
            query=str(task["task_input"]),
            response=generation.text,
            criteria=job.criteria,
            max_tokens=judge_max_tokens,
            judge_client=judge_client,
            parse_max_attempts=parse_max_attempts,
        )
        overall_score = average_writingbench_scores(scores) if status == "success" else None
        first_judge_model = next(
            (call.get("model") for call in judge_calls if call.get("model")),
            judge_model_id,
        )
        row = {
            "schema_version": "writingbench-official-eval/v1",
            "pack_id": pack_id,
            "task_id": task_id,
            "source_task_id": task.get("source_task_id"),
            "mode": job.mode,
            "evaluator_kind": JUDGE_KIND,
            "official_prompt_file": templates.prompt_file,
            "status": status,
            "generation": generation.to_json(),
            "judge_calls": judge_calls,
            "scores": scores,
            "overall_score": overall_score,
            "solver_model": generation.model or solver_model_id,
            "judge_model": first_judge_model,
        }
        row.update(metadata)
        return cell, row, f"{overall_score} ({status})"
    except Exception as exc:  # noqa: BLE001 - provider failures should not kill whole eval.
        row = eval_model_error_row(
            pack=pack,
            task=task,
            mode=job.mode,
            error=exc,
            solver_model=solver_model_id,
            judge_model=judge_model_id,
            metadata=metadata,
        )
        return cell, row, row["error"]


def run_eval_jobs(
    jobs: list[WritingBenchEvalJob],
    *,
    config: ChatCompletionConfig,
    templates: Any,
    temperature: float | None,
    max_tokens: int,
    judge_max_tokens: int,
    max_material_chars: int,
    metadata: dict[str, Any],
    num_threads: int,
    out: Path,
    rows: list[dict[str, Any]],
    completed_cells: set[ScoreCell],
    judge_config: ChatCompletionConfig | None = None,
    parse_max_attempts: int = 1,
) -> None:
    def run_one(job: WritingBenchEvalJob) -> tuple[ScoreCell, dict[str, Any], str]:
        return evaluate_writingbench_job(
            job,
            config=config,
            templates=templates,
            temperature=temperature,
            max_tokens=max_tokens,
            judge_max_tokens=judge_max_tokens,
            max_material_chars=max_material_chars,
            metadata=metadata,
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
    parser.add_argument(
        "--writingbench-root",
        type=Path,
        default=Path("../WritingBench"),
        help="Local WritingBench repository containing prompt.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/writingbench_official_eval.qwen.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("runs/writingbench_official_eval.qwen.summary.json"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--reuse-candidates-from",
        type=Path,
        default=None,
        help=(
            "Path to an existing writingbench-official-eval JSONL. When set, "
            "skip solver generation and reuse row['generation'] for each "
            "(pack_id, task_id, mode) cell whose source row had status=success. "
            "Cells without a reusable candidate are recorded as "
            "'missing_reused_candidate'. Use this for judge-swap A/B without "
            "re-burning solver tokens or adding sampling noise."
        ),
    )
    parser.add_argument(
        "--judge-config-prefix",
        default=None,
        help=(
            "If set (e.g. 'JUDGE'), load judge-side config from <PREFIX>_BASE_URL / "
            "<PREFIX>_API_KEY / <PREFIX>_MODEL. <PREFIX>_MODEL must be set; URL/key "
            "fall back to BAILIAN_*. Use this to break Qwen-judge monoculture."
        ),
    )
    parser.add_argument("--pack-id", action="append")
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
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--max-material-chars", type=int, default=4000)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument(
        "--parse-max-attempts",
        type=int,
        default=1,
        help=(
            "Retry each WritingBench judge criterion when a complete response "
            "cannot be parsed as a valid score. Provider/network retries remain "
            "controlled by --max-retries."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Concurrent WritingBench cells. Defaults to *_NUM_THREADS from env.",
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument(
        "--judge-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override judge-side thinking independently from solver thinking.",
    )
    parser.add_argument(
        "--judge-thinking-budget",
        type=int,
        help="Judge-side thinking token budget when judge thinking is enabled.",
    )
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
        "--allow-output-prune",
        action="store_true",
        help=(
            "Allow rewriting an existing --out file with only the currently selected "
            "cells. Without this flag, --resume fails closed when --out contains "
            "rows outside the selected pack/task/mode set."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing success rows from --out and skip completed pack/task/mode cells.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reuse_enabled = args.reuse_candidates_from is not None

    packs = load_jsonl(args.packs)
    selected = select_packs(
        [pack for pack in packs if pack.get("source") == "WritingBench"],
        pack_ids=set(args.pack_id) if args.pack_id else None,
        source=None,
        limit=args.limit_packs,
    )
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unsupported_modes = sorted(set(modes) & {"layout_plan_examples_plus_feature_skill"})
    if unsupported_modes:
        print(
            "error: layout_plan_examples_plus_feature_skill is PresentBench-surrogate only; "
            "use scripts/eval/run_heldout_eval.py for that ablation.",
            file=sys.stderr,
        )
        return 2
    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2
    if not selected and not args.dry_run and not args.allow_empty:
        print(
            "error: no WritingBench packs selected for official-prompt evaluation",
            file=sys.stderr,
        )
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
                    "model_calls_estimate": task_count * len(modes) * 6,
                    "evaluator_kind": JUDGE_KIND,
                    "writingbench_prompt": str(args.writingbench_root / "prompt.py"),
                }
            )
        )
        return 0

    try:
        config = ChatCompletionConfig.from_env(args.env_file)
        templates = load_writingbench_prompt_templates(args.writingbench_root)
    except (ConfigError, OSError, ImportError, AttributeError) as exc:
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

    temperature = args.temperature if args.temperature is not None else config.temperature
    if args.judge_config_prefix:
        try:
            judge_config = ChatCompletionConfig.from_env(
                args.env_file, prefix=args.judge_config_prefix
            )
        except ConfigError as exc:
            print(f"judge configuration error: {exc}", file=sys.stderr)
            return 2
        if args.timeout_seconds is not None:
            judge_config = replace(judge_config, timeout_seconds=args.timeout_seconds)
        if args.max_retries is not None:
            judge_config = replace(judge_config, max_retries=args.max_retries)
        if args.judge_enable_thinking is not None:
            judge_config = replace(judge_config, enable_thinking=args.judge_enable_thinking)
        if args.judge_thinking_budget is not None:
            judge_config = replace(judge_config, thinking_budget=args.judge_thinking_budget)
        print(
            f"solver_model={config.model}  judge_model={judge_config.model} "
            f"(prefix={args.judge_config_prefix!r})",
            flush=True,
        )
    else:
        judge_config = None
        if args.judge_enable_thinking is not None or args.judge_thinking_budget is not None:
            judge_config = replace(
                config,
                enable_thinking=(
                    config.enable_thinking
                    if args.judge_enable_thinking is None
                    else args.judge_enable_thinking
                ),
                thinking_budget=(
                    config.thinking_budget
                    if args.judge_thinking_budget is None
                    else args.judge_thinking_budget
                ),
            )
        print(
            f"solver_model={config.model}  judge_model={(judge_config or config).model} "
            "(monoculture; pass --judge-config-prefix to split)",
            flush=True,
        )
    solver_model = config.model
    judge_model = (judge_config or config).model
    private_index = private_eval_index(load_jsonl(args.private_eval))
    reuse_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if reuse_enabled:
        for source_row in load_jsonl(args.reuse_candidates_from):
            key = (
                str(source_row.get("pack_id") or ""),
                str(source_row.get("task_id") or ""),
                str(source_row.get("mode") or ""),
            )
            if all(key):
                reuse_index[key] = source_row
        print(
            f"reuse-candidates-from {args.reuse_candidates_from}: "
            f"{len(reuse_index)} source rows indexed",
            flush=True,
        )
    skill_paths = args.skills or [Path("runs/skill_mvp.qwen.jsonl")]
    skills = {} if reuse_enabled else skill_index(load_skill_rows(skill_paths))

    expected_cells = expected_score_cells(
        selected,
        modes,
        limit_heldout=args.limit_heldout,
    )
    metadata = runtime_metadata(
        config=config,
        judge_config=judge_config,
        reuse_candidates_from=args.reuse_candidates_from,
        temperature=temperature,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        max_material_chars=args.max_material_chars,
    )
    outside_cells = existing_rows_outside_expected_cells(args.out, expected_cells)
    if args.resume and outside_cells and not args.allow_output_prune:
        preview = [
            {"pack_id": pack_id, "task_id": task_id, "mode": mode}
            for pack_id, task_id, mode in outside_cells[:5]
        ]
        print(
            "error: existing --out contains rows outside the selected cells; "
            "use a new --out path or pass --allow-output-prune if truncating is intentional. "
            f"examples={json.dumps(preview, ensure_ascii=False)}",
            file=sys.stderr,
        )
        return 2
    rows = (
        load_compatible_resume_success_rows(
            args.out,
            expected_cells,
            expected_judge_model=judge_model,
            metadata=metadata,
        )
        if args.resume
        else []
    )
    completed_cells = successful_score_cells(rows)
    if args.resume and rows:
        print(f"Loaded {len(rows)} successful checkpoint rows from {args.out}", flush=True)
    jobs: list[WritingBenchEvalJob] = []
    for pack_index, pack in enumerate(selected, start=1):
        pack_id = str(pack["pack_id"])
        examples = user_examples_from_pack(pack)
        heldout_tasks = pack.get("heldout_tasks", [])
        if args.limit_heldout is not None:
            heldout_tasks = heldout_tasks[: args.limit_heldout]
        print(
            f"[{pack_index}/{len(selected)}] WritingBench official eval {pack_id} "
            f"({len(heldout_tasks)} heldout tasks)",
            flush=True,
        )
        for task in heldout_tasks:
            task_id = str(task["task_id"])
            private_eval = private_index.get((pack_id, task_id))
            criteria = []
            if private_eval:
                supervision = private_eval.get("supervision") or {}
                raw_items = supervision.get("items") or supervision.get("criteria") or []
                criteria = [item for item in raw_items if isinstance(item, dict)]
            for mode in modes:
                cell = (pack_id, task_id, mode)
                if cell in completed_cells:
                    print(f"  {task_id} {mode}: skipped checkpoint success", flush=True)
                    continue
                reused_row = reuse_index.get(cell) if reuse_enabled else None
                if reuse_enabled and not is_reusable_candidate_row(reused_row):
                    append_checkpoint_row(
                        args.out,
                        rows,
                        eval_failure_row(
                            pack=pack,
                            task=task,
                            mode=mode,
                            status="missing_reused_candidate",
                            solver_model=(reused_row or {}).get("solver_model"),
                            judge_model=judge_model,
                            metadata=metadata,
                        ),
                    )
                    print(f"  {task_id} {mode}: missing_reused_candidate", flush=True)
                    continue
                skill_md = mode_skill(mode, skills, pack_id)
                if mode_needs_skill(mode, reused_row) and not skill_md:
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
                            solver_model=solver_model,
                            judge_model=judge_model,
                            metadata=metadata,
                        )
                    )
                    continue
                if not criteria:
                    append_checkpoint_row(
                        args.out,
                        rows,
                        eval_failure_row(
                            pack=pack,
                            task=task,
                            mode=mode,
                            status="missing_writingbench_criteria",
                            solver_model=solver_model,
                            judge_model=judge_model,
                            metadata=metadata,
                        )
                    )
                    continue
                jobs.append(
                    WritingBenchEvalJob(
                        pack=pack,
                        task=task,
                        mode=mode,
                        examples=examples,
                        criteria=criteria,
                        skill_md=skill_md,
                        reused_row=reused_row,
                    )
                )

    print(f"Running {len(jobs)} WritingBench cells with num_threads={num_threads}", flush=True)
    run_eval_jobs(
        jobs,
        config=config,
        templates=templates,
        temperature=temperature,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        max_material_chars=args.max_material_chars,
        metadata=metadata,
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
