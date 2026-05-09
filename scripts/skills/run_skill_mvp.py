#!/usr/bin/env python3
"""Run one-shot and feature-driven auto-skill MVP induction."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, prompt_sha256, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import (  # noqa: E402
    PromptRunResult,
    SkillInductionResult,
    build_leave_one_out_validation_prompt,
    build_one_shot_skill_prompt,
    build_validation_aware_skill_merge_prompt,
    parse_json_object,
    user_examples_from_pack,
)
from auto_skill.skill_induction import (  # noqa: E402
    build_cross_example_analysis_prompt,
    build_feature_extraction_prompt,
    build_skill_compilation_prompt,
)

SYSTEM_PROMPT = """You are an auto-skill research assistant.
Follow the project boundary exactly: skill induction may use only user-visible examples,
their visible outputs, visible materials, and explicit user emphasis.
Never use hidden benchmark rubrics, judge prompts, critique traces, or heldout feedback
as skill-induction input."""
DEFAULT_PARSE_MAX_ATTEMPTS = 3


class InductionError(RuntimeError):
    """Raised when a model stage cannot produce usable induction evidence."""


@dataclass(frozen=True)
class StageCallConfig:
    """Generation parameters that affect one resumable model stage."""

    model: str
    temperature: float | None
    max_tokens: int
    enable_thinking: bool | None
    thinking_budget: int | None
    stream: bool
    max_attempts: int = 1
    parse_max_attempts: int = DEFAULT_PARSE_MAX_ATTEMPTS
    retry_base_seconds: float = 2.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        data = self.to_json()
        data.pop("parse_max_attempts", None)
        return prompt_sha256(json.dumps(data, ensure_ascii=False, sort_keys=True))


def retryable_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code in {408, 409, 429} or status_code >= 500):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }


def call_model(
    client: ChatCompletionClient,
    prompt: str,
    *,
    config: StageCallConfig,
) -> PromptRunResult:
    attempts = max(1, config.max_attempts)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            result = client.complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                enable_thinking=config.enable_thinking,
                thinking_budget=config.thinking_budget,
            )
            duration_seconds = round(time.monotonic() - started, 3)
            return PromptRunResult(
                text=result.text,
                model=result.model,
                usage=result.usage,
                finish_reason=result.finish_reason,
                request_id=result.request_id,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed around provider errors.
            last_error = exc
            if attempt >= attempts or not retryable_provider_error(exc):
                break
            sleep_seconds = config.retry_base_seconds * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0, config.retry_base_seconds)  # noqa: S311
            print(
                f"  retryable provider error on attempt {attempt}/{attempts}: "
                f"{type(exc).__name__}; sleeping {sleep_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise InductionError(f"model call failed: {type(last_error).__name__}") from last_error


def require_complete_run(run: PromptRunResult, *, stage: str) -> None:
    if run.finish_reason != "stop":
        raise InductionError(f"{stage} did not finish cleanly: {run.finish_reason}")
    if not run.text.strip():
        raise InductionError(f"{stage} returned empty text")


def parse_required_json(text: str, *, stage: str) -> dict[str, Any]:
    parsed = parse_json_object(text)
    if "parse_error" in parsed:
        raise InductionError(f"{stage} returned invalid JSON: {parsed['parse_error']}")
    return parsed


class StageLedger:
    """Append-only stage cache for long skill-induction runs."""

    def __init__(self, path: Path, *, resume: bool):
        self.path = path
        self._success_rows: dict[tuple[str, ...], dict[str, Any]] = {}
        if resume and path.exists():
            for row in load_jsonl(path):
                if row.get("status") == "success":
                    self._success_rows[self._row_key(row)] = row
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    @staticmethod
    def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(row.get("pack_id") or ""),
            str(row.get("stage") or ""),
            str(row.get("example_id") or ""),
            str(row.get("validation_example_id") or ""),
            str(row.get("prompt_sha256") or ""),
            str(row.get("config_sha256") or ""),
        )

    def success(
        self,
        *,
        pack_id: str,
        stage: str,
        prompt_hash: str,
        config_hash: str,
        example_id: str | None = None,
        validation_example_id: str | None = None,
    ) -> dict[str, Any] | None:
        key = (
            pack_id,
            stage,
            example_id or "",
            validation_example_id or "",
            prompt_hash,
            config_hash,
        )
        return self._success_rows.get(key)

    def append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row.get("status") == "success":
            self._success_rows[self._row_key(row)] = row


def ledger_row_base(
    *,
    pack_id: str,
    stage: str,
    prompt_hash: str,
    call_config: StageCallConfig,
    example_id: str | None = None,
    validation_example_id: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": "skill-induction-stage/v1",
        "pack_id": pack_id,
        "stage": stage,
        "example_id": example_id,
        "validation_example_id": validation_example_id,
        "prompt_sha256": prompt_hash,
        "config_sha256": call_config.stable_hash(),
        "config": call_config.to_json(),
    }
    return {key: value for key, value in row.items() if value is not None}


def run_stage(
    *,
    client: ChatCompletionClient,
    ledger: StageLedger,
    pack_id: str,
    stage: str,
    prompt: str,
    call_config: StageCallConfig,
    parse_json: bool,
    example_id: str | None = None,
    validation_example_id: str | None = None,
) -> tuple[PromptRunResult, dict[str, Any] | None]:
    prompt_hash = prompt_sha256(prompt)
    config_hash = call_config.stable_hash()
    cached = ledger.success(
        pack_id=pack_id,
        stage=stage,
        example_id=example_id,
        validation_example_id=validation_example_id,
        prompt_hash=prompt_hash,
        config_hash=config_hash,
    )
    if cached is not None:
        return PromptRunResult(**cached["run"]), cached.get("parsed_json")

    base = ledger_row_base(
        pack_id=pack_id,
        stage=stage,
        example_id=example_id,
        validation_example_id=validation_example_id,
        prompt_hash=prompt_hash,
        call_config=call_config,
    )
    run: PromptRunResult | None = None
    parsed: dict[str, Any] | None = None
    parse_attempts = max(1, call_config.parse_max_attempts) if parse_json else 1
    parse_attempt = 0
    try:
        for parse_attempt in range(1, parse_attempts + 1):
            run = call_model(client, prompt, config=call_config)
            require_complete_run(run, stage=stage)
            if not parse_json:
                break
            parsed_candidate = parse_json_object(run.text)
            if "parse_error" not in parsed_candidate:
                parsed = parsed_candidate
                break
            if parse_attempt >= parse_attempts:
                raise InductionError(
                    f"{stage} returned invalid JSON after {parse_attempts} attempt(s): "
                    f"{parsed_candidate['parse_error']}"
                )
            print(
                f"  parse error on {stage} attempt {parse_attempt}/{parse_attempts}: "
                f"{parsed_candidate['parse_error']}; retrying stage",
                file=sys.stderr,
                flush=True,
            )
    except InductionError as exc:
        failure = {
            **base,
            "status": "stage_error",
            "run": run.to_json() if run is not None else None,
            "parse_attempts": parse_attempt,
            "error": str(exc),
        }
        ledger.append(failure)
        raise

    ledger.append(
        {
            **base,
            "status": "success",
            "run": run.to_json(),
            "parsed_json": parsed,
            "parse_attempts": parse_attempt,
        }
    )
    return run, parsed


def induce_one_shot(
    pack: dict[str, Any],
    client: ChatCompletionClient,
    *,
    ledger: StageLedger,
    stage_config: StageCallConfig,
) -> SkillInductionResult:
    examples = user_examples_from_pack(pack)
    prompt = build_one_shot_skill_prompt(examples)
    run, _ = run_stage(
        client=client,
        ledger=ledger,
        pack_id=str(pack["pack_id"]),
        stage="one_shot_skill",
        prompt=prompt,
        call_config=stage_config,
        parse_json=False,
    )
    return SkillInductionResult(
        pack_id=str(pack["pack_id"]),
        mode="one_shot_skill_from_examples",
        skill_md=run.text,
        model_calls=[{"stage": "one_shot_skill", **run.to_json()}],
    )


def induce_feature_driven(
    pack: dict[str, Any],
    client: ChatCompletionClient,
    *,
    ledger: StageLedger,
    stage_config_for: Any,
    leave_one_out: bool,
) -> list[SkillInductionResult]:
    examples = user_examples_from_pack(pack)
    model_calls = []
    feature_reports = []
    for example in examples:
        stage = "feature_extraction"
        prompt = build_feature_extraction_prompt(example)
        run, parsed = run_stage(
            client=client,
            ledger=ledger,
            pack_id=str(pack["pack_id"]),
            stage=stage,
            example_id=example.example_id,
            prompt=prompt,
            call_config=stage_config_for(stage),
            parse_json=True,
        )
        assert parsed is not None
        report = parsed
        report["example_id"] = example.example_id
        feature_reports.append(report)
        model_calls.append(
            {
                "stage": stage,
                "example_id": example.example_id,
                **run.to_json(),
            }
        )

    stage = "cross_example_analysis"
    cross_prompt = build_cross_example_analysis_prompt(
        [json.dumps(report, ensure_ascii=False) for report in feature_reports]
    )
    cross_run, parsed_cross = run_stage(
        client=client,
        ledger=ledger,
        pack_id=str(pack["pack_id"]),
        stage=stage,
        prompt=cross_prompt,
        call_config=stage_config_for(stage),
        parse_json=True,
    )
    assert parsed_cross is not None
    cross_report = parsed_cross
    model_calls.append({"stage": stage, **cross_run.to_json()})

    stage = "skill_compilation"
    skill_prompt = build_skill_compilation_prompt(
        json.dumps(cross_report, ensure_ascii=False, indent=2),
        examples,
    )
    skill_run, _ = run_stage(
        client=client,
        ledger=ledger,
        pack_id=str(pack["pack_id"]),
        stage=stage,
        prompt=skill_prompt,
        call_config=stage_config_for(stage),
        parse_json=False,
    )
    model_calls.append({"stage": stage, **skill_run.to_json()})

    no_validation_result = SkillInductionResult(
        pack_id=str(pack["pack_id"]),
        mode="auto_skill_feature_driven_no_validation",
        skill_md=skill_run.text,
        feature_reports=feature_reports,
        cross_example_report=cross_report,
        model_calls=list(model_calls),
    )

    loo_reports = []
    loo_candidate_skills = []
    if leave_one_out and len(examples) >= 3:
        for validation_example in examples:
            training_reports = [
                report
                for report in feature_reports
                if report.get("example_id") != validation_example.example_id
            ]
            stage = "leave_one_out_cross_example_analysis"
            loo_cross_prompt = build_cross_example_analysis_prompt(
                [json.dumps(report, ensure_ascii=False) for report in training_reports]
            )
            loo_cross_run, parsed_loo_cross = run_stage(
                client=client,
                ledger=ledger,
                pack_id=str(pack["pack_id"]),
                stage=stage,
                validation_example_id=validation_example.example_id,
                prompt=loo_cross_prompt,
                call_config=stage_config_for(stage),
                parse_json=True,
            )
            assert parsed_loo_cross is not None
            loo_cross_report = parsed_loo_cross
            model_calls.append(
                {
                    "stage": stage,
                    "validation_example_id": validation_example.example_id,
                    **loo_cross_run.to_json(),
                }
            )

            stage = "leave_one_out_skill_compilation"
            loo_skill_prompt = build_skill_compilation_prompt(
                json.dumps(loo_cross_report, ensure_ascii=False, indent=2),
                [
                    example
                    for example in examples
                    if example.example_id != validation_example.example_id
                ],
            )
            loo_skill_run, _ = run_stage(
                client=client,
                ledger=ledger,
                pack_id=str(pack["pack_id"]),
                stage=stage,
                validation_example_id=validation_example.example_id,
                prompt=loo_skill_prompt,
                call_config=stage_config_for(stage),
                parse_json=False,
            )
            loo_candidate_skills.append(
                {
                    "validation_example_id": validation_example.example_id,
                    "training_example_ids": [
                        example.example_id
                        for example in examples
                        if example.example_id != validation_example.example_id
                    ],
                    "skill_md": loo_skill_run.text,
                }
            )
            model_calls.append(
                {
                    "stage": stage,
                    "validation_example_id": validation_example.example_id,
                    **loo_skill_run.to_json(),
                }
            )

            stage = "leave_one_out_validation"
            validation_prompt = build_leave_one_out_validation_prompt(
                candidate_skill_md=loo_skill_run.text,
                validation_example=validation_example,
            )
            validation_run, parsed_validation = run_stage(
                client=client,
                ledger=ledger,
                pack_id=str(pack["pack_id"]),
                stage=stage,
                validation_example_id=validation_example.example_id,
                prompt=validation_prompt,
                call_config=stage_config_for(stage),
                parse_json=True,
            )
            assert parsed_validation is not None
            validation_report = parsed_validation
            validation_report["validation_example_id"] = validation_example.example_id
            loo_reports.append(validation_report)
            model_calls.append(
                {
                    "stage": stage,
                    "validation_example_id": validation_example.example_id,
                    **validation_run.to_json(),
                }
            )

    if not loo_reports:
        return [no_validation_result]

    stage = "validation_aware_skill_merge"
    merge_prompt = build_validation_aware_skill_merge_prompt(
        candidate_skills=loo_candidate_skills,
        leave_one_out_reports=loo_reports,
    )
    merge_run, _ = run_stage(
        client=client,
        ledger=ledger,
        pack_id=str(pack["pack_id"]),
        stage=stage,
        prompt=merge_prompt,
        call_config=stage_config_for(stage),
        parse_json=False,
    )
    model_calls.append({"stage": stage, **merge_run.to_json()})

    full_result = SkillInductionResult(
        pack_id=str(pack["pack_id"]),
        mode="auto_skill_ours_full",
        skill_md=merge_run.text,
        feature_reports=feature_reports,
        cross_example_report=cross_report,
        leave_one_out=loo_reports,
        model_calls=model_calls,
    )
    return [no_validation_result, full_result]


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
        if len(user_examples_from_pack(pack)) < 2:
            continue
        selected.append(pack)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def requested_output_modes(
    pack: dict[str, Any],
    requested_modes: set[str],
    *,
    leave_one_out: bool,
) -> set[str]:
    output_modes = set()
    if "one_shot_skill_from_examples" in requested_modes:
        output_modes.add("one_shot_skill_from_examples")
    if "auto_skill_feature_driven" in requested_modes:
        output_modes.add("auto_skill_feature_driven_no_validation")
        if leave_one_out and len(user_examples_from_pack(pack)) >= 3:
            output_modes.add("auto_skill_ours_full")
    return output_modes


def existing_success_modes(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_pack: dict[str, set[str]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        pack_id = row.get("pack_id")
        mode = row.get("mode")
        skill_md = row.get("skill_md")
        if isinstance(pack_id, str) and isinstance(mode, str) and isinstance(skill_md, str):
            if skill_md.strip():
                by_pack.setdefault(pack_id, set()).add(mode)
    return by_pack


def checkpoint_rows(
    out: Path,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    write_jsonl(out, rows + failures)


JSON_STAGES = {
    "feature_extraction",
    "cross_example_analysis",
    "leave_one_out_cross_example_analysis",
    "leave_one_out_validation",
}


def parse_stage_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def default_stage_ledger_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.stages.jsonl")


def stage_max_tokens(stage: str, args: argparse.Namespace) -> int:
    if stage == "feature_extraction":
        return args.feature_max_tokens or min(args.max_tokens, 4096)
    if stage in {"cross_example_analysis", "leave_one_out_cross_example_analysis"}:
        return args.analysis_max_tokens or args.max_tokens
    if stage == "leave_one_out_validation":
        return args.validation_max_tokens or min(args.max_tokens, 4096)
    return args.skill_max_tokens or args.max_tokens


def build_stage_config_factory(
    *,
    client_config: ChatCompletionConfig,
    args: argparse.Namespace,
    default_temperature: float | None,
) -> Any:
    thinking_stages = parse_stage_set(args.thinking_stages)

    def stage_config(stage: str) -> StageCallConfig:
        if thinking_stages is None:
            enable_thinking = client_config.enable_thinking
        else:
            enable_thinking = stage in thinking_stages
        thinking_budget = None
        if enable_thinking:
            thinking_budget = args.thinking_budget or client_config.thinking_budget or 4096
        temperature = default_temperature
        if stage in JSON_STAGES and args.json_temperature is not None:
            temperature = args.json_temperature
        return StageCallConfig(
            model=client_config.model,
            temperature=temperature,
            max_tokens=stage_max_tokens(stage, args),
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            stream=client_config.stream,
            max_attempts=args.stage_max_attempts,
            parse_max_attempts=args.parse_max_attempts,
            retry_base_seconds=args.retry_base_seconds,
        )

    return stage_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/skill_mvp.qwen.jsonl"))
    parser.add_argument(
        "--stage-ledger",
        type=Path,
        help="Append-only per-stage checkpoint JSONL. Defaults to <out stem>.stages.jsonl.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--source", choices=["WritingBench", "PresentBench"])
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--json-temperature",
        type=float,
        default=None,
        help="Optional temperature override for JSON-producing stages.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--feature-max-tokens", type=int)
    parser.add_argument("--analysis-max-tokens", type=int)
    parser.add_argument("--skill-max-tokens", type=int)
    parser.add_argument("--validation-max-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument(
        "--thinking-stages",
        help=(
            "Comma-separated stage names that should enable thinking. "
            "When set, all other stages force thinking off."
        ),
    )
    parser.add_argument("--stage-max-attempts", type=int, default=1)
    parser.add_argument(
        "--parse-max-attempts",
        type=int,
        default=DEFAULT_PARSE_MAX_ATTEMPTS,
        help=(
            "Retry JSON-producing stages when the provider returns complete but "
            "unparseable JSON. Provider/network retries remain controlled by "
            "--stage-max-attempts and --max-retries."
        ),
    )
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming chat completions.")
    parser.add_argument(
        "--stream-log",
        action="store_true",
        help="Print streamed model deltas to stderr. Useful only for sequential debugging.",
    )
    parser.add_argument("--leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--modes",
        default="one_shot_skill_from_examples,auto_skill_feature_driven",
        help="Comma-separated modes to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful rows already present in --out and retry only missing modes.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Write successful skill rows even if some selected packs fail. "
            "Default is fail-closed."
        ),
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 when filters select no packs. Default is fail-closed.",
    )
    args = parser.parse_args()

    packs = load_jsonl(args.packs)
    selected = select_packs(
        packs,
        pack_ids=set(args.pack_id) if args.pack_id else None,
        source=args.source,
        limit=args.limit_packs,
    )
    modes = {mode.strip() for mode in args.modes.split(",") if mode.strip()}
    if args.stage_max_attempts <= 0:
        print("error: --stage-max-attempts must be positive", file=sys.stderr)
        return 2
    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2
    if args.retry_base_seconds <= 0:
        print("error: --retry-base-seconds must be positive", file=sys.stderr)
        return 2

    if not selected and not args.allow_empty:
        print("error: no packs selected for skill induction", file=sys.stderr)
        return 3
    if not selected and args.allow_empty and not args.dry_run:
        write_jsonl(args.out, [])
        print(f"Wrote 0 skill rows to {args.out}")
        return 0

    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected_pack_ids": [pack["pack_id"] for pack in selected],
                    "modes": sorted(modes),
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
    temperature = args.temperature if args.temperature is not None else config.temperature
    client = ChatCompletionClient(config)
    stage_ledger_path = args.stage_ledger or default_stage_ledger_path(args.out)
    stage_ledger = StageLedger(stage_ledger_path, resume=args.resume)
    stage_config_for = build_stage_config_factory(
        client_config=config,
        args=args,
        default_temperature=temperature,
    )

    if args.resume and args.out.exists():
        rows = [
            row
            for row in load_jsonl(args.out)
            if row.get("status") == "success" and row.get("skill_md")
        ]
        print(f"Loaded {len(rows)} successful existing rows from {args.out}", flush=True)
    else:
        rows = []
    existing_modes = existing_success_modes(rows)
    failures = []
    for index, pack in enumerate(selected, start=1):
        pack_id = str(pack["pack_id"])
        missing_modes = requested_output_modes(
            pack,
            modes,
            leave_one_out=args.leave_one_out,
        ) - existing_modes.get(pack_id, set())
        if not missing_modes:
            print(f"[{index}/{len(selected)}] skipping {pack_id} (already complete)", flush=True)
            continue
        print(
            f"[{index}/{len(selected)}] inducing skills for {pack_id} "
            f"(missing={sorted(missing_modes)})",
            flush=True,
        )
        try:
            if "one_shot_skill_from_examples" in missing_modes:
                rows.append(
                    induce_one_shot(
                        pack,
                        client,
                        ledger=stage_ledger,
                        stage_config=stage_config_for("one_shot_skill"),
                    ).to_json()
                )
                existing_modes.setdefault(pack_id, set()).add("one_shot_skill_from_examples")
                checkpoint_rows(args.out, rows, failures)
            feature_modes = {
                "auto_skill_feature_driven_no_validation",
                "auto_skill_ours_full",
            }
            if missing_modes & feature_modes:
                feature_results = induce_feature_driven(
                    pack,
                    client,
                    ledger=stage_ledger,
                    stage_config_for=stage_config_for,
                    leave_one_out=args.leave_one_out,
                )
                rows.extend(result.to_json() for result in feature_results)
                existing_modes[pack_id] = existing_modes.get(pack_id, set()) | {
                    result.mode for result in feature_results
                }
                checkpoint_rows(args.out, rows, failures)
        except InductionError as exc:
            failure = {
                "schema_version": "skill-induction/v1",
                "pack_id": pack["pack_id"],
                "mode": "pack_induction",
                "status": "induction_error",
                "skill_md": None,
                "feature_reports": [],
                "cross_example_report": None,
                "leave_one_out": [],
                "model_calls": [],
                "solver_model": None,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"  induction_error: {failure['error']}", file=sys.stderr, flush=True)
            checkpoint_rows(args.out, rows, failures)
            if not args.allow_partial:
                print(
                    f"Wrote {len(rows)} successful rows and {len(failures)} failures to {args.out}"
                )
                return 3

    checkpoint_rows(args.out, rows, failures)
    print(f"Wrote {len(rows)} skill rows to {args.out}")
    if failures and not args.allow_partial:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
