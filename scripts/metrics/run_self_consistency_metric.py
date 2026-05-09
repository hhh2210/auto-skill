#!/usr/bin/env python3
"""Run the eval-only skill encoding self-consistency diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, prompt_sha256, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.metrics import (  # noqa: E402
    build_abstract_signatures_prompt,
    build_signature_consistency_judge_prompt,
    literal_leakage_report,
    parse_abstract_signatures,
    parse_self_consistency_report,
    self_consistency_summary,
)
from auto_skill.mvp import PromptRunResult, user_examples_from_pack  # noqa: E402

SYSTEM_PROMPT = """You are evaluating skill encoding quality.
Use only user-visible examples and the generated skill. Never use private rubrics,
checklists, heldout feedback, official scores, or benchmark judge traces."""
DEFAULT_PARSE_MAX_ATTEMPTS = 3


def call_model(
    client: ChatCompletionClient,
    *,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
) -> PromptRunResult:
    result = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
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


def pack_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["pack_id"]): row
        for row in rows
        if row.get("pack_id") and user_examples_from_pack(row)
    }


def selected_skill_rows(
    rows: list[dict[str, Any]],
    *,
    pack_ids: set[str] | None,
    modes: set[str],
    limit_packs: int | None,
) -> list[dict[str, Any]]:
    selected = []
    seen_packs: set[str] = set()
    for row in rows:
        if row.get("status") != "success":
            continue
        pack_id = str(row.get("pack_id") or "")
        mode = str(row.get("mode") or "")
        if not pack_id or not mode or mode not in modes:
            continue
        if pack_ids and pack_id not in pack_ids:
            continue
        if limit_packs is not None and pack_id not in seen_packs and len(seen_packs) >= limit_packs:
            continue
        selected.append(row)
        seen_packs.add(pack_id)
    return selected


def dry_run_row(
    *,
    pack: dict[str, Any],
    skill_row: dict[str, Any],
    signature_prompt: str,
) -> dict[str, Any]:
    return {
        "schema_version": "skill-self-consistency-job/v1",
        "diagnostic_kind": "skill_encoding_self_consistency",
        "pack_id": skill_row["pack_id"],
        "mode": skill_row["mode"],
        "status": "dry_run",
        "signature_prompt_sha256": prompt_sha256(signature_prompt),
        "signature_prompt": signature_prompt,
        "train_example_ids": [
            example.example_id for example in user_examples_from_pack(pack)
        ],
        "uses_private_eval": False,
        "uses_official_scores": False,
    }


def failure_row(
    *,
    skill_row: dict[str, Any],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema_version": "skill-self-consistency/v1",
        "diagnostic_kind": "skill_encoding_self_consistency",
        "pack_id": skill_row.get("pack_id"),
        "mode": skill_row.get("mode"),
        "status": status,
        "abstract_signatures": [],
        "signature_generation": None,
        "literal_leakage": None,
        "judge": None,
        "judge_report": None,
        "uses_private_eval": False,
        "uses_official_scores": False,
    }
    if error:
        row["error"] = error
    return row


def evaluate_row(
    *,
    client: ChatCompletionClient,
    pack: dict[str, Any],
    skill_row: dict[str, Any],
    n: int,
    temperature: float | None,
    max_tokens: int,
    judge_max_tokens: int,
    parse_max_attempts: int,
    judge_model_id: str | None = None,
) -> dict[str, Any]:
    skill_md = skill_row.get("skill_md")
    if not isinstance(skill_md, str) or not skill_md.strip():
        return failure_row(skill_row=skill_row, status="missing_skill_md")

    signature_prompt = build_abstract_signatures_prompt(skill_md=skill_md, n=n)
    signature_generation, signatures, signature_parse_attempts = (
        generate_signatures_with_parse_retry(
            client=client,
            prompt=signature_prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            parse_max_attempts=parse_max_attempts,
        )
    )
    if signature_generation.finish_reason != "stop":
        return {
            **failure_row(skill_row=skill_row, status="signature_generation_incomplete"),
            "signature_generation": signature_generation.to_json(),
            "signature_parse_attempts": signature_parse_attempts,
        }

    if not signatures:
        return {
            **failure_row(skill_row=skill_row, status="signature_parse_error"),
            "signature_generation": signature_generation.to_json(),
            "signature_parse_attempts": signature_parse_attempts,
        }

    leakage = literal_leakage_report(pack=pack, generated_signatures=signatures)
    judge_prompt = build_signature_consistency_judge_prompt(
        train_pack=pack,
        abstract_signatures=signatures,
        literal_leakage=leakage,
    )
    judge, judge_report, status, judge_parse_attempts = judge_with_parse_retry(
        client=client,
        prompt=judge_prompt,
        max_tokens=judge_max_tokens,
        parse_max_attempts=parse_max_attempts,
    )

    return {
        "schema_version": "skill-self-consistency/v1",
        "diagnostic_kind": "skill_encoding_self_consistency",
        "pack_id": skill_row["pack_id"],
        "mode": skill_row["mode"],
        "status": status,
        "abstract_signatures": signatures,
        "signature_generation": signature_generation.to_json(),
        "signature_parse_attempts": signature_parse_attempts,
        "literal_leakage": leakage,
        "judge": judge.to_json(),
        "judge_parse_attempts": judge_parse_attempts,
        "judge_report": judge_report,
        "uses_private_eval": False,
        "uses_official_scores": False,
        "solver_model": skill_row.get("solver_model"),
        "judge_model": judge.model or judge_model_id,
    }


def generate_signatures_with_parse_retry(
    *,
    client: ChatCompletionClient,
    prompt: str,
    n: int,
    temperature: float | None,
    max_tokens: int,
    parse_max_attempts: int,
) -> tuple[PromptRunResult, list[dict[str, Any]], list[dict[str, Any]]]:
    attempts = max(1, parse_max_attempts)
    parse_attempts: list[dict[str, Any]] = []
    last_generation: PromptRunResult | None = None
    last_signatures: list[dict[str, Any]] = []

    for attempt in range(1, attempts + 1):
        generation = call_model(
            client,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        signatures = (
            parse_abstract_signatures(generation.text, limit=n)
            if generation.finish_reason == "stop"
            else []
        )
        status = "success" if signatures else "signature_parse_error"
        if generation.finish_reason != "stop":
            status = "signature_generation_incomplete"
        parse_attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "finish_reason": generation.finish_reason,
                "model": generation.model,
                "request_id": generation.request_id,
                "usage": generation.usage,
                "signature_count": len(signatures),
            }
        )
        last_generation = generation
        last_signatures = signatures
        if status != "signature_parse_error":
            break

    assert last_generation is not None
    return last_generation, last_signatures, parse_attempts


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
        judge = call_model(
            client,
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        report = parse_self_consistency_report(judge.text)
        status = "success"
        if judge.finish_reason != "stop":
            status = "judge_incomplete"
        elif "parse_error" in report:
            status = "judge_parse_error"
        parse_attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "finish_reason": judge.finish_reason,
                "model": judge.model,
                "request_id": judge.request_id,
                "usage": judge.usage,
                "parse_error": report.get("parse_error"),
                "overall_self_consistency": report.get("overall_self_consistency"),
            }
        )
        last_judge = judge
        last_report = report
        last_status = status
        if status != "judge_parse_error":
            break

    assert last_judge is not None
    return last_judge, last_report, last_status, parse_attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument("--skills", type=Path, default=Path("runs/skill_mvp.qwen.mvp.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/self_consistency.qwen.jsonl"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("runs/self_consistency.qwen.summary.json"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--judge-config-prefix",
        default=None,
        help=(
            "If set (e.g. 'JUDGE'), load the diagnostic-judge config from <PREFIX>_*. "
            "<PREFIX>_MODEL is required; URL/key fall back to BAILIAN_*."
        ),
    )
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument(
        "--modes",
        default="one_shot_skill_from_examples,auto_skill_feature_driven_no_validation",
        help="Comma-separated skill row modes to diagnose.",
    )
    parser.add_argument("--signatures-per-skill", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument(
        "--parse-max-attempts",
        type=int,
        default=DEFAULT_PARSE_MAX_ATTEMPTS,
        help=(
            "Retry signature/judge calls when the provider returns complete but "
            "unparseable JSON. Provider/network retries remain controlled by "
            "--max-retries."
        ),
    )
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.signatures_per_skill <= 0:
        print("error: --signatures-per-skill must be positive", file=sys.stderr)
        return 2
    if args.parse_max_attempts <= 0:
        print("error: --parse-max-attempts must be positive", file=sys.stderr)
        return 2

    packs = pack_index(load_jsonl(args.packs))
    modes = {mode.strip() for mode in args.modes.split(",") if mode.strip()}
    skill_rows = selected_skill_rows(
        load_jsonl(args.skills),
        pack_ids=set(args.pack_id) if args.pack_id else None,
        modes=modes,
        limit_packs=args.limit_packs,
    )
    skill_rows = [row for row in skill_rows if str(row.get("pack_id")) in packs]

    if args.dry_run:
        rows = []
        for row in skill_rows:
            prompt = build_abstract_signatures_prompt(
                skill_md=str(row.get("skill_md") or ""),
                n=args.signatures_per_skill,
            )
            rows.append(
                dry_run_row(
                    pack=packs[str(row["pack_id"])],
                    skill_row=row,
                    signature_prompt=prompt,
                )
            )
        write_jsonl(args.out, rows)
        print(f"Wrote {len(rows)} self-consistency dry-run jobs to {args.out}")
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
    print(
        f"diagnostic_kind=skill_encoding_self_consistency  judge_model={config.model}"
        + (f" (prefix={args.judge_config_prefix!r})" if args.judge_config_prefix else ""),
        flush=True,
    )
    if args.timeout_seconds is not None:
        config = replace(config, timeout_seconds=args.timeout_seconds)
    if args.max_retries is not None:
        config = replace(config, max_retries=args.max_retries)
    if args.stream is not None:
        config = replace(config, stream=args.stream)
    if args.enable_thinking is not None:
        config = replace(config, enable_thinking=args.enable_thinking)
    if args.thinking_budget is not None:
        config = replace(config, thinking_budget=args.thinking_budget)

    client = ChatCompletionClient(config)
    rows = []
    for row in skill_rows:
        try:
            result = evaluate_row(
                client=client,
                pack=packs[str(row["pack_id"])],
                skill_row=row,
                n=args.signatures_per_skill,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                judge_max_tokens=args.judge_max_tokens,
                parse_max_attempts=args.parse_max_attempts,
                judge_model_id=config.model,
            )
        except Exception as exc:  # noqa: BLE001 - keep long metric runs resumable by artifact.
            result = failure_row(
                skill_row=row,
                status="model_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        rows.append(result)
        write_jsonl(args.out, rows)
        print(f"  {row['pack_id']} {row['mode']}: {result['status']}", flush=True)
    write_jsonl(args.out, rows)
    summary = self_consistency_summary(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} self-consistency rows to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(row.get("status") == "success" for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
