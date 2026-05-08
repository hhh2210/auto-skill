#!/usr/bin/env python3
"""Audit generated train-example outputs with private benchmark criteria."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.eval_summary import summarize_score_rows  # noqa: E402
from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.writingbench_eval import (  # noqa: E402
    average_writingbench_scores,
    load_writingbench_prompt_templates,
)
from scripts.eval.run_heldout_eval import select_packs  # noqa: E402
from scripts.eval.run_writingbench_official_eval import (  # noqa: E402
    score_with_writingbench_prompt,
)

AUDIT_SCHEMA_VERSION = "train-example-quality-audit/v1"
AUDIT_MODE = "desired_output"
AUDIT_KIND = "private_train_example_quality_audit"


def train_private_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for pack in rows:
        pack_id = str(pack.get("pack_id"))
        for item in pack.get("train_private", []):
            index[(pack_id, str(item.get("task_ref")))] = item
    return index


def desired_output_text(example: dict[str, Any]) -> tuple[str | None, str | None]:
    desired_output = example.get("desired_output")
    if not isinstance(desired_output, dict):
        return None, "missing_desired_output"
    if desired_output.get("status") != "generated":
        return None, "desired_output_not_generated"
    text = desired_output.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "empty_desired_output"
    return text, None


def audit_failure_row(
    *,
    pack: dict[str, Any],
    example: dict[str, Any],
    status: str,
    judge_model: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "pack_id": str(pack["pack_id"]),
        "task_id": str(example.get("example_id") or ""),
        "example_id": str(example.get("example_id") or ""),
        "source": pack.get("source"),
        "source_task_id": example.get("source_task_id"),
        "mode": AUDIT_MODE,
        "evaluator_kind": AUDIT_KIND,
        "status": status,
        "scores": {},
        "overall_score": None,
        "judge_calls": [],
        "judge_model": judge_model,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_score_rows(
        rows,
        evaluator_kind=AUDIT_KIND,
        score_summary_key="mean_train_example_quality_score",
        baseline_mode=AUDIT_MODE,
    )
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row.get("source") or "unknown")
        status = str(row.get("status") or "unknown")
        source_counts = by_source.setdefault(source, {})
        source_counts[status] = source_counts.get(status, 0) + 1
    summary["status_counts_by_source"] = {
        source: dict(sorted(counts.items())) for source, counts in sorted(by_source.items())
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument(
        "--private-eval",
        type=Path,
        default=Path("artifacts/private/example_private_eval.jsonl"),
    )
    parser.add_argument("--writingbench-root", type=Path, default=Path("../WritingBench"))
    parser.add_argument("--out", type=Path, default=Path("runs/train_example_quality_audit.jsonl"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("runs/train_example_quality_audit.summary.json"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--judge-config-prefix",
        default="MIMO",
        help="Judge config prefix, e.g. MIMO or JUDGE. Defaults to MIMO.",
    )
    parser.add_argument("--source", choices=["WritingBench", "PresentBench"])
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    packs = load_jsonl(args.packs)
    if args.source:
        packs = [pack for pack in packs if pack.get("source") == args.source]
    selected = select_packs(
        packs,
        pack_ids=set(args.pack_id) if args.pack_id else None,
        source=None,
        limit=args.limit_packs,
    )
    total_examples = sum(len(pack.get("train_examples", [])) for pack in selected)
    if args.limit_examples is not None:
        total_examples = min(total_examples, args.limit_examples)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected_pack_ids": [pack["pack_id"] for pack in selected],
                    "train_examples": total_examples,
                    "judge_config_prefix": args.judge_config_prefix,
                    "audit_kind": AUDIT_KIND,
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        judge_config = ChatCompletionConfig.from_env(
            args.env_file, prefix=args.judge_config_prefix
        )
        templates = load_writingbench_prompt_templates(args.writingbench_root)
    except (ConfigError, OSError, ImportError, AttributeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.timeout_seconds is not None:
        judge_config = replace(judge_config, timeout_seconds=args.timeout_seconds)
    if args.max_retries is not None:
        judge_config = replace(judge_config, max_retries=args.max_retries)

    judge_client = ChatCompletionClient(judge_config)
    private_index = train_private_index(load_jsonl(args.private_eval))
    rows: list[dict[str, Any]] = []
    example_budget = args.limit_examples

    print(
        f"train example audit judge_model={judge_config.model} "
        f"prefix={args.judge_config_prefix!r}",
        flush=True,
    )
    for pack_index, pack in enumerate(selected, start=1):
        pack_id = str(pack["pack_id"])
        examples = pack.get("train_examples", [])
        print(f"[{pack_index}/{len(selected)}] auditing {pack_id}", flush=True)
        for example in examples:
            if example_budget is not None and example_budget <= 0:
                break
            example_id = str(example.get("example_id") or "")
            text, missing_status = desired_output_text(example)
            if missing_status is not None:
                rows.append(
                    audit_failure_row(
                        pack=pack,
                        example=example,
                        status=missing_status,
                        judge_model=judge_config.model,
                    )
                )
                print(f"  {example_id}: {missing_status}", flush=True)
                if example_budget is not None:
                    example_budget -= 1
                continue
            private_eval = private_index.get((pack_id, example_id))
            if not private_eval:
                rows.append(
                    audit_failure_row(
                        pack=pack,
                        example=example,
                        status="missing_private_eval",
                        judge_model=judge_config.model,
                    )
                )
                print(f"  {example_id}: missing_private_eval", flush=True)
                if example_budget is not None:
                    example_budget -= 1
                continue
            if pack.get("source") != "WritingBench":
                rows.append(
                    audit_failure_row(
                        pack=pack,
                        example=example,
                        status="unsupported_source_for_writingbench_audit",
                        judge_model=judge_config.model,
                    )
                )
                print(f"  {example_id}: unsupported_source_for_writingbench_audit", flush=True)
                if example_budget is not None:
                    example_budget -= 1
                continue
            supervision = private_eval.get("supervision") or {}
            raw_items = supervision.get("items") or supervision.get("criteria") or []
            criteria = [item for item in raw_items if isinstance(item, dict)]
            if not criteria:
                rows.append(
                    audit_failure_row(
                        pack=pack,
                        example=example,
                        status="missing_writingbench_criteria",
                        judge_model=judge_config.model,
                    )
                )
                print(f"  {example_id}: missing_writingbench_criteria", flush=True)
                if example_budget is not None:
                    example_budget -= 1
                continue

            try:
                status, scores, judge_calls = score_with_writingbench_prompt(
                    client=judge_client,
                    system_prompt=templates.evaluate_system,
                    prompt_template=templates.evaluate_prompt,
                    query=str(example.get("task_input") or ""),
                    response=text or "",
                    criteria=criteria,
                    max_tokens=args.judge_max_tokens,
                    judge_client=judge_client,
                )
            except Exception as exc:  # noqa: BLE001 - audit should record flaky API calls.
                rows.append(
                    audit_failure_row(
                        pack=pack,
                        example=example,
                        status="model_error",
                        judge_model=judge_config.model,
                    )
                    | {
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
                print(f"  {example_id}: model_error ({type(exc).__name__})", flush=True)
                if example_budget is not None:
                    example_budget -= 1
                continue
            overall_score = average_writingbench_scores(scores) if status == "success" else None
            first_judge_model = next(
                (call.get("model") for call in judge_calls if call.get("model")),
                judge_config.model,
            )
            rows.append(
                {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "pack_id": pack_id,
                    "task_id": example_id,
                    "example_id": example_id,
                    "source": pack.get("source"),
                    "source_task_id": example.get("source_task_id"),
                    "mode": AUDIT_MODE,
                    "evaluator_kind": AUDIT_KIND,
                    "official_prompt_file": templates.prompt_file,
                    "status": status,
                    "scores": scores,
                    "overall_score": overall_score,
                    "judge_calls": judge_calls,
                    "judge_model": first_judge_model,
                }
            )
            print(f"  {example_id}: {overall_score} ({status})", flush=True)
            if example_budget is not None:
                example_budget -= 1
        if example_budget is not None and example_budget <= 0:
            break

    write_jsonl(args.out, rows)
    summary = summarize_rows(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} audit rows to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(row.get("status") != "success" for row in rows) and not args.allow_partial:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
