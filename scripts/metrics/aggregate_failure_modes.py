#!/usr/bin/env python3
"""Aggregate judge-reason text into operational failure-mode categories.

This script performs exactly one chat-completion call unless ``--dry-run`` is
set. It sends already-evaluated MIMO judge rationale text; it does not generate
new benchmark outputs.
"""

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

from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import parse_json_object  # noqa: E402

SYSTEM_PROMPT = """You analyze benchmark judge rationales for an auto-skill research prototype.
Return only one strict JSON object. Do not wrap it in Markdown fences."""


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars].rstrip() + f" [truncated {omitted} chars]"


def build_reason_corpus(
    rows: list[dict[str, Any]], *, max_reason_chars: int
) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    for row in rows:
        criteria = []
        for item in row.get("criteria", []):
            if not isinstance(item, dict):
                continue
            baseline = item.get("baseline") if isinstance(item.get("baseline"), dict) else {}
            candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
            criteria.append(
                {
                    "criterion_name": item.get("criterion_name"),
                    "delta": item.get("delta"),
                    "baseline_score": baseline.get("score"),
                    "baseline_reason": truncate_text(
                        str(baseline.get("reason") or ""), max_reason_chars
                    ),
                    "candidate_score": candidate.get("score"),
                    "candidate_reason": truncate_text(
                        str(candidate.get("reason") or ""), max_reason_chars
                    ),
                }
            )
        corpus.append(
            {
                "task_id": row.get("task_id"),
                "pack_id": row.get("pack_id"),
                "overall_delta": row.get("overall_delta"),
                "worst_criterion": row.get("worst_criterion"),
                "criteria": criteria,
            }
        )
    return corpus


def build_prompt(corpus: list[dict[str, Any]]) -> str:
    return (
        "You are given 30 WritingBench tasks where the auto-skill candidate "
        "`ours_no_validation` scored lower than the baseline "
        "`few_shot_examples_only` under a MIMO judge. The corpus contains paired "
        "per-criterion scores and reason strings for both modes.\n\n"
        "Your job is to explain WHY ours scored lower than example_only. Do not "
        "summarize absolute quality. Compare candidate reasons against baseline "
        "reasons and focus on recurring deficits that are operationalizable in a "
        "future induction prompt. Penalize generic labels such as quality, clarity, "
        "relevance, detail, or low_quality unless the label names a concrete "
        "extractable pattern.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "categories": [\n'
        "    {\n"
        '      "category_name": "snake_case_specific_name",\n'
        '      "definition": "one sentence",\n'
        '      "task_count": 0,\n'
        '      "task_ids": ["..."],\n'
        '      "quotes": [\n'
        '        {"task_id": "...", "quote": "verbatim judge reason excerpt, <=30 words"}\n'
        "      ],\n"
        '      "fix_recommendation": "one concrete prompt-change recommendation"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Constraints:\n"
        "- Produce 6 to 8 categories total, sorted by task_count descending.\n"
        "- A task can belong to multiple categories, but task_count must count unique task_ids.\n"
        "- Each category needs exactly 3 short verbatim quotes from judge reasons.\n"
        "- Quotes must be <= 30 words and cite task_id.\n"
        "- Do not invent task_ids, criteria, or quotes.\n\n"
        "Corpus JSON:\n"
        + json.dumps(corpus, ensure_ascii=False)
    )


def validate_failure_modes(parsed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    categories = parsed.get("categories")
    if not isinstance(categories, list):
        return ["categories_must_be_list"]
    if not 6 <= len(categories) <= 8:
        errors.append("categories_must_have_6_to_8_items")
    for idx, category in enumerate(categories):
        if not isinstance(category, dict):
            errors.append(f"category_{idx}_must_be_object")
            continue
        name = category.get("category_name")
        if not isinstance(name, str) or not name or name.lower() != name or " " in name:
            errors.append(f"category_{idx}_category_name_must_be_snake_case_string")
        if not isinstance(category.get("definition"), str) or not category["definition"].strip():
            errors.append(f"category_{idx}_definition_must_be_string")
        if not isinstance(category.get("task_count"), int):
            errors.append(f"category_{idx}_task_count_must_be_integer")
        if not isinstance(category.get("task_ids"), list):
            errors.append(f"category_{idx}_task_ids_must_be_list")
        quotes = category.get("quotes")
        if not isinstance(quotes, list) or len(quotes) != 3:
            errors.append(f"category_{idx}_quotes_must_have_3_items")
            continue
        for q_idx, quote in enumerate(quotes):
            if not isinstance(quote, dict):
                errors.append(f"category_{idx}_quote_{q_idx}_must_be_object")
                continue
            if not isinstance(quote.get("task_id"), str) or not quote["task_id"].strip():
                errors.append(f"category_{idx}_quote_{q_idx}_task_id_must_be_string")
            text = quote.get("quote")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"category_{idx}_quote_{q_idx}_quote_must_be_string")
            elif len(text.split()) > 30:
                errors.append(f"category_{idx}_quote_{q_idx}_quote_over_30_words")
        if not isinstance(category.get("fix_recommendation"), str) or not category[
            "fix_recommendation"
        ].strip():
            errors.append(f"category_{idx}_fix_recommendation_must_be_string")
    return errors


def provider_label(config: ChatCompletionConfig, prefix: str | None) -> str:
    if prefix:
        return f"{prefix}_MODEL={config.model}; base_url={config.base_url}"
    return f"BAILIAN/OPENAI default model={config.model}; base_url={config.base_url}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--judge-config-prefix", default="JUDGE")
    parser.add_argument("--model", default="mimo-v2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-reason-chars", type=int, default=1500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    corpus = build_reason_corpus(rows, max_reason_chars=args.max_reason_chars)
    prompt = build_prompt(corpus)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": len(corpus),
                    "prompt_chars": len(prompt),
                    "max_reason_chars": args.max_reason_chars,
                    "input": str(args.input),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config: ChatCompletionConfig | None = None
    config_source = f"prefix:{args.judge_config_prefix}"
    try:
        config = ChatCompletionConfig.from_env(args.env_file, prefix=args.judge_config_prefix)
    except ConfigError as prefixed_exc:
        if args.judge_config_prefix:
            try:
                config = ChatCompletionConfig.from_env(args.env_file)
                config_source = "default:BAILIAN/OPENAI"
                print(
                    (
                        f"warning: {prefixed_exc}; falling back to "
                        "BAILIAN/OPENAI default config"
                    ),
                    file=sys.stderr,
                )
            except ConfigError as default_exc:
                print(
                    (
                        "configuration error: could not load JUDGE_* or default "
                        f"BAILIAN/OPENAI config: {default_exc}"
                    ),
                    file=sys.stderr,
                )
                return 2
        else:
            print(f"configuration error: {prefixed_exc}", file=sys.stderr)
            return 2
    config = replace(
        config,
        model=args.model,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        stream=False,
    )
    client = ChatCompletionClient(config)
    try:
        result = client.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=False,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should persist provider failures.
        payload = {
            "schema_version": "failure-mode-aggregation/v1",
            "input": str(args.input),
            "task_count": len(corpus),
            "provider": {
                "config_source": config_source,
                "label": provider_label(
                    config,
                    args.judge_config_prefix
                    if config_source.startswith("prefix:")
                    else None,
                ),
                "model": config.model,
                "base_url": config.base_url,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_reason_chars": args.max_reason_chars,
            },
            "request": {"one_llm_call": True},
            "raw_response": "",
            "parsed": {"raw_text": "", "parse_error": "provider_exception"},
            "validation_errors": [f"{type(exc).__name__}: {exc}"],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}", file=sys.stderr)
        print(f"provider error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    parsed = parse_json_object(result.text)
    validation_errors = (
        [str(parsed.get("parse_error"))]
        if "parse_error" in parsed
        else validate_failure_modes(parsed)
    )
    payload = {
        "schema_version": "failure-mode-aggregation/v1",
        "input": str(args.input),
        "task_count": len(corpus),
        "provider": {
            "config_source": config_source,
            "label": provider_label(
                config,
                args.judge_config_prefix if config_source.startswith("prefix:") else None,
            ),
            "model": config.model,
            "base_url": config.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_reason_chars": args.max_reason_chars,
        },
        "request": {
            "one_llm_call": True,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
            "request_id": result.request_id,
        },
        "raw_response": result.text,
        "parsed": parsed,
        "validation_errors": validation_errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        json.dumps(
            {
                "provider": payload["provider"],
                "task_count": len(corpus),
                "validation_errors": validation_errors,
                "category_count": len(parsed.get("categories", []))
                if isinstance(parsed.get("categories"), list)
                else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not validation_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
