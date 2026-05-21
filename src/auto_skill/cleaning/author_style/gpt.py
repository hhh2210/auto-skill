"""GPT/Codex-backed audit and hard-negative rerank helpers."""

from __future__ import annotations

import argparse
import time
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from auto_skill.cleaning.author_style.gpt_filters import (
    audit_passes_thresholds,
    audit_threshold_policy,
    filter_by_pack_ids,
    filter_negatives_by_pack_ids,
    pack_ids_with_negative_coverage,
)
from auto_skill.cleaning.author_style.gpt_prompting import (
    build_gpt_audit_prompt,
    build_negative_rerank_prompt,
    private_heldout_by_task,
    rerank_candidates_from_json,
    select_reranked_negatives,
    truncate,
    visible_match_features,
)
from auto_skill.cleaning.author_style.gpt_transport import (
    codex_response_text,
    load_codex_auth,
)
from auto_skill.mvp import parse_json_object

__all__ = [
    "audit_passes_thresholds",
    "audit_threshold_policy",
    "build_gpt_audit_prompt",
    "build_negative_rerank_prompt",
    "codex_response_text",
    "filter_by_pack_ids",
    "filter_negatives_by_pack_ids",
    "load_codex_auth",
    "pack_ids_with_negative_coverage",
    "private_heldout_by_task",
    "rerank_candidates_from_json",
    "run_gpt_audits",
    "run_gpt_negative_rerank",
    "select_reranked_negatives",
    "truncate",
    "visible_match_features",
]


def run_gpt_negative_rerank(
    packs: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    packs_by_id = {pack["pack_id"]: pack for pack in packs}
    private_by_pack = {row.get("pack_id"): row for row in private_rows if isinstance(row, dict)}
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_target[str(row["target_task_ref"])].append(row)

    instructions = (
        "You are a skeptical dataset cleaning judge. Return JSON only. "
        "Do not identify or infer real people."
    )
    tasks = sorted(by_target.items())

    def rerank_one(
        index: int,
        target_task_ref: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pack_id = target_task_ref.split("::heldout::", maxsplit=1)[0]
        pack = packs_by_id.get(pack_id)
        private_eval = private_by_pack.get(pack_id)
        rows = rows[: args.gpt_rerank_candidates]
        if pack is None or not isinstance(private_eval, dict):
            return rows[: args.negatives_per_heldout]
        prompt = build_negative_rerank_prompt(
            pack,
            private_eval,
            target_task_ref,
            rows,
            candidate_chars=args.gpt_rerank_candidate_chars,
        )
        parsed: dict[str, Any] = {}
        usage: dict[str, Any] | None = None
        last_error: Exception | None = None
        transport_error: Exception | None = None
        for attempt in range(1, args.gpt_max_attempts + 1):
            try:
                text, usage = codex_response_text(
                    prompt,
                    model=args.gpt_model,
                    auth_path=args.auth,
                    instructions=instructions,
                    service_tier=args.service_tier,
                    timeout_seconds=args.gpt_timeout_seconds,
                )
                parsed = parse_json_object(text)
                if "parse_error" in parsed and attempt < args.gpt_max_attempts:
                    continue
                break
            except Exception as exc:  # noqa: BLE001 - retry local data cleaning calls.
                last_error = exc
                if attempt >= args.gpt_max_attempts:
                    transport_error = exc
                    break
        try:
            if transport_error is not None:
                selected = []
                for row in rows[: args.negatives_per_heldout]:
                    fallback = dict(row)
                    fallback["prefilter_negative_id"] = fallback["negative_id"]
                    fallback["negative_id"] = (
                        f"{fallback['target_task_ref']}::negative::{len(selected) + 1}"
                    )
                    fallback["gpt_rerank"] = {
                        "status": "error_fallback_prefilter",
                        "error": str(transport_error)[:500],
                        "error_type": type(transport_error).__name__,
                        "attempts": args.gpt_max_attempts,
                    }
                    selected.append(fallback)
            else:
                selected = select_reranked_negatives(
                    rows,
                    parsed,
                    limit=args.negatives_per_heldout,
                )
            for row in selected:
                row["gpt_rerank"] = {
                    **row.get("gpt_rerank", {}),
                    "status": row.get("gpt_rerank", {}).get("status", "selected"),
                    "model": args.gpt_model,
                    "usage": usage,
                    "attempts": args.gpt_max_attempts if "parse_error" in parsed else attempt,
                }
            if transport_error is not None:
                status = "error"
            else:
                status = "success" if "parse_error" not in parsed else "parse_error"
        except Exception as exc:  # noqa: BLE001 - local data cleaning should continue.
            selected = []
            for row in rows[: args.negatives_per_heldout]:
                fallback = dict(row)
                fallback["prefilter_negative_id"] = fallback["negative_id"]
                fallback["gpt_rerank"] = {
                    "status": "error_fallback_prefilter",
                    "error": str(last_error or exc)[:500],
                    "error_type": type(last_error or exc).__name__,
                    "attempts": args.gpt_max_attempts,
                }
                selected.append(fallback)
            status = "error"
        print(f"[negative-rerank {index}/{len(tasks)}] {status} {target_task_ref}")
        return selected

    if args.gpt_rerank_num_threads <= 1:
        selected_groups = [
            rerank_one(index, target_ref, rows)
            for index, (target_ref, rows) in enumerate(tasks, start=1)
        ]
    else:
        selected_groups = []
        with ThreadPoolExecutor(max_workers=args.gpt_rerank_num_threads) as executor:
            futures = {
                executor.submit(rerank_one, index, target_ref, rows): index
                for index, (target_ref, rows) in enumerate(tasks, start=1)
            }
            for future in as_completed(futures):
                selected_groups.append(future.result())
    selected = [row for group in selected_groups for row in group]
    selected.sort(key=lambda row: row["negative_id"])
    return selected


def run_gpt_audits(
    packs: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    by_pack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in negatives:
        pack_id = row["target_task_ref"].split("::heldout::", maxsplit=1)[0]
        by_pack[pack_id].append(row)
    private_by_pack = {row.get("pack_id"): row for row in private_rows if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    instructions = (
        "You are a skeptical dataset quality auditor. Return JSON only. "
        "Do not reveal or infer real identities."
    )
    audit_packs = packs[: args.gpt_limit_authors]

    def audit_one(index: int, pack: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        prompt = build_gpt_audit_prompt(
            pack,
            by_pack.get(pack["pack_id"], []),
            private_by_pack.get(pack["pack_id"]),
        )
        last_http_error: urllib.error.HTTPError | None = None
        last_error: Exception | None = None
        text = ""
        usage: dict[str, Any] | None = None
        parsed: dict[str, Any] = {}
        try:
            for attempt in range(1, args.gpt_max_attempts + 1):
                try:
                    text, usage = codex_response_text(
                        prompt,
                        model=args.gpt_model,
                        auth_path=args.auth,
                        instructions=instructions,
                        service_tier=args.service_tier,
                        timeout_seconds=args.gpt_timeout_seconds,
                    )
                    parsed = parse_json_object(text)
                    if "parse_error" in parsed and attempt < args.gpt_max_attempts:
                        continue
                    break
                except urllib.error.HTTPError as exc:
                    last_http_error = exc
                    if attempt >= args.gpt_max_attempts:
                        raise
                except Exception as exc:  # noqa: BLE001 - retry local data cleaning calls.
                    last_error = exc
                    if attempt >= args.gpt_max_attempts:
                        raise
            status = "success" if "parse_error" not in parsed else "parse_error"
            row = {
                "schema_version": "author-style-gpt-audit/v1",
                "status": status,
                "pack_id": pack["pack_id"],
                "model": args.gpt_model,
                "usage": usage,
                "attempts": args.gpt_max_attempts if status == "parse_error" else attempt,
                "latency_seconds": round(time.monotonic() - started, 3),
                "audit": parsed,
            }
            if status != "success":
                row["raw_text"] = text[:2000]
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")[:1000]
            row = {
                "schema_version": "author-style-gpt-audit/v1",
                "status": "http_error",
                "pack_id": pack["pack_id"],
                "model": args.gpt_model,
                "attempts": args.gpt_max_attempts,
                "latency_seconds": round(time.monotonic() - started, 3),
                "error": error_text,
                "status_code": exc.code,
            }
        except Exception as exc:  # noqa: BLE001 - local smoke should record failures.
            row = {
                "schema_version": "author-style-gpt-audit/v1",
                "status": "error",
                "pack_id": pack["pack_id"],
                "model": args.gpt_model,
                "attempts": args.gpt_max_attempts,
                "latency_seconds": round(time.monotonic() - started, 3),
                "error": str(last_http_error or last_error or exc)[:1000],
                "error_type": type(last_http_error or last_error or exc).__name__,
            }
        row["_index"] = index
        return row

    if args.gpt_num_threads <= 1:
        rows = [audit_one(index, pack) for index, pack in enumerate(audit_packs, start=1)]
    else:
        with ThreadPoolExecutor(max_workers=args.gpt_num_threads) as executor:
            futures = {
                executor.submit(audit_one, index, pack): (index, pack)
                for index, pack in enumerate(audit_packs, start=1)
            }
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda item: int(item.get("_index") or 0))

    for row in rows:
        print(f"[gpt-audit {row['_index']}/{len(audit_packs)}] {row['status']} {row['pack_id']}")
        row.pop("_index", None)
    return rows
