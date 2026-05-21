"""GPT/Codex-backed audit and hard-negative rerank helpers."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from auto_skill.cleaning.author_style.common import CODEX_RESPONSES_URL
from auto_skill.mvp import parse_json_object


def load_codex_auth(path: Path) -> tuple[str, str | None]:
    with path.open("r", encoding="utf-8") as handle:
        auth = json.load(handle)
    tokens = auth.get("tokens") or {}
    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id")
    if not access_token:
        raise RuntimeError(f"No access token found in {path}")
    return str(access_token), str(account_id) if account_id else None


def codex_response_text(
    prompt: str,
    *,
    model: str,
    auth_path: Path,
    instructions: str,
    service_tier: str | None,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any] | None]:
    access_token, account_id = load_codex_auth(auth_path)
    body: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": prompt}],
        "stream": True,
        "store": False,
    }
    if service_tier:
        body["service_tier"] = service_tier
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if account_id:
        headers["OpenAI-Account-ID"] = account_id
    request = urllib.request.Request(
        CODEX_RESPONSES_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    text = ""
    usage = None
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            typ = event.get("type")
            if typ == "response.output_text.delta":
                text += event.get("delta", "")
            elif typ == "response.completed":
                usage = (event.get("response") or {}).get("usage")
    return text, usage


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


def build_gpt_audit_prompt(
    pack: dict[str, Any],
    negatives: list[dict[str, Any]],
    private_eval: dict[str, Any] | None = None,
) -> str:
    examples = []
    for example in pack["train_examples"]:
        examples.append(
            {
                "content_tags": example["metadata"]["public_content_tags"],
                "text": truncate(example["desired_output"]["text"], 900),
            }
        )
    heldout = []
    heldout_private_by_task = {
        row.get("task_ref"): row
        for row in (private_eval or {}).get("heldout_private", [])
        if isinstance(row, dict)
    }
    for task in pack["heldout_tasks"]:
        private_task = heldout_private_by_task.get(task.get("task_id")) or {}
        text = private_task.get("reference_output_private")
        if not isinstance(text, str) or not text.strip():
            continue
        heldout.append(
            {
                "content_tags": task["metadata"]["public_content_tags"],
                "text": truncate(text, 650),
            }
        )
    negative_samples = [
        {
            "negative_type": row["negative_type"],
            "content_tags": row["public_negative_content_tags"],
            "text": truncate(row["public_negative_text"], 650),
            "match_features": {
                key: value
                for key, value in row["match_features"].items()
                if not key.endswith("_private")
            },
        }
        for row in negatives[:3]
    ]
    payload = {
        "candidate_cluster_tags": (private_eval or {}).get("cluster_tags"),
        "style_summary": (private_eval or {}).get("style_summary"),
        "train_examples": examples,
        "heldout_samples": heldout,
        "hard_negative_samples": negative_samples,
    }
    return (
        "Audit this anonymous personal-writing style candidate for an auto-skill "
        "benchmark. Judge whether the examples have a stable, learnable personal "
        "style signal beyond topic. Also judge whether the negatives are strong. "
        "Do not identify the person. Return compact JSON with keys: usable "
        "(boolean), style_extractability_1_to_5, style_cluster_tags (array of "
        "short snake_case tags), topic_leakage_risk_1_to_5, model_familiarity_risk_1_to_5, "
        "negative_strength_1_to_5, should_use_for_smoke (boolean), evidence "
        "(array), reject_reasons (array).\n\n" + json.dumps(payload, ensure_ascii=False)
    )


def visible_match_features(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.get("match_features", {}).items()
        if not key.endswith("_private")
    }


def private_heldout_by_task(private_eval: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("task_ref")): row
        for row in (private_eval or {}).get("heldout_private", [])
        if isinstance(row, dict) and row.get("task_ref")
    }


def build_negative_rerank_prompt(
    pack: dict[str, Any],
    private_eval: dict[str, Any],
    target_task_ref: str,
    candidates: list[dict[str, Any]],
    *,
    candidate_chars: int,
) -> str:
    train_examples = [
        {
            "example_id": example.get("example_id"),
            "content_tags": example.get("metadata", {}).get("public_content_tags", []),
            "text": truncate(example["desired_output"]["text"], 520),
        }
        for example in pack.get("train_examples", [])
        if isinstance(example.get("desired_output"), dict)
        and isinstance(example["desired_output"].get("text"), str)
    ]
    private_heldout = private_heldout_by_task(private_eval).get(target_task_ref, {})
    heldout_text = str(private_heldout.get("reference_output_private") or "")
    payload = {
        "target_author_train_examples": train_examples,
        "target_heldout_reference": truncate(heldout_text, 620),
        "candidate_negatives": [
            {
                "candidate_id": row["negative_id"],
                "content_tags": row.get("public_negative_content_tags", []),
                "match_features": visible_match_features(row),
                "text": truncate(row["public_negative_text"], candidate_chars),
            }
            for row in candidates
        ],
    }
    return (
        "Select hard negative examples for an anonymous author-style benchmark.\n"
        "Goal: choose different-author candidates that would be hard to reject "
        "using style alone. Prefer candidates with similar punctuation rhythm, "
        "sentence shape, register, spelling habits, discourse moves, and era. "
        "Do not prefer candidates merely because they mention the same topic, "
        "people, place, date, or event. Reject candidates that are copied text, "
        "lyrics, templates, list memes, metadata/header artifacts, or obviously "
        "different genre/register.\n\n"
        "Return compact JSON only with key `ranked_negatives`, an array of objects: "
        "{candidate_id, style_confusability_1_to_5, topic_shortcut_risk_1_to_5, "
        "reject, reason}. Include every candidate_id exactly once, ordered from "
        "hardest to easiest negative.\n\n" + json.dumps(payload, ensure_ascii=False)
    )


def rerank_candidates_from_json(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = parsed.get("ranked_negatives")
    if isinstance(ranked, list):
        rows = [row for row in ranked if isinstance(row, dict)]
        if rows:
            return rows
    ids = parsed.get("ranked_candidate_ids")
    if isinstance(ids, list):
        return [{"candidate_id": str(candidate_id)} for candidate_id in ids]
    return []


def select_reranked_negatives(
    candidates: list[dict[str, Any]],
    parsed: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_id = {row["negative_id"]: row for row in candidates}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, item in enumerate(rerank_candidates_from_json(parsed), start=1):
        candidate_id = str(item.get("candidate_id") or item.get("negative_id") or item.get("id"))
        if candidate_id in seen or candidate_id not in by_id:
            continue
        seen.add(candidate_id)
        if item.get("reject") is True:
            continue
        row = dict(by_id[candidate_id])
        row["prefilter_negative_id"] = row["negative_id"]
        row["negative_id"] = f"{row['target_task_ref']}::negative::{len(selected) + 1}"
        row["gpt_rerank"] = {
            "status": "selected",
            "rank": rank,
            "style_confusability_1_to_5": item.get("style_confusability_1_to_5"),
            "topic_shortcut_risk_1_to_5": item.get("topic_shortcut_risk_1_to_5"),
            "reason": item.get("reason"),
        }
        selected.append(row)
        if len(selected) >= limit:
            break
    if len(selected) >= limit:
        return selected
    for row in candidates:
        if row["negative_id"] in seen:
            continue
        fallback = dict(row)
        fallback["prefilter_negative_id"] = fallback["negative_id"]
        fallback["negative_id"] = f"{fallback['target_task_ref']}::negative::{len(selected) + 1}"
        fallback["gpt_rerank"] = {"status": "fallback_prefilter"}
        selected.append(fallback)
        if len(selected) >= limit:
            break
    return selected


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


def audit_threshold_policy(args: argparse.Namespace) -> str:
    return str(getattr(args, "audit_thresholds", "none") or "none")


def audit_passes_thresholds(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if row.get("status") != "success":
        return False
    audit = row.get("audit")
    if not isinstance(audit, dict):
        return False
    policy = audit_threshold_policy(args)
    if policy == "none":
        return True
    if policy != "smoke":
        raise ValueError(f"unknown audit threshold policy: {policy}")
    if not (audit.get("usable") and audit.get("should_use_for_smoke")):
        return False
    return (
        float(audit.get("style_extractability_1_to_5") or 0) >= args.min_style_extractability
        and float(audit.get("negative_strength_1_to_5") or 0) >= args.min_negative_strength
        and float(audit.get("topic_leakage_risk_1_to_5") or 6) <= args.max_topic_leakage
        and float(audit.get("model_familiarity_risk_1_to_5") or 6) <= args.max_model_familiarity
    )


def filter_by_pack_ids(
    rows: list[dict[str, Any]],
    accepted_pack_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("pack_id") in accepted_pack_ids]


def filter_negatives_by_pack_ids(
    rows: list[dict[str, Any]],
    accepted_pack_ids: set[str],
    *,
    require_gpt_selected: bool,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        pack_id = str(row.get("target_task_ref", "")).split("::heldout::", maxsplit=1)[0]
        if pack_id not in accepted_pack_ids:
            continue
        if require_gpt_selected and row.get("gpt_rerank", {}).get("status") != "selected":
            continue
        filtered.append(row)
    return filtered


def pack_ids_with_negative_coverage(
    private_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    accepted_pack_ids: set[str],
    *,
    min_negatives_per_heldout: int,
) -> set[str]:
    if min_negatives_per_heldout <= 0:
        return set(accepted_pack_ids)
    counts = Counter(str(row.get("target_task_ref") or "") for row in negative_rows)
    covered_pack_ids: set[str] = set()
    for row in private_rows:
        pack_id = str(row.get("pack_id") or "")
        if pack_id not in accepted_pack_ids:
            continue
        heldout_rows = row.get("heldout_private")
        if not isinstance(heldout_rows, list) or not heldout_rows:
            continue
        task_refs = [
            str(heldout.get("task_ref") or "")
            for heldout in heldout_rows
            if isinstance(heldout, dict)
        ]
        has_enough_negatives = all(
            counts[task_ref] >= min_negatives_per_heldout for task_ref in task_refs
        )
        if task_refs and has_enough_negatives:
            covered_pack_ids.add(pack_id)
    return covered_pack_ids
