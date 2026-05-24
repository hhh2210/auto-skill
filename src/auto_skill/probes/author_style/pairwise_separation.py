"""Pairwise same-author vs cross-author separation probe runner."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_skill.cleaning.author_style.gpt_transport import codex_response_text
from auto_skill.io.jsonl import load_jsonl
from auto_skill.llm.parse import parse_json_object


@dataclass(frozen=True)
class PairwiseSeparationConfig:
    jobs: Path
    out: Path
    summary_out: Path
    private_debug_out: Path | None = None
    model: str = "gpt-5.5"
    auth: Path = Path("~/.codex/auth.json").expanduser()
    service_tier: str = "priority"
    reasoning_effort: str = "high"
    timeout_seconds: float = 240.0
    num_threads: int = 16
    max_chars: int = 5000
    debug_sample_size: int = 60
    seed: int = 20260523
    limit: int | None = None
    resume: bool = False


def build_pairwise_prompt(job: dict[str, Any], *, max_chars: int) -> str:
    return f"""You are testing whether author identity is a usable style label.

Score how similar the reusable writing style is on a 0-10 scale. Focus on
sentence rhythm, punctuation habits, register, discourse moves, spelling quirks,
hedging, formatting, and author voice. Do not reward shared topic by itself.

Return strict JSON:
{{
  "similarity_score": 0,
  "confidence": "low" | "medium" | "high",
  "rationale": "brief reason"
}}

Text A:
{truncate(str(job["left_text"]), max_chars)}

Text B:
{truncate(str(job["right_text"]), max_chars)}
"""


def parse_pairwise_report(text: str) -> dict[str, Any]:
    report = parse_json_object(text)
    if "parse_error" in report:
        return {"parse_error": "json_parse_error"}
    score = report.get("similarity_score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 10:
        return {"parse_error": "similarity_score_must_be_0_to_10"}
    if report.get("confidence") not in {"low", "medium", "high"}:
        return {"parse_error": "confidence_must_be_low_medium_or_high"}
    if not isinstance(report.get("rationale"), str):
        return {"parse_error": "rationale_must_be_string"}
    report["similarity_score"] = round(float(score), 3)
    return report


def evaluate_pairwise_job(
    job: dict[str, Any],
    config: PairwiseSeparationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        text, usage = codex_response_text(
            build_pairwise_prompt(job, max_chars=config.max_chars),
            model=config.model,
            auth_path=config.auth,
            instructions=(
                "You are a strict evaluator of anonymous author-style similarity. "
                "Return only strict JSON."
            ),
            service_tier=config.service_tier,
            reasoning_effort=config.reasoning_effort,
            timeout_seconds=config.timeout_seconds,
        )
        report = parse_pairwise_report(text)
        status = "success" if "parse_error" not in report else "judge_parse_error"
        judge = judge_metadata(config, usage=usage, started=started)
        return (
            public_pairwise_row(job, status=status, judge=judge, report=report),
            private_pairwise_row(job, judge=judge, report=report),
        )
    except Exception as exc:  # noqa: BLE001 - checkpoint model errors per row.
        judge = judge_metadata(config)
        report = {"parse_error": "model_error", "rationale": f"{type(exc).__name__}: {exc}"}
        return (
            public_pairwise_row(job, status="model_error", judge=judge, report=report),
            private_pairwise_row(job, judge=judge, report=report),
        )


def judge_metadata(
    config: PairwiseSeparationConfig,
    *,
    usage: dict[str, Any] | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    judge: dict[str, Any] = {
        "provider": "codex_oauth",
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "service_tier": config.service_tier,
    }
    if usage is not None:
        judge["usage"] = usage
    if started is not None:
        judge["latency_seconds"] = round(time.monotonic() - started, 3)
    return judge


def public_pairwise_row(
    job: dict[str, Any],
    *,
    status: str,
    judge: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "author-style-pairwise-separation/v1",
        "job_id": job["job_id"],
        "pair_kind": job["pair_kind"],
        "corpus": job["corpus"],
        "left_sha256": job["left_sha256"],
        "right_sha256": job["right_sha256"],
        "left_word_count": job["left_word_count"],
        "right_word_count": job["right_word_count"],
        "status": status,
        "similarity_score": report.get("similarity_score") if status == "success" else None,
        "confidence": report.get("confidence") if status == "success" else None,
        "judge": judge,
        "judge_report": {
            key: report[key]
            for key in ("similarity_score", "confidence", "parse_error")
            if key in report
        },
    }


def private_pairwise_row(
    job: dict[str, Any],
    *,
    judge: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "author-style-pairwise-private-debug/v1",
        "job_id": job["job_id"],
        "pair_kind": job["pair_kind"],
        "corpus": job["corpus"],
        "left_author_hash": job["left_author_hash"],
        "right_author_hash": job["right_author_hash"],
        "left_text_excerpt": truncate(str(job["left_text"]), 500),
        "right_text_excerpt": truncate(str(job["right_text"]), 500),
        "similarity_score": report.get("similarity_score"),
        "confidence": report.get("confidence"),
        "judge_full_rationale": report.get("rationale"),
        "parse_error": report.get("parse_error"),
        "judge": judge,
    }


def summarize_pairwise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = latest_rows(rows)
    success = [row for row in latest if row.get("status") == "success"]
    by_corpus = {}
    for corpus in sorted({row["corpus"] for row in success}):
        within = scores(success, corpus=corpus, pair_kind="within_author")
        cross = scores(success, corpus=corpus, pair_kind="cross_author")
        by_corpus[corpus] = {
            "within_n": len(within),
            "cross_n": len(cross),
            "within_mean": mean(within),
            "cross_mean": mean(cross),
            "ks_d": ks_d(within, cross),
            "auc_within_gt_cross": auc(within, cross),
        }
    return {
        "schema_version": "author-style-pairwise-separation-summary/v1",
        "rows": len(latest),
        "raw_attempt_rows": len(rows),
        "success": len(success),
        "status_counts": dict(Counter(str(row.get("status") or "unknown") for row in latest)),
        "by_corpus": by_corpus,
    }


def scores(rows: list[dict[str, Any]], *, corpus: str, pair_kind: str) -> list[float]:
    return [
        float(row["similarity_score"])
        for row in rows
        if row["corpus"] == corpus and row["pair_kind"] == pair_kind
    ]


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_job_id = {str(row.get("job_id") or ""): row for row in rows}
    return list(by_job_id.values())


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def ks_d(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    values = sorted(set(left + right))
    return round(max(abs(cdf(left, value) - cdf(right, value)) for value in values), 4)


def cdf(values: list[float], threshold: float) -> float:
    return sum(1 for value in values if value <= threshold) / len(values)


def auc(within: list[float], cross: list[float]) -> float | None:
    if not within or not cross:
        return None
    wins = 0.0
    for left in within:
        for right in cross:
            wins += 1.0 if left > right else 0.5 if left == right else 0.0
    return round(wins / (len(within) * len(cross)), 4)


def run_pairwise_separation(
    config: PairwiseSeparationConfig,
    *,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    jobs = load_jsonl(config.jobs)
    if config.limit is not None:
        jobs = jobs[: config.limit]
    rows = load_jsonl(config.out) if config.resume and config.out.exists() else []
    done = {row["job_id"] for row in rows if row.get("status") == "success"}
    pending = [job for job in jobs if job["job_id"] not in done]
    sample_ids = debug_ids(jobs, config.debug_sample_size, config.seed)
    print_fn(f"pairwise jobs={len(jobs)} pending={len(pending)} threads={config.num_threads}")
    with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
        futures = {executor.submit(evaluate_pairwise_job, job, config): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            public, private = future.result()
            rows.append(public)
            append_jsonl(config.out, public)
            if config.private_debug_out and job["job_id"] in sample_ids:
                append_jsonl(config.private_debug_out, private)
            print_fn(f"  {job['job_id']}: {public['status']}")
    summary = summarize_pairwise_rows(rows)
    config.summary_out.parent.mkdir(parents=True, exist_ok=True)
    config.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def debug_ids(jobs: list[dict[str, Any]], sample_size: int, seed: int) -> set[str]:
    ordered = sorted(
        jobs, key=lambda job: hashlib.sha256(f"{seed}:{job['job_id']}".encode()).hexdigest()
    )
    return {job["job_id"] for job in ordered[:sample_size]}


def truncate(text: str, max_chars: int) -> str:
    return (
        text
        if len(text) <= max_chars
        else text[:max_chars].rstrip() + f"\n...[truncated {len(text) - max_chars} chars]"
    )
