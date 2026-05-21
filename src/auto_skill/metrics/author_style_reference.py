"""Reference-retrieval oracle metric for personal author-style data."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from auto_skill.llm.parse import parse_json_object
from auto_skill.metrics.author_style_reference_report import (  # noqa: F401
    public_reference_retrieval_attempts,
    public_reference_retrieval_report,
    reference_retrieval_summary_markdown,
)

SCHEMA_VERSION = "author-style-reference-retrieval/v1"
EVALUATOR_KIND = "source_reference_retrieval"


@dataclass(frozen=True)
class ReferenceCandidate:
    candidate_id: str
    kind: str
    text: str
    source_task_id: str | None = None
    negative_id: str | None = None


@dataclass(frozen=True)
class ReferenceRetrievalJob:
    pack_id: str
    task_id: str
    source: str | None
    source_task_id: str | None
    target_text: str
    expected_candidate_id: str
    candidates: list[ReferenceCandidate]


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars].rstrip() + f"\n...[truncated {omitted} chars]"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_reference_retrieval_prompt(
    *,
    target_text: str,
    candidates: list[ReferenceCandidate],
    max_target_chars: int = 5000,
    max_candidate_chars: int = 3000,
) -> str:
    candidate_blocks = []
    for candidate in candidates:
        candidate_blocks.append(
            "Candidate "
            f"{candidate.candidate_id}\n"
            f"{truncate_text(candidate.text, max_candidate_chars)}"
        )
    return f"""You are evaluating whether an anonymous author-style benchmark has
a real same-author signal.

You will see one target text and several candidate reference texts. Exactly one
candidate is a same-author reference from the same style cluster. The other
candidates are hard negatives from other authors/styles.

Choose the candidate whose reusable writing style is closest to the target.
Focus on style: sentence rhythm, punctuation habits, register, discourse moves,
spelling quirks, hedging, formatting, and author voice. Do not choose merely by
topic, named entities, events, or factual overlap.

Return strict JSON:
{{
  "most_similar_candidate_id": "candidate id",
  "ranked_candidate_ids": ["best id", "second id"],
  "confidence": "low" | "medium" | "high",
  "rationale": "brief reason"
}}

Target text:
{truncate_text(target_text, max_target_chars)}

Candidate reference texts:
{chr(10).join(candidate_blocks)}
"""


def parse_reference_retrieval_report(
    text: str,
    *,
    candidate_ids: set[str],
) -> dict[str, Any]:
    report = parse_json_object(text)
    if "parse_error" in report:
        return report
    selected = _normalize_candidate_id(report.get("most_similar_candidate_id"), candidate_ids)
    if not isinstance(selected, str) or selected not in candidate_ids:
        report["parse_error"] = "most_similar_candidate_id_must_match_candidate_id"
        return report
    report["most_similar_candidate_id"] = selected
    ranked = report.get("ranked_candidate_ids")
    if not isinstance(ranked, list) or not ranked:
        report["parse_error"] = "ranked_candidate_ids_must_be_nonempty_list"
        return report
    clean_ranked = []
    seen = set()
    for item in ranked:
        normalized = _normalize_candidate_id(item, candidate_ids)
        if not isinstance(normalized, str) or normalized not in candidate_ids:
            report["parse_error"] = "ranked_candidate_ids_must_match_candidate_ids"
            return report
        if normalized in seen:
            report["parse_error"] = "ranked_candidate_ids_must_not_repeat"
            return report
        seen.add(normalized)
        clean_ranked.append(normalized)
    if clean_ranked[0] != selected:
        report["parse_error"] = "ranked_candidate_ids_first_must_equal_selected"
        return report
    report["ranked_candidate_ids"] = clean_ranked
    if report.get("confidence") not in {"low", "medium", "high"}:
        report["parse_error"] = "confidence_must_be_low_medium_or_high"
        return report
    if not isinstance(report.get("rationale"), str):
        report["parse_error"] = "rationale_must_be_string"
    return report


def reference_retrieval_status(
    *,
    finish_reason: str | None,
    report: dict[str, Any],
) -> str:
    if finish_reason not in {None, "stop"}:
        return "judge_incomplete"
    if "parse_error" in report:
        return "judge_parse_error"
    return "success"


def expected_rank(report: dict[str, Any], expected_candidate_id: str) -> int | None:
    ranked = report.get("ranked_candidate_ids")
    if not isinstance(ranked, list):
        return None
    for index, candidate_id in enumerate(ranked, start=1):
        if candidate_id == expected_candidate_id:
            return index
    return None


def build_reference_retrieval_jobs(
    *,
    packs: list[dict[str, Any]],
    private_eval: list[dict[str, Any]],
    hard_negatives: list[dict[str, Any]],
    references_per_target: int = 1,
    negatives_per_target: int = 4,
) -> tuple[list[ReferenceRetrievalJob], list[dict[str, Any]]]:
    if references_per_target != 1:
        raise ValueError("references_per_target must be 1 for this oracle metric")

    private_by_pack = {str(row.get("pack_id") or ""): row for row in private_eval}
    negatives_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hard_negatives:
        task_ref = str(row.get("target_task_ref") or "")
        if task_ref:
            negatives_by_task[task_ref].append(row)

    jobs: list[ReferenceRetrievalJob] = []
    skipped: list[dict[str, Any]] = []
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        private_row = private_by_pack.get(pack_id)
        if private_row is None:
            skipped.append({"pack_id": pack_id, "reason": "missing_private_eval"})
            continue
        train_candidates = _train_reference_candidates(
            pack,
            limit=references_per_target,
        )
        if not train_candidates:
            skipped.append({"pack_id": pack_id, "reason": "missing_train_reference"})
            continue
        heldout_by_task = {
            str(row.get("task_ref") or ""): row
            for row in private_row.get("heldout_private", [])
            if isinstance(row, dict)
        }
        for task in pack.get("heldout_tasks", []):
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("task_id") or "")
            private_task = heldout_by_task.get(task_id)
            target_text = (
                str(private_task.get("reference_output_private") or "")
                if isinstance(private_task, dict)
                else ""
            )
            if not target_text.strip():
                skipped.append(
                    {"pack_id": pack_id, "task_id": task_id, "reason": "missing_target_text"}
                )
                continue
            negative_candidates = _hard_negative_candidates(
                negatives_by_task.get(task_id, []),
                limit=negatives_per_target,
            )
            if not negative_candidates:
                skipped.append(
                    {"pack_id": pack_id, "task_id": task_id, "reason": "missing_negatives"}
                )
                continue
            expected = train_candidates[0].candidate_id
            candidates = _stable_candidate_order(
                train_candidates + negative_candidates,
                salt=task_id,
            )
            jobs.append(
                ReferenceRetrievalJob(
                    pack_id=pack_id,
                    task_id=task_id,
                    source=pack.get("source") if isinstance(pack.get("source"), str) else None,
                    source_task_id=(
                        task.get("source_task_id")
                        if isinstance(task.get("source_task_id"), str)
                        else None
                    ),
                    target_text=target_text,
                    expected_candidate_id=expected,
                    candidates=candidates,
                )
            )
    return jobs, skipped


def build_success_row(
    *,
    job: ReferenceRetrievalJob,
    judge: dict[str, Any],
    report: dict[str, Any],
    status: str,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    rank = expected_rank(report, job.expected_candidate_id) if status == "success" else None
    selected = report.get("most_similar_candidate_id") if status == "success" else None
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": job.pack_id,
        "task_id": job.task_id,
        "source": job.source,
        "source_task_id": job.source_task_id,
        "evaluator_kind": EVALUATOR_KIND,
        "status": status,
        "target_provenance": "source_private_heldout_reference",
        "target_sha256": text_sha256(job.target_text),
        "expected_candidate_id": job.expected_candidate_id,
        "selected_candidate_id": selected,
        "oracle_correct": selected == job.expected_candidate_id if status == "success" else None,
        "expected_rank": rank,
        "candidate_count": len(job.candidates),
        "hard_negative_count": _hard_negative_count(job.candidates),
        "candidate_manifest": [
            {
                "candidate_id": candidate.candidate_id,
                "kind": candidate.kind,
                "text_sha256": text_sha256(candidate.text),
                "source_task_id": candidate.source_task_id,
                "negative_id": candidate.negative_id,
            }
            for candidate in job.candidates
        ],
        "judge": judge,
        "judge_parse_attempts": public_reference_retrieval_attempts(attempts),
        "judge_report": public_reference_retrieval_report(report),
    }


def build_error_row(
    *,
    job: ReferenceRetrievalJob,
    status: str,
    error: str,
    judge_model: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": job.pack_id,
        "task_id": job.task_id,
        "source": job.source,
        "source_task_id": job.source_task_id,
        "evaluator_kind": EVALUATOR_KIND,
        "status": status,
        "target_provenance": "source_private_heldout_reference",
        "target_sha256": text_sha256(job.target_text),
        "expected_candidate_id": job.expected_candidate_id,
        "selected_candidate_id": None,
        "oracle_correct": None,
        "expected_rank": None,
        "candidate_count": len(job.candidates),
        "hard_negative_count": _hard_negative_count(job.candidates),
        "candidate_manifest": [
            {
                "candidate_id": candidate.candidate_id,
                "kind": candidate.kind,
                "text_sha256": text_sha256(candidate.text),
                "source_task_id": candidate.source_task_id,
                "negative_id": candidate.negative_id,
            }
            for candidate in job.candidates
        ],
        "judge": {"model": judge_model} if judge_model else None,
        "judge_parse_attempts": [],
        "judge_report": None,
        "error": error,
    }


def summarize_reference_retrieval_rows(
    rows: list[dict[str, Any]],
    *,
    skipped: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    latest = _latest_rows(rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in latest)
    success_rows = [row for row in latest if row.get("status") == "success"]
    correct_rows = [row for row in success_rows if row.get("oracle_correct") is True]
    ranks = [
        int(row["expected_rank"])
        for row in success_rows
        if isinstance(row.get("expected_rank"), int)
    ]
    by_source: dict[str, dict[str, Any]] = {}
    for source, source_rows in _group_by_source(success_rows).items():
        source_correct = [row for row in source_rows if row.get("oracle_correct") is True]
        by_source[source] = {
            "success": len(source_rows),
            "correct": len(source_correct),
            "accuracy": _rate(len(source_correct), len(source_rows)),
            "mean_expected_rank": _mean(
                [
                    int(row["expected_rank"])
                    for row in source_rows
                    if isinstance(row.get("expected_rank"), int)
                ]
            ),
        }
    return {
        "schema_version": "author-style-reference-retrieval-summary/v1",
        "evaluator_kind": EVALUATOR_KIND,
        "rows": len(latest),
        "status_counts": dict(status_counts),
        "success": len(success_rows),
        "correct": len(correct_rows),
        "accuracy": _rate(len(correct_rows), len(success_rows)),
        "mean_expected_rank": _mean(ranks),
        "median_expected_rank": _median(ranks),
        "by_source": by_source,
        "skipped": {
            "rows": len(skipped or []),
            "reason_counts": dict(
                Counter(str(row.get("reason") or "unknown") for row in skipped or [])
            ),
        },
    }


def _train_reference_candidates(
    pack: dict[str, Any],
    *,
    limit: int,
) -> list[ReferenceCandidate]:
    candidates = []
    for example in pack.get("train_examples", []):
        if not isinstance(example, dict):
            continue
        desired = example.get("desired_output")
        text = desired.get("text") if isinstance(desired, dict) else None
        candidate_id = example.get("example_id")
        if not isinstance(text, str) or not text.strip() or not isinstance(candidate_id, str):
            continue
        candidates.append(
            ReferenceCandidate(
                candidate_id=candidate_id,
                kind="same_author_reference",
                text=text,
                source_task_id=(
                    example.get("source_task_id")
                    if isinstance(example.get("source_task_id"), str)
                    else None
                ),
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def _normalize_candidate_id(value: Any, candidate_ids: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    if value in candidate_ids:
        return value
    stripped = value.strip()
    if stripped in candidate_ids:
        return stripped
    lower_prefix = "candidate "
    if stripped.lower().startswith(lower_prefix):
        suffix = stripped[len(lower_prefix) :].strip()
        if suffix in candidate_ids:
            return suffix
    return stripped


def _hard_negative_count(candidates: list[ReferenceCandidate]) -> int:
    return sum(1 for candidate in candidates if candidate.kind != "same_author_reference")


def _hard_negative_candidates(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[ReferenceCandidate]:
    candidates = []
    for row in rows[:limit]:
        text = row.get("public_negative_text")
        negative_id = row.get("negative_id")
        if not isinstance(text, str) or not text.strip() or not isinstance(negative_id, str):
            continue
        candidates.append(
            ReferenceCandidate(
                candidate_id=f"negative::{len(candidates) + 1}",
                kind=str(row.get("negative_type") or "hard_negative"),
                text=text,
                negative_id=negative_id,
            )
        )
    return candidates


def _stable_candidate_order(
    candidates: list[ReferenceCandidate],
    *,
    salt: str,
) -> list[ReferenceCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{salt}:{candidate.candidate_id}".encode()
        ).hexdigest(),
    )


def _latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    unkeyed = []
    for row in rows:
        pack_id = row.get("pack_id")
        task_id = row.get("task_id")
        if isinstance(pack_id, str) and isinstance(task_id, str):
            latest[(pack_id, task_id)] = row
        else:
            unkeyed.append(row)
    return unkeyed + list(latest.values())


def _group_by_source(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "unknown")].append(row)
    return grouped


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return round((ordered[middle - 1] + ordered[middle]) / 2, 3)
