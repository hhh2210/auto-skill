"""Pairwise output-likeness judge helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from auto_skill.mvp import parse_json_object
from auto_skill.schemas import UserExample

SCHEMA_VERSION = "pairwise-likeness-eval/v1"
EVALUATOR_KIND = "example_likeness_pairwise"

SWAP_AUDIT_SCHEMA_VERSION = "pairwise-likeness-swap-audit/v1"
SWAP_PAIR_STATUSES = (
    "swap_stable_candidate_wins",
    "swap_stable_anchor_wins",
    "swap_stable_tie",
    "swap_flip_decisive",
    "swap_one_decisive_one_tie",
    "swap_pair_incomplete",
)
SWAP_ORDERS = ("anchor_first", "candidate_first")
CONFIDENCE_LEVELS = ("low", "medium", "high", "unknown")


def truncate_text(text: str, max_chars: int) -> str:
    """Keep prompts bounded while making truncation explicit."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars].rstrip() + f"\n...[truncated {omitted} chars]"


def build_pairwise_likeness_prompt(
    *,
    examples: list[UserExample],
    heldout_task: dict[str, Any],
    output_a: str,
    output_b: str,
    max_example_chars: int = 3500,
    max_task_chars: int = 4000,
    max_output_chars: int = 6000,
) -> str:
    """Build a blind pairwise prompt comparing which output better matches examples."""

    example_blocks = []
    for index, example in enumerate(examples, start=1):
        example_blocks.append(
            f"Example {index}\n"
            f"Task:\n{truncate_text(example.task_input, max_example_chars)}\n\n"
            f"Reference output:\n{truncate_text(example.output or '', max_example_chars)}"
        )
    examples_text = "\n\n---\n\n".join(example_blocks)
    return f"""Compare two heldout-task outputs against the reusable writing
patterns visible in the user examples.

Use only the user examples, heldout task input, and the two candidate outputs.
Do not use hidden rubrics, official scores, skill text, model names, or any
private benchmark metadata.

Judge "which output looks more like the examples in reusable ways" while still
answering the heldout task. Reward transferable structure, specificity depth,
tone, constraint handling, and example-family conventions. Penalize copying
one-off facts from examples, generic answers, missing concrete depth, wrong
format, and task noncompliance.

Return strict JSON:
{{
  "winner": "A" | "B" | "tie",
  "confidence": "low" | "medium" | "high",
  "pattern_fidelity_winner": "A" | "B" | "tie",
  "task_fit_winner": "A" | "B" | "tie",
  "rationale": "brief reason"
}}

User examples:
{examples_text}

Heldout task:
{truncate_text(str(heldout_task.get("task_input") or ""), max_task_chars)}

Output A:
{truncate_text(output_a, max_output_chars)}

Output B:
{truncate_text(output_b, max_output_chars)}
"""


def parse_pairwise_likeness_report(text: str) -> dict[str, Any]:
    report = parse_json_object(text)
    if "parse_error" in report:
        return report
    for key in ("winner", "pattern_fidelity_winner", "task_fit_winner"):
        if report.get(key) not in {"A", "B", "tie"}:
            report["parse_error"] = f"{key}_must_be_A_B_or_tie"
            return report
    if report.get("confidence") not in {"low", "medium", "high"}:
        report["parse_error"] = "confidence_must_be_low_medium_or_high"
        return report
    if not isinstance(report.get("rationale"), str):
        report["parse_error"] = "rationale_must_be_string"
    return report


def pairwise_status(*, finish_reason: str | None, report: dict[str, Any]) -> str:
    if finish_reason != "stop":
        return "judge_incomplete"
    if "parse_error" in report:
        return "judge_parse_error"
    return "success"


def winner_mode(report: dict[str, Any], *, mode_a: str, mode_b: str) -> str | None:
    winner = report.get("winner")
    if winner == "A":
        return mode_a
    if winner == "B":
        return mode_b
    if winner == "tie":
        return "tie"
    return None


def summarize_pairwise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize call-level and swapped-pair-level anchor comparisons."""

    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    by_mode: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("candidate_mode") or "")
        if not mode:
            continue
        entry = by_mode.setdefault(
            mode,
            {
                "success_calls": 0,
                "candidate_wins": 0,
                "anchor_wins": 0,
                "ties": 0,
            },
        )
        if row.get("status") != "success":
            continue
        entry["success_calls"] += 1
        verdict = row.get("winner_mode")
        if verdict == mode:
            entry["candidate_wins"] += 1
        elif verdict == row.get("anchor_mode"):
            entry["anchor_wins"] += 1
        elif verdict == "tie":
            entry["ties"] += 1
        grouped[
            (
                str(row.get("pack_id") or ""),
                str(row.get("task_id") or ""),
                mode,
            )
        ].append(row)

    paired_by_mode: dict[str, dict[str, Any]] = {}
    for (_, _, mode), group_rows in grouped.items():
        success_rows = [row for row in group_rows if row.get("status") == "success"]
        if not success_rows:
            continue
        entry = paired_by_mode.setdefault(
            mode,
            {
                "paired_tasks": 0,
                "candidate_wins": 0,
                "anchor_wins": 0,
                "ties_or_splits": 0,
                "two_order_tasks": 0,
            },
        )
        entry["paired_tasks"] += 1
        if len(success_rows) >= 2:
            entry["two_order_tasks"] += 1
        votes = Counter(str(row.get("winner_mode") or "") for row in success_rows)
        anchor_mode = str(success_rows[0].get("anchor_mode") or "")
        candidate_votes = votes[mode]
        anchor_votes = votes[anchor_mode]
        if candidate_votes > anchor_votes:
            entry["candidate_wins"] += 1
        elif anchor_votes > candidate_votes:
            entry["anchor_wins"] += 1
        else:
            entry["ties_or_splits"] += 1

    for entry in by_mode.values():
        decisive = entry["candidate_wins"] + entry["anchor_wins"]
        entry["candidate_win_rate_decisive_calls"] = (
            entry["candidate_wins"] / decisive if decisive else None
        )
        total = entry["success_calls"]
        entry["candidate_win_rate_all_calls"] = (
            entry["candidate_wins"] / total if total else None
        )
    for entry in paired_by_mode.values():
        decisive = entry["candidate_wins"] + entry["anchor_wins"]
        entry["candidate_win_rate_decisive_tasks"] = (
            entry["candidate_wins"] / decisive if decisive else None
        )
        total = entry["paired_tasks"]
        entry["candidate_win_rate_all_tasks"] = (
            entry["candidate_wins"] / total if total else None
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_kind": EVALUATOR_KIND,
        "rows": len(rows),
        "status_counts": dict(status_counts),
        "call_level": by_mode,
        "paired_task_level": paired_by_mode,
    }


def swap_flip_audit(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Audit whether swapped pairwise judge orders agree on the winner."""

    row_list = list(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    status_counts = Counter(str(row.get("status") or "unknown") for row in row_list)
    for row in row_list:
        key = (
            str(row.get("pack_id") or ""),
            str(row.get("task_id") or ""),
            str(row.get("candidate_mode") or ""),
        )
        if all(key):
            grouped[key].append(row)

    by_candidate: dict[str, dict[str, Any]] = {}
    pair_records: list[dict[str, Any]] = []
    for (pack_id, task_id, candidate_mode), group_rows in sorted(grouped.items()):
        entry = by_candidate.setdefault(candidate_mode, _empty_swap_candidate_entry())
        anchor_mode = _first_string(group_rows, "anchor_mode")
        pair_status, winners_by_order = _classify_swap_pair(
            group_rows,
            anchor_mode=anchor_mode,
            candidate_mode=candidate_mode,
        )
        entry["pairs"] += 1
        entry["counts"][pair_status] += 1
        entry["confidence_by_pair_status"].setdefault(
            pair_status, _empty_confidence_counts()
        )
        for row in group_rows:
            confidence = _pairwise_confidence(row)
            entry["call_confidence_counts"][confidence] += 1
            entry["confidence_by_pair_status"][pair_status][confidence] += 1
            entry["call_status_counts"][str(row.get("status") or "unknown")] += 1

        pair_records.append(
            {
                "pack_id": pack_id,
                "task_id": task_id,
                "anchor_mode": anchor_mode,
                "candidate_mode": candidate_mode,
                "status": pair_status,
                "orders_present": sorted(
                    str(row.get("order") or "") for row in group_rows if row.get("order")
                ),
                "winner_by_order": winners_by_order,
                "confidence_by_order": {
                    str(row.get("order") or ""): _pairwise_confidence(row)
                    for row in group_rows
                    if row.get("order")
                },
                "row_status_by_order": {
                    str(row.get("order") or ""): str(row.get("status") or "unknown")
                    for row in group_rows
                    if row.get("order")
                },
            }
        )

    overall = _empty_swap_candidate_entry()
    for entry in by_candidate.values():
        _finalize_swap_entry(entry)
        _merge_swap_entries(overall, entry)
    _finalize_swap_entry(overall)
    return {
        "schema_version": SWAP_AUDIT_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        "evaluator_kind": EVALUATOR_KIND,
        "rows": len(row_list),
        "swap_pairs": len(pair_records),
        "status_counts": dict(status_counts),
        "by_candidate": {
            mode: _jsonable_swap_entry(entry)
            for mode, entry in sorted(by_candidate.items())
        },
        "overall": _jsonable_swap_entry(overall),
        "pairs": pair_records,
    }


def pairwise_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Pairwise Example-Likeness Win-Rate",
        "",
        "## Call level",
        "",
        "| Candidate vs anchor | Success calls | Candidate wins | Anchor wins | "
        "Ties | Win-rate decisive | Win-rate all |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in sorted(summary.get("call_level", {}).items()):
        lines.append(
            f"| {mode} | {row['success_calls']} | {row['candidate_wins']} | "
            f"{row['anchor_wins']} | {row['ties']} | "
            f"{_pct(row.get('candidate_win_rate_decisive_calls'))} | "
            f"{_pct(row.get('candidate_win_rate_all_calls'))} |"
        )
    lines.extend(
        [
            "",
            "## Swapped-order task level",
            "",
            "| Candidate vs anchor | Tasks | Two-order tasks | Candidate wins | "
            "Anchor wins | Ties/splits | Win-rate decisive | Win-rate all |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, row in sorted(summary.get("paired_task_level", {}).items()):
        lines.append(
            f"| {mode} | {row['paired_tasks']} | {row['two_order_tasks']} | "
            f"{row['candidate_wins']} | {row['anchor_wins']} | "
            f"{row['ties_or_splits']} | "
            f"{_pct(row.get('candidate_win_rate_decisive_tasks'))} | "
            f"{_pct(row.get('candidate_win_rate_all_tasks'))} |"
        )
    lines.extend(
        ["", "## Status", "", "```json", _json_dump(summary.get("status_counts", {})), "```"]
    )
    return "\n".join(lines) + "\n"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _json_dump(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _empty_swap_candidate_entry() -> dict[str, Any]:
    return {
        "pairs": 0,
        "counts": Counter({status: 0 for status in SWAP_PAIR_STATUSES}),
        "call_status_counts": Counter(),
        "call_confidence_counts": Counter({level: 0 for level in CONFIDENCE_LEVELS}),
        "confidence_by_pair_status": {
            status: _empty_confidence_counts() for status in SWAP_PAIR_STATUSES
        },
    }


def _empty_confidence_counts() -> Counter[str]:
    return Counter({level: 0 for level in CONFIDENCE_LEVELS})


def _classify_swap_pair(
    rows: list[dict[str, Any]],
    *,
    anchor_mode: str,
    candidate_mode: str,
) -> tuple[str, dict[str, str]]:
    rows_by_order = {str(row.get("order") or ""): row for row in rows}
    if len(rows) != len(rows_by_order):
        return "swap_pair_incomplete", _winner_by_order(rows, anchor_mode, candidate_mode)
    if any(order not in rows_by_order for order in SWAP_ORDERS):
        return "swap_pair_incomplete", _winner_by_order(rows, anchor_mode, candidate_mode)
    if any(row.get("status") != "success" for row in rows_by_order.values()):
        return "swap_pair_incomplete", _winner_by_order(rows, anchor_mode, candidate_mode)

    winners = {
        order: _normalized_winner(
            rows_by_order[order],
            anchor_mode=anchor_mode,
            candidate_mode=candidate_mode,
        )
        for order in SWAP_ORDERS
    }
    winner_values = list(winners.values())
    if any(winner == "unknown" for winner in winner_values):
        return "swap_pair_incomplete", winners
    if winner_values == ["candidate", "candidate"]:
        return "swap_stable_candidate_wins", winners
    if winner_values == ["anchor", "anchor"]:
        return "swap_stable_anchor_wins", winners
    if winner_values == ["tie", "tie"]:
        return "swap_stable_tie", winners
    if set(winner_values) == {"candidate", "anchor"}:
        return "swap_flip_decisive", winners
    return "swap_one_decisive_one_tie", winners


def _winner_by_order(
    rows: list[dict[str, Any]],
    anchor_mode: str,
    candidate_mode: str,
) -> dict[str, str]:
    return {
        str(row.get("order") or ""): _normalized_winner(
            row,
            anchor_mode=anchor_mode,
            candidate_mode=candidate_mode,
        )
        for row in rows
        if row.get("order")
    }


def _normalized_winner(
    row: dict[str, Any],
    *,
    anchor_mode: str,
    candidate_mode: str,
) -> str:
    verdict = row.get("winner_mode")
    if verdict == candidate_mode:
        return "candidate"
    if verdict == anchor_mode:
        return "anchor"
    if verdict == "tie":
        return "tie"
    return "unknown"


def _pairwise_confidence(row: dict[str, Any]) -> str:
    report = row.get("pairwise_report")
    if isinstance(report, dict) and report.get("confidence") in {"low", "medium", "high"}:
        return str(report["confidence"])
    return "unknown"


def _first_string(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _finalize_swap_entry(entry: dict[str, Any]) -> None:
    counts = entry["counts"]
    decisive_pairs = (
        counts["swap_stable_candidate_wins"]
        + counts["swap_stable_anchor_wins"]
        + counts["swap_flip_decisive"]
    )
    stable_decisive_pairs = (
        counts["swap_stable_candidate_wins"] + counts["swap_stable_anchor_wins"]
    )
    entry["decisive_pair_count"] = decisive_pairs
    entry["swap_flip_rate_decisive_pairs"] = (
        counts["swap_flip_decisive"] / decisive_pairs if decisive_pairs else None
    )
    entry["swap_stable_decisive_pair_count"] = stable_decisive_pairs
    entry["swap_stable_candidate_win_rate_decisive"] = (
        counts["swap_stable_candidate_wins"] / stable_decisive_pairs
        if stable_decisive_pairs
        else None
    )


def _merge_swap_entries(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["pairs"] += source["pairs"]
    target["counts"].update(source["counts"])
    target["call_status_counts"].update(source["call_status_counts"])
    target["call_confidence_counts"].update(source["call_confidence_counts"])
    for status, counts in source["confidence_by_pair_status"].items():
        target["confidence_by_pair_status"].setdefault(status, _empty_confidence_counts())
        target["confidence_by_pair_status"][status].update(counts)


def _jsonable_swap_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "pairs": entry["pairs"],
        "counts": {status: int(entry["counts"][status]) for status in SWAP_PAIR_STATUSES},
        "decisive_pair_count": entry.get("decisive_pair_count", 0),
        "swap_flip_rate_decisive_pairs": entry.get("swap_flip_rate_decisive_pairs"),
        "swap_stable_decisive_pair_count": entry.get("swap_stable_decisive_pair_count", 0),
        "swap_stable_candidate_win_rate_decisive": entry.get(
            "swap_stable_candidate_win_rate_decisive"
        ),
        "call_status_counts": dict(sorted(entry["call_status_counts"].items())),
        "call_confidence_counts": {
            level: int(entry["call_confidence_counts"][level])
            for level in CONFIDENCE_LEVELS
        },
        "confidence_by_pair_status": {
            status: {
                level: int(entry["confidence_by_pair_status"][status][level])
                for level in CONFIDENCE_LEVELS
            }
            for status in SWAP_PAIR_STATUSES
        },
    }
