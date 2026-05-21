"""Swapped-order audit helpers for pairwise likeness diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

SOURCE_SCHEMA_VERSION = "pairwise-likeness-eval/v1"
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
        "source_schema_version": SOURCE_SCHEMA_VERSION,
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
