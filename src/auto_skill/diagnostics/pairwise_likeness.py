"""Pairwise output-likeness judge helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from auto_skill.diagnostics import pairwise_swap as _pairwise_swap
from auto_skill.mvp import parse_json_object
from auto_skill.schemas import UserExample

SCHEMA_VERSION = "pairwise-likeness-eval/v1"
EVALUATOR_KIND = "example_likeness_pairwise"

SWAP_AUDIT_SCHEMA_VERSION = _pairwise_swap.SWAP_AUDIT_SCHEMA_VERSION
SWAP_PAIR_STATUSES = _pairwise_swap.SWAP_PAIR_STATUSES
SWAP_ORDERS = _pairwise_swap.SWAP_ORDERS
CONFIDENCE_LEVELS = _pairwise_swap.CONFIDENCE_LEVELS
swap_flip_audit = _pairwise_swap.swap_flip_audit


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
