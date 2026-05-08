#!/usr/bin/env python3
"""Export judge-disagreement packets for calibration review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402

Cell = tuple[str, str, str]
TaskCell = tuple[str, str]


def row_cell(row: dict[str, Any]) -> Cell:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("mode") or ""),
    )


def success_index(rows: list[dict[str, Any]]) -> dict[Cell, dict[str, Any]]:
    index: dict[Cell, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        cell = row_cell(row)
        if all(cell) and isinstance(row.get("overall_score"), (int, float)):
            index[cell] = row
    return index


def sign(value: float, *, epsilon: float = 1e-9) -> str:
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "tie"


def truncate_text(value: Any, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n...[truncated]"


def generation_text(row: dict[str, Any], *, max_chars: int) -> str | None:
    generation = row.get("generation")
    if not isinstance(generation, dict):
        return None
    return truncate_text(generation.get("text"), max_chars=max_chars)


def full_generation_text(row: dict[str, Any]) -> str | None:
    generation = row.get("generation")
    if not isinstance(generation, dict):
        return None
    text = generation.get("text")
    return text if isinstance(text, str) else None


def compact_scores(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores")
    if not isinstance(scores, dict):
        return {}
    compact: dict[str, Any] = {}
    for name, values in sorted(scores.items()):
        if not isinstance(values, list):
            continue
        compact[name] = [
            {
                key: item.get(key)
                for key in ("score", "reason", "parse_error")
                if isinstance(item, dict) and key in item
            }
            for item in values
            if isinstance(item, dict)
        ]
    return compact


def pack_task_context(
    packs: list[dict[str, Any]] | None,
    *,
    max_chars: int,
) -> dict[TaskCell, dict[str, Any]]:
    if not packs:
        return {}
    context: dict[TaskCell, dict[str, Any]] = {}
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        for task in pack.get("heldout_tasks") or []:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("task_id") or "")
            if not pack_id or not task_id:
                continue
            materials = task.get("materials") if isinstance(task.get("materials"), list) else []
            context[(pack_id, task_id)] = {
                "source": pack.get("source"),
                "domain": pack.get("domain"),
                "source_task_id": task.get("source_task_id"),
                "task_input": truncate_text(task.get("task_input"), max_chars=max_chars),
                "materials": [
                    {
                        key: truncate_text(value, max_chars=max_chars)
                        if isinstance(value, str)
                        else value
                        for key, value in material.items()
                    }
                    for material in materials
                    if isinstance(material, dict)
                ],
            }
    return context


def private_eval_context(
    private_rows: list[dict[str, Any]] | None,
) -> dict[TaskCell, dict[str, Any]]:
    if not private_rows:
        return {}
    context: dict[TaskCell, dict[str, Any]] = {}
    for row in private_rows:
        pack_id = str(row.get("pack_id") or "")
        for task in row.get("heldout_private") or []:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("task_ref") or "")
            if not pack_id or not task_id:
                continue
            context[(pack_id, task_id)] = {
                "supervision": task.get("supervision"),
                "judge": task.get("judge"),
            }
    return context


def build_disagreement_packets(
    *,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_label: str,
    right_label: str,
    baseline_mode: str,
    modes: list[str] | None = None,
    max_output_chars: int = 2000,
    packs: list[dict[str, Any]] | None = None,
    private_rows: list[dict[str, Any]] | None = None,
    require_same_outputs: bool = True,
) -> list[dict[str, Any]]:
    left = success_index(left_rows)
    right = success_index(right_rows)
    task_context = pack_task_context(packs, max_chars=max_output_chars)
    eval_context = private_eval_context(private_rows)
    observed_modes = {
        mode for _, _, mode in set(left) & set(right) if mode != baseline_mode
    }
    target_modes = set(modes or sorted(observed_modes))
    packets: list[dict[str, Any]] = []
    for pack_id, task_id, mode in sorted(set(left) & set(right)):
        if mode not in target_modes:
            continue
        baseline_cell = (pack_id, task_id, baseline_mode)
        if baseline_cell not in left or baseline_cell not in right:
            continue
        left_score = float(left[(pack_id, task_id, mode)]["overall_score"])
        right_score = float(right[(pack_id, task_id, mode)]["overall_score"])
        left_baseline = float(left[baseline_cell]["overall_score"])
        right_baseline = float(right[baseline_cell]["overall_score"])
        left_delta = left_score - left_baseline
        right_delta = right_score - right_baseline
        left_sign = sign(left_delta)
        right_sign = sign(right_delta)
        if left_sign == right_sign:
            continue
        candidate_left = left[(pack_id, task_id, mode)]
        candidate_right = right[(pack_id, task_id, mode)]
        baseline_left = left[baseline_cell]
        baseline_right = right[baseline_cell]
        candidate_left_full = full_generation_text(candidate_left)
        candidate_right_full = full_generation_text(candidate_right)
        baseline_left_full = full_generation_text(baseline_left)
        baseline_right_full = full_generation_text(baseline_right)
        candidate_left_text = generation_text(candidate_left, max_chars=max_output_chars)
        candidate_right_text = generation_text(candidate_right, max_chars=max_output_chars)
        baseline_left_text = generation_text(baseline_left, max_chars=max_output_chars)
        baseline_right_text = generation_text(baseline_right, max_chars=max_output_chars)
        same_candidate = candidate_left_full == candidate_right_full
        same_baseline = baseline_left_full == baseline_right_full
        if require_same_outputs and (not same_candidate or not same_baseline):
            raise ValueError(
                "left/right eval files do not contain identical candidate outputs for "
                f"{pack_id} {task_id} {mode}; pass --allow-output-mismatch only for "
                "solver-output disagreement analysis"
            )
        packets.append(
            {
                "schema_version": "judge-disagreement/v1",
                "pack_id": pack_id,
                "task_id": task_id,
                "mode": mode,
                "baseline_mode": baseline_mode,
                "disagreement_kind": (
                    "sign_flip" if {left_sign, right_sign} == {"positive", "negative"}
                    else "sign_or_tie_disagreement"
                ),
                "left": {
                    "label": left_label,
                    "score": left_score,
                    "baseline_score": left_baseline,
                    "delta": left_delta,
                    "delta_sign": left_sign,
                    "judge_model": candidate_left.get("judge_model"),
                    "scores": compact_scores(candidate_left),
                },
                "right": {
                    "label": right_label,
                    "score": right_score,
                    "baseline_score": right_baseline,
                    "delta": right_delta,
                    "delta_sign": right_sign,
                    "judge_model": candidate_right.get("judge_model"),
                    "scores": compact_scores(candidate_right),
                },
                "same_candidate_output": same_candidate,
                "same_baseline_output": same_baseline,
                "candidate_output": candidate_left_text,
                "baseline_output": baseline_left_text,
                "right_candidate_output": None if same_candidate else candidate_right_text,
                "right_baseline_output": None if same_baseline else baseline_right_text,
                "task_context": task_context.get((pack_id, task_id)),
                "private_eval_context": eval_context.get((pack_id, task_id)),
            }
        )
    return packets


def markdown_summary(packets: list[dict[str, Any]]) -> str:
    lines = [
        "# Judge Disagreement Packet",
        "",
        f"Total disagreements: {len(packets)}",
        "",
        "| Pack | Mode | Left delta | Right delta | Kind |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for packet in packets:
        lines.append(
            "| {pack} | {mode} | {left:+.3f} | {right:+.3f} | {kind} |".format(
                pack=packet["pack_id"],
                mode=packet["mode"],
                left=packet["left"]["delta"],
                right=packet["right"]["delta"],
                kind=packet["disagreement_kind"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--baseline-mode", default="prompt_only")
    parser.add_argument("--modes", help="Comma-separated modes to compare.")
    parser.add_argument("--packs", type=Path, help="Optional example packs JSONL for task context.")
    parser.add_argument(
        "--private-eval",
        type=Path,
        help="Optional private eval JSONL for rubric/checklist context.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--max-output-chars", type=int, default=2000)
    parser.add_argument(
        "--allow-output-mismatch",
        action="store_true",
        help="Permit left/right candidate text mismatches and include right-side outputs.",
    )
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()] if args.modes else None
    packets = build_disagreement_packets(
        left_rows=load_jsonl(args.left),
        right_rows=load_jsonl(args.right),
        left_label=args.left_label,
        right_label=args.right_label,
        baseline_mode=args.baseline_mode,
        modes=modes,
        max_output_chars=args.max_output_chars,
        packs=load_jsonl(args.packs) if args.packs else None,
        private_rows=load_jsonl(args.private_eval) if args.private_eval else None,
        require_same_outputs=not args.allow_output_mismatch,
    )
    write_jsonl(args.out, packets)
    summary = markdown_summary(packets)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
