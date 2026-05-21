"""Benchmark judge prompt helpers."""

from __future__ import annotations

import json
from typing import Any


def build_judge_prompt(
    *,
    task: dict[str, Any],
    candidate_output: str,
    private_eval: dict[str, Any] | None,
) -> str:
    if private_eval is None:
        raise ValueError("private_eval is required for benchmark scoring")
    judge = (private_eval or {}).get("judge") or {}
    judge_reference = {
        key: value for key, value in judge.items() if key != "domain_common_prompt"
    }
    common_prompt = judge.get("domain_common_prompt")
    if isinstance(common_prompt, dict):
        judge_reference["domain_common_prompt_keys"] = sorted(common_prompt)
    criteria = evaluation_criteria(private_eval)
    if not criteria:
        raise ValueError("private_eval has no rubric/checklist criteria")
    criteria_text = json.dumps(criteria, ensure_ascii=False, indent=2)
    return f"""Score the candidate output for the task.

Use the benchmark rubric/checklist below only for evaluation. It is not user-visible input.
This is an LLM rubric/checklist surrogate judge, not an official visual/PPT evaluator.
Follow the JSON schema below even if the original benchmark prompt used a different output format.
Return strict JSON with:
- overall_score: number from 1 to 10
- criterion_scores: object mapping criterion names to 1-10 scores
- rationale: concise evidence-based explanation
- major_failures: list of important issues

Judge metadata:
{json.dumps(judge_reference, ensure_ascii=False, indent=2)}

Rubric or checklist:
{criteria_text}

Task input:
{task["task_input"]}

Candidate output:
{candidate_output}
"""


def evaluation_criteria(private_eval: dict[str, Any] | None) -> list[Any]:
    """Extract actual rubric/checklist criteria from private eval metadata."""

    if private_eval is None:
        return []
    supervision = private_eval.get("supervision") or {}
    items = supervision.get("items") or supervision.get("criteria")
    if isinstance(items, list) and items:
        return items
    checklists = supervision.get("checklists")
    flattened = []
    if isinstance(checklists, dict):
        for group_name, group_items in sorted(checklists.items()):
            if not isinstance(group_items, list):
                continue
            for index, item in enumerate(group_items, start=1):
                flattened.append(
                    {
                        "group": group_name,
                        "index": index,
                        "requirement": item,
                    }
                )
    return flattened


def extract_overall_score(judge_report: dict[str, Any]) -> float | None:
    value = judge_report.get("overall_score")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
