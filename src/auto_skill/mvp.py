"""MVP helpers for example-driven auto-skill induction and evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from auto_skill.example_packs import material_context
from auto_skill.schemas import UserExample


@dataclass(frozen=True)
class PromptRunResult:
    """Result from one model call used by the MVP scripts."""

    text: str
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    duration_seconds: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillInductionResult:
    """A generated skill package plus the evidence used to build it."""

    pack_id: str
    mode: str
    skill_md: str
    feature_reports: list[dict[str, Any]] = field(default_factory=list)
    cross_example_report: dict[str, Any] | None = None
    leave_one_out: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = "skill-induction/v1"
        row["status"] = "success"
        row["solver_model"] = _first_call_model(self.model_calls)
        return row


def _first_call_model(model_calls: list[dict[str, Any]]) -> str | None:
    """Return the first non-empty ``model`` string from a model_calls list."""

    for call in model_calls:
        if not isinstance(call, dict):
            continue
        model = call.get("model")
        if isinstance(model, str) and model.strip():
            return model
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in an LLM response."""

    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match is not None:
        candidates.insert(0, fence_match.group(1).strip())

    last_error = "no_json_object_found"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if match is None:
                continue
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as nested_exc:
                last_error = str(nested_exc)
                continue
        if isinstance(parsed, dict):
            return parsed
        return {"raw_text": text, "parse_error": "json_root_is_not_object"}
    return {"raw_text": text, "parse_error": last_error}


def user_examples_from_pack(pack: dict[str, Any]) -> list[UserExample]:
    """Convert a clean example pack into the canonical user-example input."""

    examples = []
    for row in pack.get("train_examples", []):
        desired = row.get("desired_output")
        if not isinstance(desired, dict) or desired.get("status") != "generated":
            continue
        text = desired.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        materials = tuple(
            str(material.get("path"))
            for material in row.get("materials", [])
            if isinstance(material, dict) and material.get("path")
        )
        examples.append(
            UserExample(
                example_id=str(row["example_id"]),
                task_input=str(row["task_input"]),
                output=text,
                materials=materials,
                metadata={
                    "source": row.get("source"),
                    "source_task_id": row.get("source_task_id"),
                    "domain": row.get("domain", {}),
                },
            )
        )
    return examples


def format_user_example(example: UserExample) -> str:
    materials = "\n".join(f"- {item}" for item in example.materials) or "- None"
    output = example.output or "(missing output)"
    return f"""### Example {example.example_id}

Task input:
{example.task_input}

Desired output:
{output}

Materials:
{materials}
"""


def build_one_shot_skill_prompt(examples: list[UserExample]) -> str:
    joined = "\n\n".join(format_user_example(example) for example in examples)
    return f"""You are creating a reusable SKILL.md from user-provided examples.

Use only the visible task inputs, outputs, and materials in the examples.
Do not infer hidden rubrics, benchmark metadata, teacher traces, or heldout feedback.

Write one concise, operational SKILL.md. Include:
- When to use this skill
- Inputs expected from the user
- Procedure
- Required rules
- Optional rules
- Do not generalize
- Self-check before final answer

Examples:
{joined}
"""


def build_leave_one_out_validation_prompt(
    *,
    candidate_skill_md: str,
    validation_example: UserExample,
) -> str:
    return f"""Validate this candidate skill against a held-out user example.

The validation example was not used to create the candidate skill.
Use only the visible example. Do not use hidden rubrics or benchmark metadata.

Return JSON with:
- supported_rules: rules visibly supported by the validation example
- contradicted_rules: rules that conflict with the validation example
- optional_or_contextual_rules: rules that might be useful only in some cases
- outlier_notes: whether this example should narrow the skill scope

Candidate SKILL.md:
{candidate_skill_md}

Validation example:
{format_user_example(validation_example)}
"""


def build_validation_aware_skill_merge_prompt(
    *,
    candidate_skills: list[dict[str, Any]],
    leave_one_out_reports: list[dict[str, Any]],
) -> str:
    return f"""Merge leave-one-out candidate skills into the final SKILL.md.

Each candidate skill was generated from n-1 user-visible examples. Its validation
report comes from the one example held out from that induction round.

Merge policy:
- Keep rules broadly supported by validation reports as required rules.
- Move partially supported or context-specific rules to optional rules.
- Remove rules contradicted by multiple validation examples.
- Narrow the skill scope when validation reports identify outliers.
- Do not add hidden benchmark rubrics, judge prompts, or heldout task feedback.

Return the final validation-aware SKILL.md only.

Leave-one-out candidate skills:
{json.dumps(candidate_skills, ensure_ascii=False, indent=2)}

Leave-one-out validation reports:
{json.dumps(leave_one_out_reports, ensure_ascii=False, indent=2)}
"""


def build_heldout_generation_prompt(
    *,
    task: dict[str, Any],
    mode: str,
    examples: list[UserExample],
    skill_md: str | None = None,
    max_material_chars: int = 4000,
) -> str:
    task_id = task.get("task_id") or task.get("example_id") or task.get("source_task_id")
    materials = material_context(task.get("materials", []), max_chars=max_material_chars)
    examples_text = ""
    if mode == "few_shot_examples_only":
        examples_text = "\n\nUser examples:\n" + "\n\n".join(
            format_user_example(example) for example in examples
        )
    skill_text = ""
    if skill_md:
        skill_text = f"\n\nReusable skill:\n{skill_md}"
    return f"""Complete the heldout task below.

Mode: {mode}
Use only the provided user-visible examples, reusable skill, task input, and material excerpts.
Do not mention benchmark construction, hidden rubrics, or evaluation metadata.
Return only the final answer.

Task ID: {task_id}
{examples_text}
{skill_text}

Heldout task input:
{task["task_input"]}

Material excerpts:
{materials}
"""


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
