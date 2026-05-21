"""Minimal example-driven auto-skill induction without LOO or merge stages."""

from __future__ import annotations

import json
import time
from typing import Any

from auto_skill.baselines._protocol import PromptRunResult
from auto_skill.baselines._shared import format_user_example, user_examples_from_pack
from auto_skill.llm.parse import parse_json_object
from auto_skill.memory.skill import MemoryScope, SkillMemoryEntry, format_memory_for_prompt
from auto_skill.schemas import UserExample

SYSTEM_PROMPT = """You are an auto-skill research assistant.
Use only user-visible examples, visible outputs, visible materials, and explicit user emphasis.
Never use hidden benchmark rubrics, judge prompts, critique traces, heldout tasks,
or evaluator feedback."""


class MinimalInductionError(RuntimeError):
    """Raised when a minimal induction stage fails closed."""


def _pack_id_from_examples(examples: list[UserExample]) -> str | None:
    for example in examples:
        pack_id = example.metadata.get("pack_id")
        if isinstance(pack_id, str) and pack_id:
            return pack_id
    return None


def _scope_entries(
    entries: list[SkillMemoryEntry], examples: list[UserExample], scope: MemoryScope
) -> list[SkillMemoryEntry]:
    if scope == "cross_pack":
        return entries
    pack_id = _pack_id_from_examples(examples)
    if pack_id is None:
        return []
    if scope == "cross_pack_holdout":
        return [entry for entry in entries if entry.pack_id != pack_id]
    return [entry for entry in entries if entry.pack_id == pack_id]


def build_extraction_prompt(
    examples: list[UserExample], memory_entries: list[SkillMemoryEntry], scope: MemoryScope
) -> str:
    scoped_entries = _scope_entries(memory_entries, examples, scope)
    positive_memory = format_memory_for_prompt(scoped_entries, "positive")
    negative_memory = format_memory_for_prompt(scoped_entries, "negative")
    joined = "\n\n".join(format_user_example(example) for example in examples)
    return f"""Induce a reusable Codex skill from the user examples.

Write a complete SKILL.md artifact. Capture stable task patterns, output constraints,
workflow steps, and specific anti-generalization warnings. Do not copy example-specific
names or facts as general rules.

## Reusable patterns from prior packs
{positive_memory}

## Past failure modes — do not generalize
{negative_memory}

## User examples
{joined}

Return only the candidate SKILL.md markdown."""


def memory_report_for_entries(
    entries: list[SkillMemoryEntry], examples: list[UserExample], scope: MemoryScope
) -> dict[str, Any]:
    scoped_entries = _scope_entries(entries, examples, scope)
    return {
        "scope": scope,
        "entry_count": len(scoped_entries),
        "positive_count": sum(1 for entry in scoped_entries if entry.polarity == "positive"),
        "negative_count": sum(1 for entry in scoped_entries if entry.polarity == "negative"),
        "memory_ids": [entry.memory_id for entry in scoped_entries],
    }


def build_supervisor_prompt(candidate_skill_md: str, examples: list[UserExample]) -> str:
    joined = "\n\n".join(format_user_example(example) for example in examples)
    schema = {
        "artifact_critique": {
            "too_generic": ["string"],
            "over_specific": ["string"],
            "missing": ["string"],
        },
        "rule_grounding": [
            {
                "rule": "string",
                "supported_by": ["example_id"],
                "contradicted_by": ["example_id"],
                "out_of_scope": ["example_id"],
            }
        ],
    }
    return f"""You are the MIMO supervisor for one candidate auto-skill artifact.

Audit whether the artifact is reusable and grounded in the user-visible training examples.
Do not regenerate example outputs. Do not hold out examples. Do not use private rubrics
or judge metadata.

## Candidate SKILL.md
{candidate_skill_md}

## User examples
{joined}

Return one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""


def build_revision_prompt(candidate_skill_md: str, supervisor_report: dict[str, Any]) -> str:
    return f"""Revise the candidate SKILL.md using the supervisor report.

Keep grounded reusable rules, remove unsupported or over-specific rules, and add missing
user-visible rules.
Return only the final SKILL.md markdown.

## Candidate SKILL.md
{candidate_skill_md}

## Supervisor report JSON
{json.dumps(supervisor_report, ensure_ascii=False, indent=2)}"""


def _call(client: Any, prompt: str, *, stage: str) -> PromptRunResult:
    started = time.monotonic()
    result = client.complete(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    )
    run = PromptRunResult(
        text=result.text,
        model=getattr(result, "model", None),
        usage=getattr(result, "usage", None),
        finish_reason=getattr(result, "finish_reason", None),
        request_id=getattr(result, "request_id", None),
        duration_seconds=round(time.monotonic() - started, 3),
    )
    if not run.text.strip():
        raise MinimalInductionError(f"{stage} returned empty text")
    if run.finish_reason != "stop":
        raise MinimalInductionError(f"{stage} did not finish cleanly: {run.finish_reason}")
    return run


def _parse_supervisor(text: str) -> dict[str, Any]:
    parsed = parse_json_object(text)
    if "parse_error" in parsed:
        raise MinimalInductionError(f"supervisor returned invalid JSON: {parsed['parse_error']}")
    validate_supervisor_report(parsed)
    return parsed


def _prompt_tokens(usage: dict[str, Any] | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get("prompt_tokens", usage.get("input_tokens"))
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _require_list(value: Any, path: str) -> None:
    if not isinstance(value, list):
        raise MinimalInductionError(f"supervisor report {path} must be a list")


def validate_supervisor_report(report: dict[str, Any]) -> None:
    critique = report.get("artifact_critique")
    if not isinstance(critique, dict):
        raise MinimalInductionError("supervisor report artifact_critique must be an object")
    for key in ("too_generic", "over_specific", "missing"):
        _require_list(critique.get(key), f"artifact_critique.{key}")

    grounding = report.get("rule_grounding")
    _require_list(grounding, "rule_grounding")
    for index, item in enumerate(grounding):
        prefix = f"supervisor report rule_grounding[{index}]"
        if not isinstance(item, dict):
            raise MinimalInductionError(f"{prefix} must be an object")
        if not isinstance(item.get("rule"), str) or not item["rule"].strip():
            raise MinimalInductionError(f"{prefix}.rule is required")
        for key in ("supported_by", "contradicted_by", "out_of_scope"):
            _require_list(item.get(key), f"rule_grounding[{index}].{key}")


def induce_minimal(
    pack: dict[str, Any],
    *,
    solver_client: Any,
    supervisor_client: Any,
    memory: list[SkillMemoryEntry],
    scope: MemoryScope,
) -> dict[str, Any]:
    examples = user_examples_from_pack(pack)
    pack_id = str(pack["pack_id"])
    examples = [
        UserExample(
            example_id=example.example_id,
            task_input=example.task_input,
            output=example.output,
            user_notes=example.user_notes,
            materials=example.materials,
            metadata={**example.metadata, "pack_id": pack_id},
        )
        for example in examples
    ]
    memory_report = memory_report_for_entries(memory, examples, scope)
    extraction = _call(
        solver_client,
        build_extraction_prompt(examples, memory, scope),
        stage="minimal_extraction",
    )
    extraction_prompt_tokens = _prompt_tokens(extraction.usage)
    supervisor = _call(
        supervisor_client,
        build_supervisor_prompt(extraction.text, examples),
        stage="minimal_supervisor",
    )
    supervisor_report = _parse_supervisor(supervisor.text)
    revision = _call(
        solver_client,
        build_revision_prompt(extraction.text, supervisor_report),
        stage="minimal_revision",
    )
    model_calls = [
        {"stage": "minimal_extraction", **extraction.to_json()},
        {"stage": "minimal_supervisor", **supervisor.to_json()},
        {"stage": "minimal_revision", **revision.to_json()},
    ]
    return {
        "schema_version": "skill-induction/v1",
        "status": "success",
        "pack_id": pack_id,
        "mode": "auto_skill_minimal",
        "skill_md": revision.text,
        "feature_reports": [],
        "cross_example_report": None,
        "leave_one_out": [],
        "model_calls": model_calls,
        "solver_model": extraction.model or revision.model,
        "supervisor_model": supervisor.model,
        "supervisor_report": supervisor_report,
        "memory_report": {
            **memory_report,
            "extraction_prompt_tokens": extraction_prompt_tokens,
        },
        "extraction_prompt_tokens": extraction_prompt_tokens,
    }
