"""Grounding checks for generated heldout outputs."""

from __future__ import annotations

from typing import Any

from auto_skill.mvp import parse_json_object

SCHEMA_VERSION = "grounding-eval/v1"
EVALUATOR_KIND = "heldout_material_grounding_llm_judge"


def truncate_text(text: str, *, max_chars: int) -> str:
    """Return text with a visible truncation marker when it is too long."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[TRUNCATED after {max_chars} chars]"


def heldout_evidence_text(task: dict[str, Any], *, max_chars: int) -> str:
    """Format heldout task input and materials as factual grounding evidence."""

    chunks = [f"Task input:\n{task.get('task_input') or ''}"]
    materials = task.get("materials") or []
    for index, material in enumerate(materials, start=1):
        if not isinstance(material, dict):
            continue
        path = material.get("path") or f"material-{index}"
        text = material.get("text") or ""
        chunks.append(f"Material {index}: {path}\n{text}")
    return truncate_text("\n\n".join(chunks), max_chars=max_chars)


def build_grounding_prompt(
    *,
    heldout_task: dict[str, Any],
    candidate_output: str,
    mode: str,
    max_evidence_chars: int = 60000,
    max_candidate_chars: int = 60000,
) -> str:
    """Build a judge prompt that checks whether concrete details are grounded."""

    evidence = heldout_evidence_text(heldout_task, max_chars=max_evidence_chars)
    candidate = truncate_text(candidate_output, max_chars=max_candidate_chars)
    return f"""Evaluate whether the candidate output is grounded in the current heldout task.

Allowed factual evidence:
- the heldout task input;
- the heldout task materials shown below.

Important:
- User examples or induced skills are not factual evidence for this heldout task.
- Do not reward plausible but unsupported specifics.
- Focus on concrete factual claims: named entities, numeric values, methods,
  datasets, standards, citations, cases, experimental results, tool names, dates,
  and other details that should come from the current task/materials.
- Generic advice, formatting choices, and common background knowledge are lower
  risk unless they are presented as task-specific facts.

Mode being checked: {mode}

Heldout evidence:
{evidence}

Candidate output:
{candidate}

Return strict JSON only, with this schema:
{{
  "grounding_score": 1-10,
  "hallucination_risk": 1-10,
  "unsupported_claim_count": integer,
  "unsupported_claims": [
    {{
      "claim": "short quote or paraphrase",
      "issue": "why the evidence does not support it",
      "severity": "low|medium|high"
    }}
  ],
  "supported_detail_examples": ["short examples of well-grounded details"],
  "summary": "one concise sentence"
}}
"""


def parse_grounding_report(text: str) -> dict[str, Any]:
    """Parse and normalize a grounding judge report."""

    parsed = parse_json_object(text)
    if "parse_error" in parsed:
        return parsed

    score = parsed.get("grounding_score")
    risk = parsed.get("hallucination_risk")
    claim_count = parsed.get("unsupported_claim_count")
    unsupported = parsed.get("unsupported_claims")

    errors: list[str] = []
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        errors.append("grounding_score_not_numeric")
    elif not 1 <= float(score) <= 10:
        errors.append("grounding_score_out_of_range")

    if isinstance(risk, bool) or not isinstance(risk, (int, float)):
        errors.append("hallucination_risk_not_numeric")
    elif not 1 <= float(risk) <= 10:
        errors.append("hallucination_risk_out_of_range")

    if isinstance(claim_count, bool) or not isinstance(claim_count, int):
        errors.append("unsupported_claim_count_not_integer")
    elif claim_count < 0:
        errors.append("unsupported_claim_count_negative")

    if not isinstance(unsupported, list):
        errors.append("unsupported_claims_not_list")

    if errors:
        return {**parsed, "parse_error": ";".join(errors)}
    return parsed


def grounding_status(report: dict[str, Any], finish_reason: str | None) -> str:
    """Classify a grounding judge response."""

    if finish_reason != "stop":
        return "judge_incomplete"
    if "parse_error" in report:
        return "judge_parse_error"
    return "success"


def summarize_grounding_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize grounding rows by mode."""

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode") or ""), []).append(row)

    modes: dict[str, dict[str, Any]] = {}
    for mode, items in sorted(by_mode.items()):
        success = [row for row in items if row.get("status") == "success"]
        scores = [
            float(row["grounding_report"]["grounding_score"])
            for row in success
            if isinstance(row.get("grounding_report"), dict)
        ]
        risks = [
            float(row["grounding_report"]["hallucination_risk"])
            for row in success
            if isinstance(row.get("grounding_report"), dict)
        ]
        counts = [
            int(row["grounding_report"]["unsupported_claim_count"])
            for row in success
            if isinstance(row.get("grounding_report"), dict)
        ]
        modes[mode] = {
            "count": len(items),
            "success": len(success),
            "mean_grounding_score": sum(scores) / len(scores) if scores else None,
            "mean_hallucination_risk": sum(risks) / len(risks) if risks else None,
            "mean_unsupported_claim_count": sum(counts) / len(counts) if counts else None,
        }

    return {
        "schema_version": "grounding-eval-summary/v1",
        "rows": len(rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status") or "") for row in rows})
        },
        "modes": modes,
    }
