"""WritingBench official-prompt evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WritingBenchPromptTemplates:
    evaluate_system: str
    evaluate_prompt: str
    prompt_file: str


def load_writingbench_prompt_templates(root: Path) -> WritingBenchPromptTemplates:
    """Load WritingBench's official evaluator prompts from its local repository."""

    prompt_path = root / "prompt.py"
    if not prompt_path.exists():
        raise FileNotFoundError(f"WritingBench prompt.py not found: {prompt_path}")
    spec = importlib.util.spec_from_file_location("writingbench_official_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import WritingBench prompt.py: {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate_system = getattr(module, "evaluate_system", None)
    evaluate_prompt = getattr(module, "evaluate_prompt", None)
    if not isinstance(evaluate_system, str) or not isinstance(evaluate_prompt, str):
        raise AttributeError("WritingBench prompt.py must define evaluate_system/evaluate_prompt")
    return WritingBenchPromptTemplates(
        evaluate_system=evaluate_system,
        evaluate_prompt=evaluate_prompt,
        prompt_file=str(prompt_path),
    )


def process_gen_field(text: str) -> str:
    """Match WritingBench's post-processing for models that emit thinking traces."""

    marker = "</think>\n\n"
    marker_pos = text.find(marker)
    if marker_pos != -1:
        return text[marker_pos + len(marker) :]
    return text


def build_writingbench_official_prompt(
    *,
    template: str,
    query: str,
    response: str,
    criteria: dict[str, Any],
) -> str:
    """Fill WritingBench's official evaluator prompt template."""

    return template.format(
        query=query,
        response=process_gen_field(response),
        criteria=json.dumps(criteria, ensure_ascii=False),
    )


def parse_writingbench_score(text: str) -> dict[str, Any]:
    """Parse and validate one WritingBench criterion score."""

    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            return {"parse_error": "no_json_object_found", "raw_text": text}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {"parse_error": str(exc), "raw_text": text}

    if not isinstance(parsed, dict):
        return {"parse_error": "json_root_is_not_object", "raw_text": text}
    score = parsed.get("score")
    if isinstance(score, str) and score.isdigit():
        score = int(score)
        parsed["score"] = score
    if isinstance(score, bool):
        parsed["parse_error"] = "score_must_be_integer_1_to_10"
        return parsed
    if not isinstance(score, int) or score not in range(1, 11):
        parsed["parse_error"] = "score_must_be_integer_1_to_10"
    if not isinstance(parsed.get("reason"), str):
        parsed["parse_error"] = "reason_must_be_string"
    return parsed


def average_writingbench_scores(scores: dict[str, list[dict[str, Any]]]) -> float | None:
    values: list[int] = []
    for evaluations in scores.values():
        for evaluation in evaluations:
            score = evaluation.get("score")
            if isinstance(score, int) and not isinstance(score, bool):
                values.append(score)
    if not values:
        return None
    return sum(values) / len(values)
