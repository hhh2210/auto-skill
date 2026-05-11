"""Demo prompt builder for skill quality / clarity judging.

This is a research demo, not a benchmark runner. It uses a Ctx2Skill-compatible
five-dimension skill-quality schema, with wording adapted to user-example-derived
SKILL.md artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DIMENSIONS = (
    "faithfulness",
    "reusability",
    "effectiveness",
    "clarity",
    "conciseness",
)

SOURCE_NOTES = {
    "ctx2skill_quality": {
        "url": "https://arxiv.org/html/2604.27660v1#Sx5.F15",
        "takeaway": (
            "Judge skills on faithfulness, reusability, effectiveness, clarity, "
            "and conciseness with strict 1-5 scoring and JSON output."
        ),
    },
    "ctx2skill_generator": {
        "url": "https://github.com/S1s-Z/Ctx2Skill/blob/main/prompts/reasoner_generator.txt",
        "takeaway": (
            "Good SKILL.md files are actionable, concise, structured, and contain "
            "checklists, procedures, self-verification, and pitfalls."
        ),
    },
    "skillx_filter": {
        "url": "https://github.com/zjunlp/SkillX/blob/main/prompts/filter_prompts.py",
        "takeaway": (
            "Quality filters reject over-specific parameters, thin wrappers, and "
            "skills that are not reusable abstractions."
        ),
    },
    "skillx_extraction": {
        "url": "https://github.com/zjunlp/SkillX/blob/main/prompts/skill_prompts.py",
        "takeaway": (
            "Reusable skills need generic names, abstract parameters, self-contained "
            "content, non-duplication, and agent-centered notes."
        ),
    },
    "agent_skills_spec": {
        "url": "https://agentskills.io/specification",
        "takeaway": (
            "A SKILL.md should have valid YAML frontmatter, a useful description, "
            "focused body instructions, and progressive disclosure when needed."
        ),
    },
}


def load_examples(path: Path | None) -> str:
    if path is None:
        return "No examples supplied for this demo."
    public_examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} must contain JSON object rows")
            public_examples.extend(public_examples_from_row(row))
    if not public_examples:
        raise ValueError(f"{path} did not contain public train examples")
    return json.dumps(public_examples, ensure_ascii=False, indent=2)


def public_examples_from_row(row: dict) -> list[dict[str, object]]:
    """Project artifact rows to user-visible train-example fields only."""
    if isinstance(row.get("train_examples"), list):
        examples = row["train_examples"]
        if not all(isinstance(example, dict) for example in examples):
            raise ValueError("train_examples must contain JSON objects")
        return [_public_example(example) for example in examples]
    if "task_input" in row:
        return [_public_example(row)]
    return []


def _public_example(example: dict) -> dict[str, object]:
    desired_output = example.get("desired_output")
    output = example.get("output")
    if isinstance(desired_output, dict):
        output = desired_output.get("text")

    public = {
        "example_id": example.get("example_id"),
        "task_input": example.get("task_input"),
        "materials": example.get("materials", []),
        "output": output,
    }
    return {key: value for key, value in public.items() if value is not None}


def build_skill_quality_prompt(*, examples_text: str, skill_md: str) -> str:
    """Return the judge prompt for skill artifact quality / clarity."""
    return f"""You are an expert evaluator for reusable agent skill artifacts.

## Role
Act as a strict, detail-oriented judge. Assess whether the candidate skill is a
clear, reusable SKILL.md-style artifact derived from the provided user examples.
Do not be lenient. Penalize hallucination, redundancy, shallow advice, and
example-specific copying.

## Inputs
### User examples
{examples_text}

### Candidate skill artifact
{skill_md}

## What to evaluate
Evaluate the candidate skill as an artifact, not whether any downstream heldout
task was solved. Do not use benchmark rubrics, heldout tasks, private metadata,
or eval scores. Only use the examples above and the candidate skill.

## Dimensions
1. faithfulness: Does the skill strictly reflect reusable patterns supported by
   the examples, without hallucinated requirements or external facts?
2. reusability: Does the skill generalize across similar tasks, rather than
   copy one example's answer, topic, names, numbers, or task-specific facts?
3. effectiveness: Does the skill provide actionable procedures, checklists,
   templates, or self-verification steps that help an agent solve the task?
4. clarity: Is the skill clearly structured, unambiguous, and easy to execute as
   a SKILL.md-style instruction artifact?
5. conciseness: Is it compact and non-redundant, without filler that competes
   with task context?

## Strict scoring rules
- Score each dimension from 1 to 5.
- Average skills should not exceed 3 unless they are clearly strong.
- If the skill contains hallucinated rules unsupported by examples, cap
  faithfulness at 2.
- If it copies concrete names, numbers, or case facts from examples as rules,
  cap reusability at 2.
- If it is mostly generic advice such as "be clear" or "follow requirements",
  cap effectiveness at 3.
- Prefer actionable procedures over declarative descriptions.
- Do not reward verbosity.

## Output
Return ONLY valid JSON with this schema:
{{
  "scores": {{
    "faithfulness": {{"score": 1, "reason": "..."}},
    "reusability": {{"score": 1, "reason": "..."}},
    "effectiveness": {{"score": 1, "reason": "..."}},
    "clarity": {{"score": 1, "reason": "..."}},
    "conciseness": {{"score": 1, "reason": "..."}}
  }},
  "summary": "..."
}}
"""


def compute_skill_quality_avg(judge_result: dict) -> float:
    """Compute Ctx2Skill-style average from parsed per-dimension scores."""
    scores = judge_result.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("judge_result must contain a scores object")

    raw_scores: list[float] = []
    for dimension in DIMENSIONS:
        item = scores.get(dimension)
        if not isinstance(item, dict):
            raise ValueError(f"missing score object for {dimension}")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError(f"{dimension} score must be a number from 1 to 5")
        numeric_score = float(score)
        if not math.isfinite(numeric_score) or numeric_score < 1 or numeric_score > 5:
            raise ValueError(f"{dimension} score must be a number from 1 to 5")
        raw_scores.append(numeric_score)
    return sum(raw_scores) / len(raw_scores) / 5 * 100


def demo_skill() -> str:
    return """---
name: product-description-writing
description: Use when writing product descriptions from a few style examples.
---

## Procedure
1. Identify the target audience, product category, and must-mention attributes.
2. Mirror the examples' reusable structure: opening value proposition, concrete
   differentiators, and a closing use-case cue.
3. Preserve the examples' level of detail without copying product names, prices,
   quantities, or case-specific claims.

## Self-check
- Required attributes are present.
- Tone and depth match the examples.
- No unsupported facts or copied example-specific details are introduced.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--skill-md", type=Path, default=None)
    parser.add_argument("--print-sources", action="store_true")
    args = parser.parse_args()

    if args.print_sources:
        print(json.dumps(SOURCE_NOTES, ensure_ascii=False, indent=2))
        return

    examples_text = load_examples(args.examples_jsonl)
    skill_md = args.skill_md.read_text(encoding="utf-8") if args.skill_md else demo_skill()
    print(build_skill_quality_prompt(examples_text=examples_text, skill_md=skill_md))


if __name__ == "__main__":
    main()
