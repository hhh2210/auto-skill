"""Literal leakage helpers for self-consistency diagnostics."""

from __future__ import annotations

import json
import re
from typing import Any

from auto_skill.mvp import user_examples_from_pack


def _json_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _literal_candidates_from_text(text: str) -> set[str]:
    candidates: set[str] = set()
    if not text:
        return candidates
    patterns = [
        r"\b\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d+(?:\.\d+)?%\b",
        r"\b\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?){1,4}\b",
        r"\b\d+(?:\.\d+)?\s*(?:元|万元|亿元|USD|RMB|dollars?|slides?|pages?)\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"https?://[^\s)]+",
        r"(?:^|[\s\"'`])(?:data|artifacts|runs|notes|docs|src|scripts|/Users)/[^\s\"'`)]+",
        r"\b(?:[A-Z][A-Za-z0-9&.-]+(?:\s+|[-:])){2,}[A-Z][A-Za-z0-9&.-]+\b",
        r"[\u4e00-\u9fffA-Za-z0-9]+(?:报告|论文|白皮书|合同|招标书|演示文稿|讲义|教材|章节|公司|大学|学院)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            literal = match.group(0).strip(" \t\n\"'`.,;:()[]{}")
            if len(literal) >= 3:
                candidates.add(literal)
    return candidates


def train_literal_candidates(pack: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for example in user_examples_from_pack(pack):
        candidates.add(example.example_id)
        source_task_id = example.metadata.get("source_task_id")
        if source_task_id:
            candidates.add(str(source_task_id))
        candidates.update(_literal_candidates_from_text(example.task_input))
        if example.output:
            candidates.update(_literal_candidates_from_text(example.output))
        candidates.update(str(material) for material in example.materials if str(material))
    return {item for item in candidates if len(item) >= 3}


def literal_leakage_report(
    *,
    pack: dict[str, Any],
    generated_signatures: Any,
) -> dict[str, Any]:
    """Detect sample-specific literals copied into abstract signatures."""

    haystack = _json_blob(generated_signatures)
    matches = sorted(
        literal for literal in train_literal_candidates(pack) if literal and literal in haystack
    )
    return {
        "candidate_literal_count": len(train_literal_candidates(pack)),
        "matched_literal_count": len(matches),
        "matched_literals": matches[:50],
        "has_literal_leakage": bool(matches),
    }
