"""Shared helpers for author-style compression diagnostics."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from auto_skill.cleaning.author_style.eval_summary import (
    latest_eval_cells,
    row_score,
    row_win_rate,
    sign,
)

TAG_PATTERNS = {
    "abrupt_topic_shifts": ["abrupt topic", "topic shift", "non sequitur", "unrelated"],
    "caps_emphasis": ["all-caps", "uppercase", "caps emphasis", "caps"],
    "casual_conversational": ["casual", "conversational", "informal"],
    "conversational_contractions": ["contraction", "don't", "can't", "i'm", "it's"],
    "direct_address": ["direct address", "second-person", "second person", "you"],
    "ellipsis_heavy": ["ellipsis", "..."],
    "emoticon_laughter": ["emoticon", "emoji", "lol", "haha", "laughter"],
    "exclamation_heavy": ["exclamation", "!"],
    "first_person_diary": [
        "first-person",
        "first person",
        "personal anecdote",
        "diary",
        "confessional",
    ],
    "hedging": ["hedg", "maybe", "probably", "kind of", "sort of"],
    "long_run_on": ["run-on", "long sentence", "breathless", "rambling"],
    "nonstandard_orthography": ["spelling", "typo", "orthograph", "elongation", "nonstandard"],
    "parenthetical_asides": ["parenthet", "aside", "("],
    "profanity": ["profan", "swear", "curse", "fuck", "shit"],
    "question_heavy": ["question", "?"],
    "sarcasm_snark": ["sarcas", "snark", "dry humor", "mocking"],
    "short_sentences": ["short sentence", "short, punchy", "brief", "fragment"],
    "supportive_polite": ["supportive", "polite", "encourag", "reassur"],
}


def pack_id(row: dict[str, Any]) -> str | None:
    value = row.get("pack_id")
    return value if isinstance(value, str) and value else None


def normalize_tag(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return text or None


def compact_counts(counter: Counter[str], *, limit: int = 20) -> dict[str, int]:
    return {tag: count for tag, count in counter.most_common(limit)}


def list_field(row: dict[str, Any], path: tuple[str, ...]) -> list[Any]:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return []
        current = current.get(key)
    return current if isinstance(current, list) else []


def judge_tags(row: dict[str, Any], field: str) -> list[str]:
    tags = []
    for value in list_field(row, ("judge_report", field)):
        normalized = normalize_tag(value)
        if normalized:
            tags.append(normalized)
    return sorted(set(tags))


def canonical_tags_in_text(text: str, candidate_tags: list[str]) -> set[str]:
    lowered = text.lower()
    hits: set[str] = set()
    for tag in candidate_tags:
        tag_text = tag.replace("_", " ")
        patterns = TAG_PATTERNS.get(tag, []) + [tag, tag_text]
        if any(pattern and pattern in lowered for pattern in patterns):
            hits.add(tag)
    return hits


def canonical_tags_from_judge_tags(tags: list[str], candidate_tags: list[str]) -> set[str]:
    return canonical_tags_in_text(" ".join(tags).replace("_", " "), candidate_tags)


def row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("pack_id")), str(row.get("task_id")), str(row.get("mode"))): row
        for row in latest_eval_cells(rows)
        if row.get("status") == "success"
        and isinstance(row.get("pack_id"), str)
        and isinstance(row.get("task_id"), str)
        and isinstance(row.get("mode"), str)
    }


def row_topic_risk(row: dict[str, Any]) -> float | None:
    report = row.get("judge_report")
    if not isinstance(report, dict):
        return None
    value = report.get("topic_shortcut_risk_1_to_5")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def generation_shape_stats(row: dict[str, Any]) -> dict[str, Any]:
    generation = row.get("generation")
    text = ""
    if isinstance(generation, dict) and isinstance(generation.get("text"), str):
        text = generation["text"]
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    sentence_marks = sum(text.count(mark) for mark in ".!?")
    return {
        "chars": len(text),
        "words": len(re.findall(r"[A-Za-z0-9']+", text)),
        "nonempty_lines": len(nonempty_lines),
        "paragraphs": len([chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]),
        "sentence_mark_count": sentence_marks,
        "question_marks": text.count("?"),
        "exclamation_marks": text.count("!"),
        "ellipsis_count": text.count("..."),
        "parenthetical_open_count": text.count("("),
    }


def skill_mode_for_eval_mode(mode: str) -> str | None:
    if mode in {"one_shot_skill_from_examples"}:
        return "one_shot_skill_from_examples"
    if mode in {"auto_skill_feature_driven", "examples_plus_feature_skill"}:
        return "auto_skill_feature_driven_no_validation"
    if mode in {"auto_skill_operational", "examples_plus_operational_skill"}:
        return "auto_skill_feature_operational_no_validation"
    return None


def skill_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        current_pack_id = pack_id(row)
        mode = row.get("mode")
        skill_md = row.get("skill_md")
        if (
            current_pack_id
            and isinstance(mode, str)
            and isinstance(skill_md, str)
            and skill_md.strip()
        ):
            indexed[(current_pack_id, mode)] = row
    return indexed


def pack_profile_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {value: row for row in rows if (value := pack_id(row))}


def profile_tags(profile: dict[str, Any], key: str) -> list[str]:
    values = profile.get(key)
    if not isinstance(values, list):
        return []
    return sorted(str(value) for value in values if isinstance(value, str) and value)


def skill_texts(row: dict[str, Any] | None) -> dict[str, str]:
    if row is None:
        return {"skill_md": "", "feature_pipeline": ""}
    skill_md = row.get("skill_md") if isinstance(row.get("skill_md"), str) else ""
    feature_reports = row.get("feature_reports")
    cross_example_report = row.get("cross_example_report")
    feature_payload = {
        "feature_reports": feature_reports if isinstance(feature_reports, list) else [],
        "cross_example_report": (
            cross_example_report if isinstance(cross_example_report, dict) else {}
        ),
    }
    return {
        "skill_md": skill_md,
        "feature_pipeline": json.dumps(feature_payload, ensure_ascii=False),
    }


def model_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "solver_model",
        "solver_backend",
        "judge_model",
        "judge_backend",
        "judge_config_prefix",
        "judge_config_model",
    ]
    return {
        key: dict(
            Counter(
                str(row.get(key) if row.get(key) is not None else "null")
                for row in rows
            ).most_common()
        )
        for key in keys
    }


def loss_bucket(delta: float | None) -> str:
    if delta is None:
        return "missing_score"
    if delta <= -2:
        return "severe_loss"
    if delta <= -1:
        return "moderate_loss"
    if delta < 0:
        return "small_loss"
    if delta == 0:
        return "tie"
    return "positive"


def score_snapshot(
    prompt: dict[str, Any] | None,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | None]:
    return {
        "baseline_score": row_score(baseline),
        "candidate_score": row_score(candidate),
        "prompt_score": row_score(prompt) if prompt is not None else None,
        "baseline_win": row_win_rate(baseline),
        "candidate_win": row_win_rate(candidate),
        "prompt_win": row_win_rate(prompt) if prompt is not None else None,
    }


def sign_counts(values: list[float]) -> dict[str, int]:
    return dict(Counter(sign(value) for value in values))
