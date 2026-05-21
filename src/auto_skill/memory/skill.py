"""Read public extraction memory for minimal skill induction prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from auto_skill.example_packs import load_jsonl
from auto_skill.memory.extraction import (
    DERIVATION_FEATURE_REPORTS,
    EVIDENCE_SOURCE,
    SCHEMA_VERSION,
    ExtractionMemoryEntry,
)

MemoryPolarity = Literal["positive", "negative"]
MemoryScope = Literal["within_pack", "cross_pack", "cross_pack_holdout"]

POSITIVE_KINDS = {"stable_feature", "candidate_rule", "optional_feature"}
NEGATIVE_KINDS = {"do_not_generalize", "outlier", "conflict"}


@dataclass(frozen=True)
class SkillMemoryEntry(ExtractionMemoryEntry):
    polarity: MemoryPolarity = "positive"

    def to_json(self) -> dict[str, object]:
        row = super().to_json()
        row["polarity"] = self.polarity
        return row


def polarity_for_lesson_kind(lesson_kind: str) -> MemoryPolarity:
    if lesson_kind in NEGATIVE_KINDS:
        return "negative"
    return "positive"


def _entry_from_row(row: dict[str, object]) -> SkillMemoryEntry | None:
    if row.get("schema_version") != SCHEMA_VERSION:
        return None
    if row.get("evidence_source") != EVIDENCE_SOURCE:
        return None
    lesson_kind = str(row.get("lesson_kind") or "")
    if lesson_kind not in POSITIVE_KINDS | NEGATIVE_KINDS:
        return None
    raw_polarity = row.get("polarity")
    polarity: MemoryPolarity
    if raw_polarity in {"positive", "negative"}:
        polarity = raw_polarity  # type: ignore[assignment]
    else:
        polarity = polarity_for_lesson_kind(lesson_kind)
    return SkillMemoryEntry(
        memory_id=str(row.get("memory_id") or ""),
        pack_id=str(row.get("pack_id") or ""),
        mode=str(row.get("mode") or ""),
        lesson_kind=lesson_kind,
        lesson=str(row.get("lesson") or ""),
        evidence_examples=tuple(str(item) for item in row.get("evidence_examples") or []),
        derivation=str(row.get("derivation") or DERIVATION_FEATURE_REPORTS),
        evidence_source=str(row.get("evidence_source") or EVIDENCE_SOURCE),
        polarity=polarity,
    )


def load_memory_for_pack(path: Path, pack_id: str, scope: MemoryScope) -> list[SkillMemoryEntry]:
    if not path.exists():
        return []
    entries = [entry for row in load_jsonl(path) if (entry := _entry_from_row(row)) is not None]
    if scope == "within_pack":
        entries = [entry for entry in entries if entry.pack_id == pack_id]
    elif scope == "cross_pack_holdout":
        entries = [entry for entry in entries if entry.pack_id != pack_id]
    return entries


def select_memory_entries(
    entries: list[SkillMemoryEntry], max_entries: int | None
) -> list[SkillMemoryEntry]:
    if max_entries is None or max_entries <= 0 or len(entries) <= max_entries:
        return entries
    return sorted(
        entries,
        key=lambda entry: (
            -len(entry.evidence_examples),
            entry.polarity != "positive",
            entry.pack_id,
            entry.lesson_kind,
            entry.memory_id,
        ),
    )[:max_entries]


def _format_memory_block(
    entries: list[SkillMemoryEntry], title: str, polarity: MemoryPolarity
) -> str:
    lines = [
        f"- [{entry.pack_id}/{entry.lesson_kind}] {entry.lesson}"
        for entry in entries
        if entry.polarity == polarity
    ]
    return f"{title}:\n" + ("\n".join(lines) if lines else "- None")


def format_memory_for_prompt(
    entries: list[SkillMemoryEntry], polarity: MemoryPolarity | None = None
) -> str:
    if polarity == "positive":
        return _format_memory_block(entries, "Positive priors", "positive")
    if polarity == "negative":
        return _format_memory_block(entries, "Negative warnings", "negative")
    if not entries:
        return "No prior extraction memory."
    blocks = []
    for title, polarity in (
        ("Positive priors", "positive"),
        ("Negative warnings", "negative"),
    ):
        blocks.append(_format_memory_block(entries, title, polarity))
    return "\n\n".join(blocks)
