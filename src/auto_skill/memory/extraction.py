"""Persist public example-derived lessons from skill extraction runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_skill.example_packs import load_jsonl, write_jsonl

SCHEMA_VERSION = "skill-extraction-memory/v1"
EVIDENCE_SOURCE = "user_examples"
DERIVATION_FEATURE_REPORTS = "feature_reports"
DERIVATION_MINIMAL_SUPERVISOR_REPORT = "minimal_supervisor_report"
VALID_DERIVATIONS = {
    DERIVATION_FEATURE_REPORTS,
    DERIVATION_MINIMAL_SUPERVISOR_REPORT,
}


@dataclass(frozen=True)
class ExtractionMemoryEntry:
    memory_id: str
    pack_id: str
    mode: str
    lesson_kind: str
    lesson: str
    evidence_examples: tuple[str, ...]
    derivation: str = DERIVATION_FEATURE_REPORTS
    evidence_source: str = EVIDENCE_SOURCE
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "pack_id": self.pack_id,
            "mode": self.mode,
            "lesson_kind": self.lesson_kind,
            "lesson": self.lesson,
            "evidence_examples": list(self.evidence_examples),
            "derivation": self.derivation,
            "evidence_source": self.evidence_source,
        }


def memory_id_for(
    *,
    pack_id: str,
    mode: str,
    lesson_kind: str,
    lesson: str,
    evidence_examples: tuple[str, ...],
    derivation: str = DERIVATION_FEATURE_REPORTS,
) -> str:
    payload = json.dumps(
        {
            "pack_id": pack_id,
            "mode": mode,
            "lesson_kind": lesson_kind,
            "lesson": lesson,
            "evidence_examples": sorted(evidence_examples),
            "derivation": derivation,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("rule"), str):
            return value["rule"].strip()
        feature = value.get("feature")
        description = value.get("description") or value.get("reason")
        if isinstance(feature, str) and isinstance(description, str):
            return f"{feature.strip()}: {description.strip()}"
        if isinstance(description, str):
            return description.strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _evidence_examples(value: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, dict):
        for key in ("supporting_example_ids", "supported_examples", "evidence_examples"):
            raw = value.get(key)
            if isinstance(raw, list):
                examples = tuple(str(item) for item in raw if str(item))
                if examples:
                    return examples
        if isinstance(value.get("example_id"), str):
            return (value["example_id"],)
    return fallback


def _entry(
    *,
    pack_id: str,
    mode: str,
    lesson_kind: str,
    lesson: str,
    evidence_examples: tuple[str, ...],
    derivation: str = DERIVATION_FEATURE_REPORTS,
) -> ExtractionMemoryEntry | None:
    lesson = lesson.strip()
    if not pack_id or not mode or not lesson:
        return None
    evidence_examples = tuple(sorted(set(evidence_examples)))
    memory_id = memory_id_for(
        pack_id=pack_id,
        mode=mode,
        lesson_kind=lesson_kind,
        lesson=lesson,
        evidence_examples=evidence_examples,
        derivation=derivation,
    )
    return ExtractionMemoryEntry(
        memory_id=memory_id,
        pack_id=pack_id,
        mode=mode,
        lesson_kind=lesson_kind,
        lesson=lesson,
        evidence_examples=evidence_examples,
        derivation=derivation,
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _minimal_supervisor_report(row: dict[str, Any]) -> dict[str, Any] | None:
    report = row.get("supervisor_report")
    if isinstance(report, dict):
        return report
    if isinstance(report, str):
        try:
            parsed = json.loads(report)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _example_ids_from_grounding(item: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = item.get(key)
    if not isinstance(raw, list):
        return ()
    return tuple(str(example_id) for example_id in raw if str(example_id))


def entries_from_minimal_skill_row(row: dict[str, Any]) -> list[ExtractionMemoryEntry]:
    """Extract memory from the minimal MIMO supervisor report.

    The report is produced from the candidate skill and user-visible examples
    only, so these entries remain public extraction memory rather than eval
    feedback.
    """

    pack_id = str(row.get("pack_id") or "")
    mode = str(row.get("mode") or "")
    report = _minimal_supervisor_report(row)
    if report is None:
        return []

    entries: list[ExtractionMemoryEntry] = []
    critique = report.get("artifact_critique")
    if isinstance(critique, dict):
        for item in _as_list(critique.get("over_specific")):
            entry = _entry(
                pack_id=pack_id,
                mode=mode,
                lesson_kind="do_not_generalize",
                lesson=_clean_text(item),
                evidence_examples=(),
                derivation=DERIVATION_MINIMAL_SUPERVISOR_REPORT,
            )
            if entry is not None:
                entries.append(entry)

    for item in _as_list(report.get("rule_grounding")):
        if not isinstance(item, dict):
            continue
        rule = _clean_text(item.get("rule"))
        supported_by = _example_ids_from_grounding(item, "supported_by")
        contradicted_by = _example_ids_from_grounding(item, "contradicted_by")
        out_of_scope = _example_ids_from_grounding(item, "out_of_scope")
        if supported_by:
            entry = _entry(
                pack_id=pack_id,
                mode=mode,
                lesson_kind="candidate_rule",
                lesson=rule,
                evidence_examples=supported_by,
                derivation=DERIVATION_MINIMAL_SUPERVISOR_REPORT,
            )
            if entry is not None:
                entries.append(entry)
        if contradicted_by:
            entry = _entry(
                pack_id=pack_id,
                mode=mode,
                lesson_kind="conflict",
                lesson=rule,
                evidence_examples=contradicted_by,
                derivation=DERIVATION_MINIMAL_SUPERVISOR_REPORT,
            )
            if entry is not None:
                entries.append(entry)
        if out_of_scope:
            entry = _entry(
                pack_id=pack_id,
                mode=mode,
                lesson_kind="outlier",
                lesson=rule,
                evidence_examples=out_of_scope,
                derivation=DERIVATION_MINIMAL_SUPERVISOR_REPORT,
            )
            if entry is not None:
                entries.append(entry)
    return dedupe_memory_entries(entries)


def entries_from_skill_row(row: dict[str, Any]) -> list[ExtractionMemoryEntry]:
    """Extract memory entries from one successful skill row.

    This intentionally consumes only artifacts produced from user examples. It
    does not read private heldout rubrics, evaluator feedback, or scores.
    """

    if row.get("status") != "success":
        return []
    if row.get("mode") == "auto_skill_minimal":
        return entries_from_minimal_skill_row(row)
    pack_id = str(row.get("pack_id") or "")
    mode = str(row.get("mode") or "")
    report = row.get("cross_example_report")
    feature_reports = row.get("feature_reports") or []
    fallback_examples = tuple(
        str(item.get("example_id"))
        for item in feature_reports
        if isinstance(item, dict) and item.get("example_id")
    )
    entries: list[ExtractionMemoryEntry] = []
    if isinstance(report, dict):
        for key, lesson_kind in (
            ("stable_features", "stable_feature"),
            ("optional_features", "optional_feature"),
            ("conflicts", "conflict"),
            ("outliers", "outlier"),
            ("candidate_rules", "candidate_rule"),
        ):
            raw_items = report.get(key) or []
            if not isinstance(raw_items, list):
                continue
            for item in raw_items:
                entry = _entry(
                    pack_id=pack_id,
                    mode=mode,
                    lesson_kind=lesson_kind,
                    lesson=_clean_text(item),
                    evidence_examples=_evidence_examples(item, fallback_examples),
                )
                if entry is not None:
                    entries.append(entry)
    for feature_report in feature_reports:
        if not isinstance(feature_report, dict):
            continue
        example_id = str(feature_report.get("example_id") or "")
        for item in feature_report.get("must_not_generalize") or []:
            entry = _entry(
                pack_id=pack_id,
                mode=mode,
                lesson_kind="do_not_generalize",
                lesson=_clean_text(item),
                evidence_examples=(example_id,) if example_id else fallback_examples,
            )
            if entry is not None:
                entries.append(entry)
    return dedupe_memory_entries(entries)


def entries_from_skill_rows(rows: list[dict[str, Any]]) -> list[ExtractionMemoryEntry]:
    entries: list[ExtractionMemoryEntry] = []
    for row in rows:
        entries.extend(entries_from_skill_row(row))
    return dedupe_memory_entries(entries)


def dedupe_memory_entries(entries: list[ExtractionMemoryEntry]) -> list[ExtractionMemoryEntry]:
    by_id: dict[str, ExtractionMemoryEntry] = {}
    for entry in entries:
        by_id.setdefault(entry.memory_id, entry)
    return [by_id[key] for key in sorted(by_id)]


def load_memory_entries(path: Path) -> list[ExtractionMemoryEntry]:
    entries = []
    for row in load_jsonl(path):
        if row.get("schema_version") != SCHEMA_VERSION:
            continue
        entries.append(
            ExtractionMemoryEntry(
                memory_id=str(row.get("memory_id") or ""),
                pack_id=str(row.get("pack_id") or ""),
                mode=str(row.get("mode") or ""),
                lesson_kind=str(row.get("lesson_kind") or ""),
                lesson=str(row.get("lesson") or ""),
                evidence_examples=tuple(str(item) for item in row.get("evidence_examples") or []),
                derivation=str(row.get("derivation") or DERIVATION_FEATURE_REPORTS),
                evidence_source=str(row.get("evidence_source") or EVIDENCE_SOURCE),
            )
        )
    return dedupe_memory_entries(entries)


def write_memory_entries(path: Path, entries: list[ExtractionMemoryEntry]) -> None:
    write_jsonl(path, [entry.to_json() for entry in dedupe_memory_entries(entries)])


def summarize_memory_entries(entries: list[ExtractionMemoryEntry]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_pack: dict[str, int] = {}
    for entry in entries:
        by_kind[entry.lesson_kind] = by_kind.get(entry.lesson_kind, 0) + 1
        by_pack[entry.pack_id] = by_pack.get(entry.pack_id, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "entries": len(entries),
        "lesson_kinds": dict(sorted(by_kind.items())),
        "packs": dict(sorted(by_pack.items())),
    }
