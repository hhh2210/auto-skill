"""Private prompt formatting helpers."""

from __future__ import annotations

import re
from typing import Any


def _append_bullets(sections: list[str], title: str, items: list[str]) -> None:
    if not items:
        return
    sections.extend(["", f"## {title}"])
    sections.extend(f"- {item}" for item in items)


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(_single_line(item))
        elif isinstance(item, dict):
            text = item.get("feature") or item.get("rule") or item.get("description")
            if isinstance(text, str) and text.strip():
                items.append(_single_line(text))
        if len(items) >= limit:
            break
    return items


def _candidate_rule_lines(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    lines = []
    for item in value:
        if isinstance(item, str):
            rule = item
            support = None
        elif isinstance(item, dict):
            rule = item.get("rule")
            support = item.get("support_count")
            if not isinstance(support, int):
                supporting = item.get("supporting_examples")
                support = len(supporting) if isinstance(supporting, list) else None
        else:
            continue
        if not isinstance(rule, str) or not rule.strip():
            continue
        suffix = f" (support={support})" if isinstance(support, int) else ""
        lines.append(_single_line(rule) + suffix)
        if len(lines) >= limit:
            break
    return lines


def _described_item_lines(value: Any, *, label_key: str, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    lines = []
    for item in value:
        if isinstance(item, str) and item.strip():
            lines.append(_single_line(item))
        elif isinstance(item, dict):
            label = item.get(label_key)
            description = item.get("description") or item.get("reason")
            if isinstance(description, str) and description.strip():
                prefix = f"{label}: " if isinstance(label, str) and label.strip() else ""
                lines.append(prefix + _single_line(description))
        if len(lines) >= limit:
            break
    return lines


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _humanize_key(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        normalized = value.casefold()
        if not value or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value)
    return output
