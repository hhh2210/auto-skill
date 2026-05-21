"""JSONL helpers for auto-skill artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_jsonl_lenient_final_line(path: Path) -> tuple[list[Any], list[str]]:
    """Read JSONL while tolerating one malformed non-empty final line.

    Append-only logs can be interrupted between writing bytes and flushing a
    newline. Earlier malformed lines still raise ``JSONDecodeError`` because
    they indicate durable corruption rather than an interrupted final append.
    """

    rows: list[Any] = []
    warnings: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    last_non_empty = non_empty_indexes[-1] if non_empty_indexes else -1
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == last_non_empty:
                warnings.append(
                    f"ignored malformed final JSONL line in {path}: "
                    f"line {index + 1}: {exc}"
                )
                continue
            raise
    return rows, warnings


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
