"""Parsing helpers for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in an LLM response."""

    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match is not None:
        candidates.insert(0, fence_match.group(1).strip())

    last_error = "no_json_object_found"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if match is None:
                continue
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as nested_exc:
                last_error = str(nested_exc)
                continue
        if isinstance(parsed, dict):
            return parsed
        return {"raw_text": text, "parse_error": "json_root_is_not_object"}
    return {"raw_text": text, "parse_error": last_error}
