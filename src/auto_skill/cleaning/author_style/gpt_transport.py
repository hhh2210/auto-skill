"""Codex Responses transport helpers for author-style cleaning."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from auto_skill.cleaning.author_style.common import CODEX_RESPONSES_URL


def load_codex_auth(path: Path) -> tuple[str, str | None]:
    with path.open("r", encoding="utf-8") as handle:
        auth = json.load(handle)
    tokens = auth.get("tokens") or {}
    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id")
    if not access_token:
        raise RuntimeError(f"No access token found in {path}")
    return str(access_token), str(account_id) if account_id else None


def codex_response_text(
    prompt: str,
    *,
    model: str,
    auth_path: Path,
    instructions: str,
    service_tier: str | None,
    reasoning_effort: str | None = None,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any] | None]:
    access_token, account_id = load_codex_auth(auth_path)
    body: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": prompt}],
        "stream": True,
        "store": False,
    }
    if service_tier:
        body["service_tier"] = service_tier
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if account_id:
        headers["OpenAI-Account-ID"] = account_id
    request = urllib.request.Request(
        CODEX_RESPONSES_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    text = ""
    usage = None
    try:
        response_context = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Codex responses HTTP {exc.code}: {detail[:500]}") from exc
    with response_context as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            typ = event.get("type")
            if typ == "response.output_text.delta":
                text += event.get("delta", "")
            elif typ == "response.completed":
                usage = (event.get("response") or {}).get("usage")
    return text, usage
