#!/usr/bin/env python3
"""Run minimal auto-skill induction with one MIMO supervisor pass."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.mvp import parse_json_object, user_examples_from_pack  # noqa: E402
from auto_skill.skill_induction_minimal import (  # noqa: E402
    MinimalInductionError,
    induce_minimal,
    validate_supervisor_report,
)
from auto_skill.skill_memory import load_memory_for_pack  # noqa: E402


@dataclass
class MockResult:
    text: str
    model: str
    finish_reason: str | None = "stop"
    usage: dict[str, int] | None = None
    request_id: str | None = "dry-run"


class DryRunClient:
    def __init__(self, model: str, supervisor: bool = False):
        self.model = model
        self.supervisor = supervisor

    def complete(self, messages: list[dict[str, str]], **_: Any) -> MockResult:
        if self.supervisor:
            text = json.dumps(
                {
                    "artifact_critique": {
                        "too_generic": [],
                        "over_specific": [],
                        "missing": [],
                    },
                    "rule_grounding": [],
                }
            )
        else:
            text = "# Dry Run Skill\n\nUse the visible examples to infer reusable rules."
        return MockResult(text=text, model=self.model, usage={"total_tokens": 0})


class ClientAdapter:
    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        temperature: float | None,
        max_tokens: int,
        enable_thinking: bool | None,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    def complete(self, messages: list[dict[str, str]], **_: Any) -> Any:
        return self.client.complete(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )


class ParseRetrySupervisor:
    def __init__(self, client: Any, max_attempts: int):
        self.client = client
        self.max_attempts = max(1, max_attempts)

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        last = None
        for _ in range(self.max_attempts):
            last = self.client.complete(messages, **kwargs)
            parsed = parse_json_object(last.text)
            if "parse_error" in parsed:
                continue
            try:
                validate_supervisor_report(parsed)
            except MinimalInductionError:
                continue
            else:
                return last
        return last


def existing_successes(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["pack_id"])
        for row in rows
        if row.get("status") == "success"
        and row.get("mode") == "auto_skill_minimal"
        and isinstance(row.get("skill_md"), str)
        and row["skill_md"].strip()
    }


def error_row(pack_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": "skill-induction/v1",
        "status": "induction_error",
        "pack_id": pack_id,
        "mode": "auto_skill_minimal",
        "skill_md": None,
        "model_calls": [],
        "solver_model": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def build_clients(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.dry_run:
        return DryRunClient("dry-run-solver"), ParseRetrySupervisor(
            DryRunClient("dry-run-mimo", supervisor=True), args.parse_max_attempts
        )
    solver_config = ChatCompletionConfig.from_env(prefix=args.solver_config_prefix)
    supervisor_config = ChatCompletionConfig.from_env(prefix=args.supervisor_config_prefix)
    enable_thinking = False if args.no_enable_thinking else None
    solver = ClientAdapter(
        ChatCompletionClient(solver_config),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=enable_thinking,
    )
    supervisor = ClientAdapter(
        ChatCompletionClient(supervisor_config),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=enable_thinking,
    )
    return solver, ParseRetrySupervisor(supervisor, args.parse_max_attempts)


def run(args: argparse.Namespace) -> int:
    packs = [pack for pack in load_jsonl(args.packs) if user_examples_from_pack(pack)]
    rows = load_jsonl(args.out) if args.resume and args.out.exists() else []
    successes = existing_successes(rows)
    solver, supervisor = build_clients(args)

    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        if args.resume and pack_id in successes:
            continue
        try:
            memory = (
                load_memory_for_pack(args.memory, pack_id, args.memory_scope)
                if args.memory is not None
                else []
            )
            rows.append(
                induce_minimal(
                    pack,
                    solver_client=solver,
                    supervisor_client=supervisor,
                    memory=memory,
                    scope=args.memory_scope,
                )
            )
            write_jsonl(args.out, rows)
        except Exception as exc:  # noqa: BLE001
            rows.append(error_row(pack_id, exc))
            write_jsonl(args.out, rows)
            if not args.allow_partial:
                raise
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory", type=Path)
    parser.add_argument(
        "--memory-scope",
        choices=("within_pack", "cross_pack"),
        default="within_pack",
    )
    parser.add_argument("--solver-config-prefix", default="BAILIAN")
    parser.add_argument("--supervisor-config-prefix", default="MIMO")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--parse-max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        return run(parse_args())
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
