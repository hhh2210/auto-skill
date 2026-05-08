#!/usr/bin/env python3
"""Run upstream PresentBench official judge with repo-local preflight checks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODE_RESULT_ROOTS = [
    "prompt_only=../PresentBench/results/prompt_only",
    "auto_skill=../PresentBench/results/auto_skill",
]


def parse_mode_result_roots(raw_values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"expected MODE=PATH, got {raw!r}")
        mode, path = raw.split("=", 1)
        mode = mode.strip()
        path = path.strip()
        if not mode or not path:
            raise ValueError(f"expected non-empty MODE=PATH, got {raw!r}")
        roots[mode] = Path(path)
    return roots


def build_judge_all_command(
    *,
    python_executable: str,
    judge_all: Path,
    agent_name: str,
    data_root: Path,
    result_root: Path,
    api_type: str,
    model: str,
    max_workers: int | None,
    thinking_level: str | None,
    min_timestamp: str | None,
) -> list[str]:
    command = [
        python_executable,
        str(judge_all),
        "--agent_name",
        agent_name,
        "--data_root",
        str(data_root),
        "--result_root",
        str(result_root),
        "--api_type",
        api_type,
        "--model",
        model,
    ]
    if max_workers is not None:
        command.extend(["--max_workers", str(max_workers)])
    if thinking_level:
        command.extend(["--thinking_level", thinking_level])
    if min_timestamp:
        command.extend(["--min_timestamp", min_timestamp])
    return command


def validate_preflight(
    *,
    code_root: Path,
    data_root: Path,
    mode_roots: dict[str, Path],
    api_type: str,
    allow_missing_env: bool,
) -> list[str]:
    errors: list[str] = []
    if not (code_root / "judge_all.py").exists():
        errors.append(f"missing upstream judge_all.py under {code_root}")
    if not data_root.exists():
        errors.append(f"PresentBench data root does not exist: {data_root}")
    for mode, result_root in mode_roots.items():
        if not result_root.exists():
            errors.append(f"{mode}: result root does not exist: {result_root}")
    if api_type.startswith("gemini") and not os.getenv("GENAI_API_KEY"):
        message = "GENAI_API_KEY is required by upstream PresentBench gemini judge"
        if allow_missing_env:
            print(f"warning: {message}", file=sys.stderr)
        else:
            errors.append(message)
    return errors


def subprocess_env_with_code_root(code_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    code_path = str(code_root.resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = code_path if not existing else f"{code_path}{os.pathsep}{existing}"
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--code-root", type=Path, default=Path("data/PresentBench_code"))
    parser.add_argument("--data-root", type=Path, default=Path("data/PresentBench_repo"))
    parser.add_argument(
        "--mode-result-root",
        action="append",
        default=None,
        help="MODE=PATH result root. Repeat for each official mode.",
    )
    parser.add_argument("--api-type", default="gemini")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--thinking-level")
    parser.add_argument("--min-timestamp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-env",
        action="store_true",
        help="Permit dry-run command rendering without GENAI_API_KEY.",
    )
    args = parser.parse_args()

    if args.allow_missing_env and not args.dry_run:
        print("error: --allow-missing-env is only valid with --dry-run", file=sys.stderr)
        return 2

    load_dotenv(args.env_file, override=False)
    try:
        mode_roots = parse_mode_result_roots(args.mode_result_root or DEFAULT_MODE_RESULT_ROOTS)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    errors = validate_preflight(
        code_root=args.code_root,
        data_root=args.data_root,
        mode_roots=mode_roots,
        api_type=args.api_type,
        allow_missing_env=args.allow_missing_env,
    )
    judge_all = args.code_root / "judge_all.py"
    commands = [
        build_judge_all_command(
            python_executable=sys.executable,
            judge_all=judge_all,
            agent_name=mode,
            data_root=args.data_root,
            result_root=result_root,
            api_type=args.api_type,
            model=args.model,
            max_workers=args.max_workers,
            thinking_level=args.thinking_level,
            min_timestamp=args.min_timestamp,
        )
        for mode, result_root in mode_roots.items()
    ]

    for command in commands:
        print(" ".join(command))

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    env = subprocess_env_with_code_root(args.code_root)
    for command in commands:
        completed = subprocess.run(command, check=False, env=env)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
