#!/usr/bin/env python3
"""Run upstream PresentBench official judge with repo-local preflight checks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.longrun import run_subprocesses  # noqa: E402
from auto_skill.presentbench_eval import (  # noqa: E402
    check_presentbench_official_eval_readiness,
    parse_mode_path_mappings,
)
from scripts.eval.run_heldout_eval import select_packs  # noqa: E402

DEFAULT_MODE_RESULT_ROOTS = [
    "prompt_only=../PresentBench/results/prompt_only",
    "auto_skill=../PresentBench/results/auto_skill",
]

CHAT_JUDGE_SCRIPT = REPO_ROOT / "scripts" / "eval" / "run_presentbench_chat_judge.py"
CHAT_JUDGE_API_TYPES = {"openai"}
GEMINI_API_TYPES = {"gemini", "gemini_inline"}


def parse_mode_result_roots(raw_values: list[str]) -> dict[str, Path]:
    return parse_mode_path_mappings(raw_values, option_label="--mode-result-root")


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


def resolve_judge_script(*, code_root: Path, api_type: str) -> Path:
    """Pick which judge entrypoint handles ``api_type`` (upstream vs in-repo chat)."""

    if api_type in CHAT_JUDGE_API_TYPES:
        return CHAT_JUDGE_SCRIPT
    return code_root / "judge.py"


def build_judge_command(
    *,
    python_executable: str,
    judge: Path,
    readiness: dict,
    api_type: str,
    model: str,
    retry: int,
    thinking_level: str | None,
    min_timestamp: str | None,
) -> list[str]:
    source_parts = Path(readiness["source_task_id"]).parts
    case_dir = Path(readiness["case_dir"])
    if not source_parts:
        raise ValueError("readiness source_task_id is empty")
    data_root = case_dir.parents[len(source_parts) - 1]
    domain_dir = data_root / source_parts[0]
    command = [
        python_executable,
        str(judge),
        "--api_type",
        api_type,
        "--model",
        model,
        "--slides",
        readiness["slide_artifact"] or str(Path(readiness["expected_result_dir"]) / "slides.pdf"),
        "--judge_prompt",
        str(case_dir / "generation_task" / "judge_prompt.json"),
        "--common_judge_prompt",
        str(domain_dir / "common_judge_prompt.json"),
        "--weights_path",
        str(domain_dir / "judge_weights.yaml"),
        "--retry",
        str(retry),
    ]
    for material in readiness.get("material_files") or []:
        if "--material" not in command:
            command.append("--material")
        command.append(material)
    if readiness["status"] == "ready_for_zero_score":
        command.append("--zero_score")
    if thinking_level:
        command.extend(["--thinking_level", thinking_level])
    if min_timestamp:
        command.extend(["--min_timestamp", min_timestamp])
    return command


def _openai_judge_env_present() -> bool:
    base_url = os.getenv("GOOGLE_THIRD_API_URL")
    api_key = os.getenv("GOOGLE_THIRD_API_KEY")
    return bool(base_url) and bool(api_key)


OPENAI_JUDGE_ENV_HINT = (
    "GOOGLE_THIRD_API_URL+GOOGLE_THIRD_API_KEY are required for --api-type openai"
)


def validate_preflight(
    *,
    code_root: Path,
    data_root: Path,
    mode_roots: dict[str, Path],
    api_type: str,
    allow_missing_env: bool,
) -> list[str]:
    errors: list[str] = []
    if api_type in CHAT_JUDGE_API_TYPES:
        if not CHAT_JUDGE_SCRIPT.exists():
            errors.append(f"missing in-repo chat judge script: {CHAT_JUDGE_SCRIPT}")
    else:
        if not (code_root / "judge.py").exists():
            errors.append(f"missing upstream judge.py under {code_root}")
    if not data_root.exists():
        errors.append(f"PresentBench data root does not exist: {data_root}")
    for mode, result_root in mode_roots.items():
        if not result_root.exists():
            errors.append(f"{mode}: result root does not exist: {result_root}")
    if api_type in GEMINI_API_TYPES and not os.getenv("GENAI_API_KEY"):
        message = "GENAI_API_KEY is required by upstream PresentBench gemini judge"
        if allow_missing_env:
            print(f"warning: {message}", file=sys.stderr)
        else:
            errors.append(message)
    if api_type in CHAT_JUDGE_API_TYPES and not _openai_judge_env_present():
        if allow_missing_env:
            print(f"warning: {OPENAI_JUDGE_ENV_HINT}", file=sys.stderr)
        else:
            errors.append(OPENAI_JUDGE_ENV_HINT)
    return errors


def selected_judge_commands(
    *,
    packs_path: Path,
    code_root: Path,
    data_root: Path,
    mode_roots: dict[str, Path],
    python_executable: str,
    api_type: str,
    model: str,
    retry: int,
    thinking_level: str | None,
    min_timestamp: str | None,
    pack_ids: set[str] | None = None,
    limit_packs: int | None = None,
    limit_heldout: int | None = None,
) -> tuple[list[list[str]], list[str]]:
    packs = load_jsonl(packs_path)
    selected = select_packs(
        [pack for pack in packs if pack.get("source") == "PresentBench"],
        pack_ids=pack_ids,
        source=None,
        limit=limit_packs,
    )
    commands: list[list[str]] = []
    errors: list[str] = []
    selected_cells = 0
    judge = resolve_judge_script(code_root=code_root, api_type=api_type)
    runnable_statuses = {"ready_for_official_judge", "ready_for_zero_score"}
    for pack in selected:
        heldout_tasks = pack.get("heldout_tasks", [])
        if limit_heldout is not None:
            heldout_tasks = heldout_tasks[:limit_heldout]
        for task in heldout_tasks:
            for mode, result_root in mode_roots.items():
                selected_cells += 1
                readiness = check_presentbench_official_eval_readiness(
                    task=task,
                    data_root=data_root,
                    result_root=result_root,
                    code_root=code_root,
                    judge_model=model,
                ).to_json()
                if readiness["status"] == "scored":
                    continue
                if readiness["status"] not in runnable_statuses:
                    errors.append(
                        f"{pack['pack_id']}::{task['task_id']}::{mode}: "
                        f"{readiness['status']}"
                    )
                    continue
                commands.append(
                    build_judge_command(
                        python_executable=python_executable,
                        judge=judge,
                        readiness=readiness,
                        api_type=api_type,
                        model=model,
                        retry=retry,
                        thinking_level=thinking_level,
                        min_timestamp=min_timestamp,
                    )
                )
    if selected_cells == 0:
        errors.append("no selected PresentBench official judge cells")
    return commands, errors


def subprocess_env_with_code_root(code_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    code_path = str(code_root.resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = code_path if not existing else f"{code_path}{os.pathsep}{existing}"
    return env


def outer_subprocess_workers(*, all_presentbench: bool, max_workers: int) -> int:
    if all_presentbench:
        return 1
    return max_workers


def preflight_warnings(*, api_type: str, allow_missing_env: bool) -> list[str]:
    warnings: list[str] = []
    if not allow_missing_env:
        return warnings
    if api_type in GEMINI_API_TYPES and not os.getenv("GENAI_API_KEY"):
        warnings.append("GENAI_API_KEY is required by upstream PresentBench gemini judge")
    if api_type in CHAT_JUDGE_API_TYPES and not _openai_judge_env_present():
        warnings.append(OPENAI_JUDGE_ENV_HINT)
    return warnings


def command_manifest(
    commands: list[list[str]],
    errors: list[str],
    warnings: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "presentbench-official-judge-commands/v1",
        "commands": [
            {
                "argv": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
            for command in commands
        ],
        "errors": errors,
        "warnings": warnings or [],
    }


def write_command_manifest(
    path: Path,
    commands: list[list[str]],
    errors: list[str],
    warnings: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            command_manifest(commands, errors, warnings),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument("--code-root", type=Path, default=Path("data/PresentBench_code"))
    parser.add_argument("--data-root", type=Path, default=Path("data/PresentBench_repo"))
    parser.add_argument(
        "--mode-result-root",
        action="append",
        default=None,
        help="MODE=PATH result root. Repeat for each official mode.",
    )
    parser.add_argument(
        "--api-type",
        default="gemini",
        help=(
            "Judge backend. 'gemini'/'gemini_inline' use upstream judge.py; "
            "'openai' routes to the in-repo chat-completions judge "
            "(GOOGLE_THIRD_API_URL/GOOGLE_THIRD_API_KEY env)."
        ),
    )
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--pack-id", action="append")
    parser.add_argument("--limit-packs", type=int)
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument(
        "--all-presentbench",
        action="store_true",
        help=(
            "Run upstream judge_all.py over the full PresentBench checkout. "
            "By default this wrapper runs only selected heldout cells from --packs."
        ),
    )
    parser.add_argument("--thinking-level")
    parser.add_argument("--min-timestamp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help=(
            "Print rendered judge commands to stdout. Commands are always "
            "written to --commands-out."
        ),
    )
    parser.add_argument(
        "--stream-subprocess-output",
        action="store_true",
        help=(
            "Stream upstream judge.py logs to the terminal. By default they are "
            "captured under --subprocess-log-dir."
        ),
    )
    parser.add_argument(
        "--subprocess-log-dir",
        type=Path,
        default=Path("runs/presentbench_official_judge_logs"),
        help="Directory for captured upstream judge.py stdout/stderr logs.",
    )
    parser.add_argument(
        "--commands-out",
        type=Path,
        help="Write rendered judge commands to a JSON manifest for handoff/debugging.",
    )
    parser.add_argument(
        "--expect-commands",
        type=int,
        help="Fail if the rendered command count differs from this value.",
    )
    parser.add_argument(
        "--allow-missing-env",
        action="store_true",
        help=(
            "Permit dry-run command rendering without judge credentials "
            "(GENAI_API_KEY for gemini; GOOGLE_THIRD_API_* for openai)."
        ),
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
    warnings = preflight_warnings(
        api_type=args.api_type,
        allow_missing_env=args.allow_missing_env,
    )
    if args.all_presentbench:
        if args.api_type in CHAT_JUDGE_API_TYPES:
            errors.append(
                "--all-presentbench is not supported with --api-type "
                f"{args.api_type!r}; only upstream gemini routes use judge_all.py. "
                "Use the per-pack flow instead."
            )
            commands = []
        else:
            if not (args.code_root / "judge_all.py").exists():
                errors.append(f"missing upstream judge_all.py under {args.code_root}")
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
    else:
        commands, selection_errors = selected_judge_commands(
            packs_path=args.packs,
            code_root=args.code_root,
            data_root=args.data_root,
            mode_roots=mode_roots,
            python_executable=sys.executable,
            api_type=args.api_type,
            model=args.model,
            retry=args.retry,
            thinking_level=args.thinking_level,
            min_timestamp=args.min_timestamp,
            pack_ids=set(args.pack_id) if args.pack_id else None,
            limit_packs=args.limit_packs,
            limit_heldout=args.limit_heldout,
        )
        errors.extend(selection_errors)

    if args.print_commands:
        for command in commands:
            print(" ".join(command))
    elif commands:
        print(f"Selected {len(commands)} PresentBench official judge command(s).")
        if args.commands_out:
            print(f"Command manifest: {args.commands_out}")
        if not args.dry_run and not args.stream_subprocess_output:
            print(f"Subprocess logs: {args.subprocess_log_dir}")
    if not commands and not errors:
        print("No unscored selected PresentBench official judge cells.")
    if args.expect_commands is not None and len(commands) != args.expect_commands:
        errors.append(
            f"expected {args.expect_commands} selected judge commands, got {len(commands)}"
        )
    if args.commands_out:
        write_command_manifest(args.commands_out, commands, errors, warnings)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    env = subprocess_env_with_code_root(args.code_root)
    return run_subprocesses(
        commands,
        env=env,
        max_workers=outer_subprocess_workers(
            all_presentbench=args.all_presentbench,
            max_workers=args.max_workers,
        ),
        log_dir=args.subprocess_log_dir,
        stream_output=args.stream_subprocess_output,
        log_prefix="presentbench_judge",
    )


if __name__ == "__main__":
    raise SystemExit(main())
