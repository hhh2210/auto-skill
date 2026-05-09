"""Small helpers for long-running script orchestration."""

from __future__ import annotations

import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubprocessRunResult:
    index: int
    returncode: int
    log_path: Path | None = None


def subprocess_log_path(log_dir: Path, index: int, *, prefix: str = "command") -> Path:
    return log_dir / f"{prefix}_{index:03d}.log"


def run_one_subprocess(
    index: int,
    command: list[str],
    *,
    env: dict[str, str],
    log_dir: Path | None,
    stream_output: bool,
    log_prefix: str = "command",
) -> SubprocessRunResult:
    if stream_output:
        completed = subprocess.run(command, check=False, env=env)
        return SubprocessRunResult(index=index, returncode=completed.returncode)
    if log_dir is None:
        completed = subprocess.run(
            command,
            check=False,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return SubprocessRunResult(index=index, returncode=completed.returncode)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = subprocess_log_path(log_dir, index, prefix=log_prefix)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            check=False,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return SubprocessRunResult(
        index=index,
        returncode=completed.returncode,
        log_path=log_path,
    )


def run_subprocesses(
    commands: list[list[str]],
    *,
    env: dict[str, str],
    max_workers: int,
    log_dir: Path | None,
    stream_output: bool = False,
    log_prefix: str = "command",
) -> int:
    if max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if max_workers == 1 or len(commands) <= 1:
        for index, command in enumerate(commands, start=1):
            result = run_one_subprocess(
                index,
                command,
                env=env,
                log_dir=log_dir,
                stream_output=stream_output,
                log_prefix=log_prefix,
            )
            if result.returncode != 0:
                if result.log_path is not None:
                    print(
                        f"error: command {index} failed; see {result.log_path}",
                        file=sys.stderr,
                    )
                return result.returncode
        return 0

    first_failure: SubprocessRunResult | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_one_subprocess,
                index,
                command,
                env=env,
                log_dir=log_dir,
                stream_output=stream_output,
                log_prefix=log_prefix,
            )
            for index, command in enumerate(commands, start=1)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result.returncode != 0 and first_failure is None:
                first_failure = result
    if first_failure is not None:
        if first_failure.log_path is not None:
            print(f"error: a command failed; see {first_failure.log_path}", file=sys.stderr)
        return first_failure.returncode
    return 0
