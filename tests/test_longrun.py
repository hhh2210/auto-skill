from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from auto_skill.longrun import run_subprocesses, subprocess_log_path


class LongRunTests(unittest.TestCase):
    def test_run_subprocesses_uses_worker_pool_and_returns_first_failure(self) -> None:
        calls: list[list[str]] = []

        def fake_run(
            command: list[str],
            *,
            check: bool,
            env: dict[str, str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            calls.append(command)
            return subprocess.CompletedProcess(command, 7 if command[-1] == "bad" else 0)

        with patch("auto_skill.longrun.subprocess.run", fake_run):
            returncode = run_subprocesses(
                [["/python", "ok1"], ["/python", "bad"], ["/python", "ok2"]],
                env={"PYTHONPATH": "/code"},
                max_workers=2,
                log_dir=None,
            )

        self.assertEqual(returncode, 7)
        self.assertCountEqual(
            calls,
            [["/python", "ok1"], ["/python", "bad"], ["/python", "ok2"]],
        )

    def test_run_subprocesses_rejects_invalid_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "max-workers"):
            run_subprocesses([["/python", "judge.py"]], env={}, max_workers=0, log_dir=None)

    def test_run_subprocesses_captures_output_to_per_command_log(self) -> None:
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            returncode = run_subprocesses(
                [
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('stdout text'); print('stderr text', file=sys.stderr)",
                    ]
                ],
                env=os.environ.copy(),
                max_workers=1,
                log_dir=log_dir,
                log_prefix="presentbench_judge",
            )

            self.assertEqual(returncode, 0)
            log_text = subprocess_log_path(
                log_dir,
                1,
                prefix="presentbench_judge",
            ).read_text(encoding="utf-8")
            self.assertIn("stdout text", log_text)
            self.assertIn("stderr text", log_text)


if __name__ == "__main__":
    unittest.main()
