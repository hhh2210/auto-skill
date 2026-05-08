from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.data.run_generation_jobs import count_statuses, has_failed_generation_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "run_generation_jobs.py"


class GenerationJobsRunnerTests(unittest.TestCase):
    def test_status_counts_and_failed_rows(self) -> None:
        rows = [
            {"status": "success"},
            {"status": "error"},
            {"status": "rejected_incomplete_generation"},
        ]

        self.assertEqual(
            count_statuses(rows),
            {"error": 1, "rejected_incomplete_generation": 1, "success": 1},
        )
        self.assertTrue(has_failed_generation_rows(rows))

    def test_success_rows_are_not_failures(self) -> None:
        self.assertFalse(has_failed_generation_rows([{"status": "success"}]))

    def test_config_prefix_requires_prefixed_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            jobs = tmp / "jobs.jsonl"
            jobs.write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "pack_id": "pack-1",
                        "example_id": "ex-1",
                        "prompt": "Write output.",
                        "prompt_sha256": "sha-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            env_file = tmp / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "BAILIAN_BASE_URL=https://example.com/v1",
                        "BAILIAN_API_KEY=test-key",
                        "BAILIAN_MODEL=qwen-plus",
                    ]
                ),
                encoding="utf-8",
            )
            env = {
                key: value
                for key, value in os.environ.items()
                if not (
                    key.startswith("BAILIAN_")
                    or key.startswith("MIMO_")
                    or key.startswith("OPENAI_")
                )
            }

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--jobs",
                    str(jobs),
                    "--out",
                    str(tmp / "out.jsonl"),
                    "--env-file",
                    str(env_file),
                    "--config-prefix",
                    "MIMO",
                ],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("MIMO_MODEL", result.stderr)


if __name__ == "__main__":
    unittest.main()
