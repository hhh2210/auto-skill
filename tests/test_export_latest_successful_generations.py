from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data import export_latest_successful_generations


class ExportLatestSuccessfulGenerationsTests(unittest.TestCase):
    def test_cli_exports_latest_successes_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generations = root / "generations.jsonl"
            out = root / "latest_success.jsonl"
            summary = root / "summary.json"
            rows = [
                {
                    "status": "rejected_incomplete_generation",
                    "job_id": "job-1",
                    "prompt_sha256": "sha-1",
                    "finish_reason": "length",
                },
                {
                    "status": "success",
                    "job_id": "job-1",
                    "prompt_sha256": "sha-1",
                    "finish_reason": "stop",
                },
            ]
            generations.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch.object(
                export_latest_successful_generations.sys,
                "argv",
                [
                    "export_latest_successful_generations.py",
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                    "--summary-out",
                    str(summary),
                    "--expect-successes",
                    "1",
                ],
            ):
                self.assertEqual(export_latest_successful_generations.main(), 0)

            exported = [json.loads(line) for line in out.read_text().splitlines()]
            report = json.loads(summary.read_text())
            self.assertEqual([row["job_id"] for row in exported], ["job-1"])
            self.assertEqual(report["input_rows"], 2)
            self.assertEqual(report["latest_rows"], 1)
            self.assertEqual(report["exported_success_rows"], 1)

    def test_cli_fails_when_latest_row_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generations = root / "generations.jsonl"
            out = root / "latest_success.jsonl"
            generations.write_text(
                json.dumps(
                    {
                        "status": "rejected_incomplete_generation",
                        "job_id": "job-1",
                        "prompt_sha256": "sha-1",
                        "finish_reason": "length",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                export_latest_successful_generations.sys,
                "argv",
                [
                    "export_latest_successful_generations.py",
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                ],
            ):
                self.assertEqual(export_latest_successful_generations.main(), 4)
            self.assertFalse(out.exists())

    def test_cli_fails_when_latest_success_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generations = root / "generations.jsonl"
            out = root / "latest_success.jsonl"
            generations.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "job_id": "job-1",
                        "prompt_sha256": "sha-1",
                        "finish_reason": "length",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                export_latest_successful_generations.sys,
                "argv",
                [
                    "export_latest_successful_generations.py",
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                ],
            ):
                self.assertEqual(export_latest_successful_generations.main(), 4)
            self.assertFalse(out.exists())

    def test_cli_does_not_write_when_expected_success_count_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generations = root / "generations.jsonl"
            out = root / "latest_success.jsonl"
            generations.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "job_id": "job-1",
                        "prompt_sha256": "sha-1",
                        "finish_reason": "stop",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                export_latest_successful_generations.sys,
                "argv",
                [
                    "export_latest_successful_generations.py",
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                    "--expect-successes",
                    "2",
                ],
            ):
                self.assertEqual(export_latest_successful_generations.main(), 3)
            self.assertFalse(out.exists())

    def test_cli_writes_partial_when_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generations = root / "generations.jsonl"
            out = root / "latest_success.jsonl"
            generations.write_text(
                json.dumps(
                    {
                        "status": "rejected_incomplete_generation",
                        "job_id": "job-1",
                        "prompt_sha256": "sha-1",
                        "finish_reason": "length",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                export_latest_successful_generations.sys,
                "argv",
                [
                    "export_latest_successful_generations.py",
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                    "--allow-incomplete-latest",
                ],
            ):
                self.assertEqual(export_latest_successful_generations.main(), 0)
            self.assertTrue(out.exists())
            self.assertEqual(out.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
