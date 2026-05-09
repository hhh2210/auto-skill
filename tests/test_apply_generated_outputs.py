from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data import apply_generated_outputs


class ApplyGeneratedOutputsCliTests(unittest.TestCase):
    def test_cli_fails_closed_without_writing_when_outputs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = root / "packs.jsonl"
            generations = root / "generations.jsonl"
            out = root / "packs.v1.jsonl"
            packs.write_text(
                json.dumps(
                    {
                        "pack_id": "pack-1",
                        "train_examples": [
                            {
                                "example_id": "ex-1",
                                "desired_output": {
                                    "status": "needs_generation",
                                    "generation_job_id": "job-1",
                                    "prompt_sha256": "sha-1",
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            generations.write_text("", encoding="utf-8")

            with patch.object(
                apply_generated_outputs.sys,
                "argv",
                [
                    "apply_generated_outputs.py",
                    "--packs",
                    str(packs),
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                ],
            ):
                self.assertEqual(apply_generated_outputs.main(), 2)

            self.assertFalse(out.exists())

    def test_cli_writes_when_missing_outputs_are_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = root / "packs.jsonl"
            generations = root / "generations.jsonl"
            out = root / "packs.v1.jsonl"
            packs.write_text(
                json.dumps(
                    {
                        "pack_id": "pack-1",
                        "train_examples": [
                            {
                                "example_id": "ex-1",
                                "desired_output": {
                                    "status": "needs_generation",
                                    "generation_job_id": "job-1",
                                    "prompt_sha256": "sha-1",
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            generations.write_text("", encoding="utf-8")

            with patch.object(
                apply_generated_outputs.sys,
                "argv",
                [
                    "apply_generated_outputs.py",
                    "--packs",
                    str(packs),
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                    "--allow-missing",
                ],
            ):
                self.assertEqual(apply_generated_outputs.main(), 0)

            self.assertTrue(out.exists())

    def test_cli_fails_closed_for_rejected_rows_even_when_missing_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = root / "packs.jsonl"
            generations = root / "generations.jsonl"
            out = root / "packs.v1.jsonl"
            packs.write_text(
                json.dumps(
                    {
                        "pack_id": "pack-1",
                        "train_examples": [
                            {
                                "example_id": "ex-1",
                                "desired_output": {
                                    "status": "needs_generation",
                                    "generation_job_id": "job-1",
                                    "prompt_sha256": "sha-1",
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
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
                apply_generated_outputs.sys,
                "argv",
                [
                    "apply_generated_outputs.py",
                    "--packs",
                    str(packs),
                    "--generations",
                    str(generations),
                    "--out",
                    str(out),
                    "--allow-missing",
                ],
            ):
                self.assertEqual(apply_generated_outputs.main(), 2)

            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
