from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_skill.example_packs import write_jsonl
from scripts.skills.run_skill_minimal import ParseRetrySupervisor, existing_successes, run


@dataclass
class MockResult:
    text: str
    finish_reason: str | None = "stop"


class SequencedClient:
    def __init__(self, results: list[MockResult]):
        self.results = list(results)
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], **_: Any) -> MockResult:
        self.calls += 1
        return self.results.pop(0)


class SkillMinimalRunnerTests(unittest.TestCase):
    def test_existing_successes_ignores_malformed_rows(self) -> None:
        rows = [
            {"status": "success", "mode": "auto_skill_minimal", "skill_md": "# Missing pack"},
            {"pack_id": "pack-1", "status": "success", "mode": "other", "skill_md": "# Skill"},
            {
                "pack_id": "pack-2",
                "status": "success",
                "mode": "auto_skill_minimal",
                "skill_md": " ",
            },
            {
                "pack_id": "pack-3",
                "status": "success",
                "mode": "auto_skill_minimal",
                "skill_md": "# Skill",
            },
        ]

        self.assertEqual(existing_successes(rows), {"pack-3"})

    def test_parse_retry_supervisor_retries_truncated_valid_json(self) -> None:
        valid_report = json.dumps(
            {
                "artifact_critique": {"too_generic": [], "over_specific": [], "missing": []},
                "rule_grounding": [],
            }
        )
        client = SequencedClient(
            [
                MockResult(valid_report, finish_reason="length"),
                MockResult(valid_report, finish_reason="stop"),
            ]
        )
        supervisor = ParseRetrySupervisor(client, max_attempts=2)

        result = supervisor.complete([{"role": "user", "content": "prompt"}])

        self.assertEqual(client.calls, 2)
        self.assertEqual(result.finish_reason, "stop")

    def test_run_fails_closed_when_no_usable_packs_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packs = Path(tmp) / "packs.jsonl"
            out = Path(tmp) / "skills.jsonl"
            write_jsonl(
                packs,
                [{"pack_id": "empty-pack", "train_examples": []}],
            )
            args = argparse.Namespace(
                packs=packs,
                out=out,
                resume=False,
                dry_run=True,
                memory=None,
                no_memory=True,
                memory_scope="within_pack",
                memory_top_k=None,
                solver_config_prefix="BAILIAN",
                supervisor_config_prefix="MIMO",
                temperature=None,
                max_tokens=8192,
                parse_max_attempts=3,
                allow_partial=False,
                no_enable_thinking=True,
            )

            self.assertEqual(run(args), 3)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
