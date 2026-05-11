from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from auto_skill.example_packs import load_jsonl, write_jsonl

SCRIPT = Path("scripts/ops/combine_skill_shards.py")


class CombineSkillShardsTests(unittest.TestCase):
    def test_fails_closed_when_pack_has_no_successful_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = root / "packs.jsonl"
            out = root / "combined.jsonl"
            prefix = root / "skill"
            write_jsonl(packs, [{"pack_id": "pack-a"}, {"pack_id": "pack-b"}])
            write_jsonl(
                Path(f"{prefix}.shard0.jsonl"),
                [
                    {
                        "pack_id": "pack-a",
                        "mode": "auto_skill_minimal",
                        "status": "success",
                        "skill_md": "Skill.",
                    },
                    {
                        "pack_id": "pack-b",
                        "mode": "auto_skill_minimal",
                        "status": "induction_error",
                    },
                ],
            )

            result = self._run(
                "--packs",
                packs,
                "--out",
                out,
                "--prefix",
                prefix,
                "--shards",
                "1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing successful pack rows", result.stderr)
            self.assertFalse(out.exists())

    def test_allow_partial_writes_available_successes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = root / "packs.jsonl"
            out = root / "combined.jsonl"
            prefix = root / "skill"
            write_jsonl(packs, [{"pack_id": "pack-a"}, {"pack_id": "pack-b"}])
            write_jsonl(
                Path(f"{prefix}.shard0.jsonl"),
                [
                    {
                        "pack_id": "pack-a",
                        "mode": "auto_skill_minimal",
                        "status": "success",
                        "skill_md": "Skill.",
                    }
                ],
            )

            result = self._run(
                "--packs",
                packs,
                "--out",
                out,
                "--prefix",
                prefix,
                "--shards",
                "1",
                "--allow-partial",
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual([row["pack_id"] for row in load_jsonl(out)], ["pack-a"])

    def test_rejects_duplicate_success_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = root / "packs.jsonl"
            out = root / "combined.jsonl"
            prefix = root / "skill"
            write_jsonl(packs, [{"pack_id": "pack-a"}])
            write_jsonl(
                Path(f"{prefix}.shard0.jsonl"),
                [
                    {
                        "pack_id": "pack-a",
                        "mode": "auto_skill_minimal",
                        "status": "success",
                        "skill_md": "First.",
                    },
                    {
                        "pack_id": "pack-a",
                        "mode": "auto_skill_minimal",
                        "status": "success",
                        "skill_md": "Duplicate.",
                    },
                ],
            )

            result = self._run(
                "--packs",
                packs,
                "--out",
                out,
                "--prefix",
                prefix,
                "--shards",
                "1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate successful skill row", result.stderr)

    def _run(self, *args: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
