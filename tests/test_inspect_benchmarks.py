from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "inspect_benchmarks.py"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class InspectBenchmarksTests(unittest.TestCase):
    def test_diagnostic_seed_examples_land_under_ignored_seed_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            writing_root = tmp / "WritingBench"
            writing_file = writing_root / "benchmark_query" / "benchmark_all.jsonl"
            writing_file.parent.mkdir(parents=True)
            writing_file.write_text(
                json.dumps(
                    {
                        "index": 1,
                        "domain1": "Finance",
                        "domain2": "Report",
                        "lang": "en",
                        "query": "Write a report.",
                        "checklist": [{"name": "Depth"}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            present_root = tmp / "PresentBench"
            case_dir = present_root / "education" / "Demo" / "CaseA"
            generation_dir = case_dir / "generation_task"
            generation_dir.mkdir(parents=True)
            (generation_dir / "instructions.md").write_text("Make slides.", encoding="utf-8")
            write_json(
                generation_dir / "judge_prompt.json",
                {"material_independent_checklist_1": ["Has title"]},
            )
            (generation_dir / "statistics.yaml").write_text("slides: 5\n", encoding="utf-8")

            out_dir = tmp / "artifacts"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--writingbench-root",
                    str(writing_root),
                    "--presentbench-root",
                    str(present_root),
                    "--limit",
                    "1",
                    "--out-dir",
                    str(out_dir),
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((out_dir / "benchmark_inspection.json").exists())
            self.assertTrue((out_dir / "seed" / "seed_examples.jsonl").exists())
            self.assertFalse((out_dir / "seed_examples.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
