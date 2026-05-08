from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "report_experiment_readiness.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def pack_row() -> dict:
    return {
        "schema_version": "example-pack/v1",
        "pack_id": "pack-1",
        "source": "WritingBench",
        "train_examples": [
            {
                "example_id": "pack-1::train::0",
                "task_input": "Task A",
                "desired_output": {"status": "generated", "text": "Output A"},
            },
            {
                "example_id": "pack-1::train::1",
                "task_input": "Task B",
                "desired_output": {"status": "generated", "text": "Output B"},
            },
        ],
        "heldout_tasks": [{"task_id": "pack-1::heldout::0", "task_input": "Heldout"}],
    }


def skill_rows() -> list[dict]:
    return [
        {
            "schema_version": "skill-induction/v1",
            "pack_id": "pack-1",
            "mode": mode,
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        }
        for mode in ("one_shot_skill_from_examples", "auto_skill_feature_driven_no_validation")
    ]


def eval_rows() -> list[dict]:
    return [
        {
            "schema_version": "heldout-eval/v1",
            "pack_id": "pack-1",
            "task_id": "pack-1::heldout::0",
            "mode": mode,
            "evaluator_kind": "qwen_llm_rubric_surrogate",
            "status": "success",
            "overall_score": 8,
        }
        for mode in (
            "prompt_only",
            "few_shot_examples_only",
            "one_shot_skill_from_examples",
            "ours_no_validation",
        )
    ]


class ReportExperimentReadinessCliTests(unittest.TestCase):
    def test_default_profile_uses_mvp_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            write_jsonl(tmp / "artifacts/packs/example_packs.v1.jsonl", [pack_row()])
            write_jsonl(tmp / "runs/skill_mvp.qwen.mvp.jsonl", skill_rows())
            write_jsonl(tmp / "runs/writingbench_official_eval.qwen.mvp.jsonl", eval_rows())
            write_jsonl(tmp / "runs/presentbench_surrogate_eval.qwen.mvp.jsonl", [])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--limit-heldout",
                    "1",
                    "--out",
                    "runs/readiness.json",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((tmp / "runs/readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(report["profile"]["name"], "mvp")
            self.assertEqual(report["status"], "ready")

    def test_full_profile_uses_full_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            write_jsonl(tmp / "artifacts/packs/example_packs.v1.jsonl", [pack_row()])
            write_jsonl(tmp / "runs/skill_mvp.qwen.mvp.jsonl", skill_rows())
            write_jsonl(tmp / "runs/writingbench_official_eval.qwen.mvp.jsonl", eval_rows())
            write_jsonl(tmp / "runs/presentbench_surrogate_eval.qwen.mvp.jsonl", [])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--profile",
                    "full",
                    "--limit-heldout",
                    "1",
                    "--expect-status",
                    "not_ready",
                    "--out",
                    "runs/readiness.json",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((tmp / "runs/readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(report["profile"]["name"], "full")
            self.assertEqual(report["status"], "not_ready")

    def test_smoke_profile_defaults_to_current_artifacts_with_looser_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            write_jsonl(tmp / "artifacts/packs/example_packs.v1.jsonl", [pack_row()])
            write_jsonl(tmp / "runs/skill_mvp.qwen.mvp.jsonl", skill_rows())
            write_jsonl(tmp / "runs/writingbench_official_eval.qwen.mvp.jsonl", [])
            write_jsonl(tmp / "runs/presentbench_surrogate_eval.qwen.mvp.jsonl", [])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--profile",
                    "smoke",
                    "--limit-heldout",
                    "1",
                    "--out",
                    "runs/readiness.json",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((tmp / "runs/readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(report["profile"]["name"], "smoke")
            self.assertEqual(report["status"], "ready")


if __name__ == "__main__":
    unittest.main()
