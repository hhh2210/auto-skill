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


def present_pack_row() -> dict:
    return {
        "schema_version": "example-pack/v1",
        "pack_id": "present-1",
        "source": "PresentBench",
        "train_examples": [
            {
                "example_id": "present-1::train::0",
                "task_input": "Task A",
                "desired_output": {"status": "generated", "text": "Output A"},
            },
            {
                "example_id": "present-1::train::1",
                "task_input": "Task B",
                "desired_output": {"status": "generated", "text": "Output B"},
            },
        ],
        "heldout_tasks": [{"task_id": "present-1::heldout::0", "task_input": "Heldout"}],
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


def present_mvp_skill_rows() -> list[dict]:
    return [
        {
            "schema_version": "skill-induction/v1",
            "pack_id": "present-1",
            "mode": mode,
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        }
        for mode in ("one_shot_skill_from_examples", "auto_skill_feature_driven_no_validation")
    ]


def ours_full_skill_rows() -> list[dict]:
    return [
        {
            "schema_version": "skill-induction/v1",
            "pack_id": "pack-1",
            "mode": "auto_skill_feature_driven_no_validation",
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        },
        {
            "schema_version": "skill-induction/v1",
            "pack_id": "pack-1",
            "mode": "auto_skill_ours_full",
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        },
    ]


def present_ours_full_skill_rows() -> list[dict]:
    return [
        {
            "schema_version": "skill-induction/v1",
            "pack_id": "present-1",
            "mode": "auto_skill_feature_driven_no_validation",
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        },
        {
            "schema_version": "skill-induction/v1",
            "pack_id": "present-1",
            "mode": "auto_skill_ours_full",
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        },
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


def present_eval_rows() -> list[dict]:
    return [
        {
            "schema_version": "heldout-eval/v1",
            "pack_id": "present-1",
            "task_id": "present-1::heldout::0",
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


def full_eval_rows() -> list[dict]:
    rows = eval_rows()
    rows.append(
        {
            "schema_version": "heldout-eval/v1",
            "pack_id": "pack-1",
            "task_id": "pack-1::heldout::0",
            "mode": "auto_skill",
            "evaluator_kind": "qwen_llm_rubric_surrogate",
            "status": "success",
            "overall_score": 8,
        }
    )
    return rows


def present_auto_skill_eval_rows() -> list[dict]:
    return [
        {
            "schema_version": "heldout-eval/v1",
            "pack_id": "present-1",
            "task_id": "present-1::heldout::0",
            "mode": "auto_skill",
            "evaluator_kind": "qwen_llm_rubric_surrogate",
            "status": "success",
            "overall_score": 8,
        }
    ]


class ReportExperimentReadinessCliTests(unittest.TestCase):
    def test_default_profile_uses_mvp_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            write_jsonl(tmp / "artifacts/packs/example_packs.v1.jsonl", [pack_row()])
            write_jsonl(tmp / "runs/skill_mvp.qwen.mvp.jsonl", skill_rows())
            write_jsonl(
                tmp
                / "runs"
                / "writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl",
                eval_rows(),
            )
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

    def test_mvp_profile_does_not_probe_missing_official_scores_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            write_jsonl(tmp / "artifacts/packs/example_packs.v1.jsonl", [present_pack_row()])
            write_jsonl(tmp / "runs/skill_mvp.qwen.mvp.jsonl", present_mvp_skill_rows())
            write_jsonl(
                tmp
                / "runs"
                / "writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl",
                [],
            )
            write_jsonl(
                tmp / "runs/presentbench_surrogate_eval.qwen.mvp.jsonl",
                present_eval_rows(),
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--profile",
                    "mvp",
                    "--limit-heldout",
                    "1",
                    "--expect-status",
                    "ready",
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
            self.assertEqual(report["evaluations"]["presentbench_official"]["rows"], 0)
            self.assertFalse(
                any("PresentBench official score" in item for item in report["warnings"]),
                report["warnings"],
            )

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

    def test_full_profile_default_merges_mvp_and_ours_full_skill_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            write_jsonl(
                tmp / "artifacts/packs/example_packs.v1.jsonl",
                [pack_row(), present_pack_row()],
            )
            write_jsonl(
                tmp / "runs/skill_mvp.qwen.mvp.jsonl",
                skill_rows() + present_mvp_skill_rows(),
            )
            write_jsonl(
                tmp / "runs/skill_mvp.qwen.ours_full.writingbench.jsonl",
                ours_full_skill_rows(),
            )
            write_jsonl(
                tmp / "runs/skill_mvp.qwen.ours_full.presentbench.jsonl",
                present_ours_full_skill_rows(),
            )
            write_jsonl(
                tmp
                / "runs"
                / "writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl",
                full_eval_rows(),
            )
            write_jsonl(
                tmp / "runs/presentbench_surrogate_eval.qwen.mvp.jsonl",
                present_eval_rows(),
            )
            write_jsonl(
                tmp / "runs/presentbench_surrogate_eval.qwen.auto_skill.jsonl",
                present_auto_skill_eval_rows(),
            )

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
            self.assertFalse(
                any("missing skill modes" in item for item in report["blockers"]),
                report["blockers"],
            )
            self.assertFalse(
                any("PresentBench surrogate eval coverage" in item for item in report["blockers"]),
                report["blockers"],
            )
            self.assertTrue(
                any(
                    "presentbench_official_scores: file not found" in item
                    for item in report["blockers"]
                ),
                report["blockers"],
            )

    def test_cli_can_merge_repeated_skill_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            write_jsonl(tmp / "packs.jsonl", [pack_row()])
            write_jsonl(
                tmp / "one.jsonl",
                [row for row in skill_rows() if row["mode"] == "one_shot_skill_from_examples"],
            )
            write_jsonl(
                tmp / "feature.jsonl",
                [
                    row
                    for row in skill_rows()
                    if row["mode"] == "auto_skill_feature_driven_no_validation"
                ],
            )
            write_jsonl(tmp / "writing.jsonl", eval_rows())
            write_jsonl(tmp / "present.jsonl", [])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--profile",
                    "mvp",
                    "--packs",
                    "packs.jsonl",
                    "--skills",
                    "one.jsonl",
                    "--skills",
                    "feature.jsonl",
                    "--writing-eval",
                    "writing.jsonl",
                    "--present-surrogate-eval",
                    "present.jsonl",
                    "--limit-heldout",
                    "1",
                    "--expect-status",
                    "ready",
                    "--out",
                    "runs/readiness.json",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

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
