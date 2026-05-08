from __future__ import annotations

import unittest

from auto_skill.readiness import (
    FULL_EVAL_MODES,
    FULL_SKILL_MODES,
    MVP_EVAL_MODES,
    MVP_SKILL_MODES,
    model_inventory,
    readiness_profile,
    readiness_report,
)


def pack(pack_id: str, source: str = "WritingBench") -> dict:
    return {
        "pack_id": pack_id,
        "source": source,
        "train_examples": [
            {
                "example_id": f"{pack_id}::train::0",
                "task_input": "Task A",
                "desired_output": {"status": "generated", "text": "Output A"},
            },
            {
                "example_id": f"{pack_id}::train::1",
                "task_input": "Task B",
                "desired_output": {"status": "generated", "text": "Output B"},
            },
        ],
        "heldout_tasks": [{"task_id": f"{pack_id}::heldout::0", "task_input": "Heldout"}],
    }


def skill_rows(pack_id: str, *, modes: tuple[str, ...] = FULL_SKILL_MODES) -> list[dict]:
    rows = {
        "one_shot_skill_from_examples": {
            "schema_version": "skill-induction/v1",
            "pack_id": pack_id,
            "mode": "one_shot_skill_from_examples",
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        },
        "auto_skill_feature_driven_no_validation": {
            "schema_version": "skill-induction/v1",
            "pack_id": pack_id,
            "mode": "auto_skill_feature_driven_no_validation",
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        },
        "auto_skill_ours_full": {
            "schema_version": "skill-induction/v1",
            "pack_id": pack_id,
            "mode": "auto_skill_ours_full",
            "status": "success",
            "skill_md": "Skill",
            "model_calls": [],
        },
    }
    return [rows[mode] for mode in modes]


def eval_rows(pack_id: str, *, modes: tuple[str, ...] = FULL_EVAL_MODES) -> list[dict]:
    task_id = f"{pack_id}::heldout::0"
    return [
        {
            "schema_version": "heldout-eval/v1",
            "pack_id": pack_id,
            "task_id": task_id,
            "mode": mode,
            "evaluator_kind": "qwen_llm_rubric_surrogate",
            "status": "success",
            "overall_score": 8,
        }
        for mode in modes
    ]


class ReadinessTests(unittest.TestCase):
    def test_readiness_blocks_empty_pack_set(self) -> None:
        report = readiness_report(
            packs=[],
            skill_rows=[],
            writing_eval_rows=[],
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertIn("no example packs loaded", report["blockers"])

    def test_readiness_includes_artifact_blockers(self) -> None:
        report = readiness_report(
            packs=[],
            skill_rows=[],
            writing_eval_rows=[],
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            artifact_blockers=["packs: file not found: missing.jsonl"],
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertIn("packs: file not found: missing.jsonl", report["blockers"])

    def test_readiness_blocks_missing_skill_modes(self) -> None:
        report = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=[
                {
                    "schema_version": "skill-induction/v1",
                    "pack_id": "pack-1",
                    "mode": "one_shot_skill_from_examples",
                    "status": "success",
                    "skill_md": "S",
                    "model_calls": [],
                }
            ],
            writing_eval_rows=eval_rows("pack-1"),
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertTrue(any("missing skill modes" in item for item in report["blockers"]))

    def test_mvp_profile_does_not_require_ours_full_or_auto_skill(self) -> None:
        profile = readiness_profile("mvp")
        report = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=skill_rows("pack-1", modes=MVP_SKILL_MODES),
            writing_eval_rows=eval_rows("pack-1", modes=MVP_EVAL_MODES),
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=profile.require_presentbench_official,
            required_skill_modes=profile.skill_modes,
            required_eval_modes=profile.eval_modes,
            require_complete_coverage=profile.require_complete_coverage,
            profile=profile.name,
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["profile"]["name"], "mvp")

    def test_full_profile_requires_ours_full_and_auto_skill(self) -> None:
        profile = readiness_profile("full")
        report = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=skill_rows("pack-1", modes=MVP_SKILL_MODES),
            writing_eval_rows=eval_rows("pack-1", modes=MVP_EVAL_MODES),
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=profile.require_presentbench_official,
            required_skill_modes=profile.skill_modes,
            required_eval_modes=profile.eval_modes,
            require_complete_coverage=profile.require_complete_coverage,
            profile=profile.name,
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertTrue(any("auto_skill_ours_full" in item for item in report["blockers"]))
        self.assertTrue(
            any("incomplete WritingBench eval coverage" in item for item in report["blockers"])
        )

    def test_smoke_profile_allows_partial_coverage_but_keeps_schema_gate(self) -> None:
        profile = readiness_profile("smoke")
        report = readiness_report(
            packs=[pack("pack-1"), pack("pack-2")],
            skill_rows=skill_rows("pack-1", modes=MVP_SKILL_MODES),
            writing_eval_rows=eval_rows("pack-1", modes=MVP_EVAL_MODES),
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=profile.require_presentbench_official,
            required_skill_modes=profile.skill_modes,
            required_eval_modes=profile.eval_modes,
            require_complete_coverage=profile.require_complete_coverage,
            profile=profile.name,
        )

        self.assertEqual(report["status"], "ready")

        malformed = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=[{"pack_id": "pack-1", "mode": "one_shot_skill_from_examples"}],
            writing_eval_rows=[],
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=profile.require_presentbench_official,
            required_skill_modes=profile.skill_modes,
            required_eval_modes=profile.eval_modes,
            require_complete_coverage=profile.require_complete_coverage,
            profile=profile.name,
        )

        self.assertEqual(malformed["status"], "not_ready")
        self.assertTrue(malformed["schema_errors"]["skills"])

    def test_readiness_blocks_missing_required_presentbench_official_scores(self) -> None:
        report = readiness_report(
            packs=[pack("pack-1", source="PresentBench")],
            skill_rows=skill_rows("pack-1"),
            writing_eval_rows=[],
            present_surrogate_rows=eval_rows("pack-1"),
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=True,
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertTrue(
            any(
                "PresentBench official score rows are absent" in item
                for item in report["blockers"]
            )
        )

    def test_readiness_allows_missing_presentbench_official_for_smoke(self) -> None:
        report = readiness_report(
            packs=[pack("pack-1", source="PresentBench")],
            skill_rows=skill_rows("pack-1"),
            writing_eval_rows=eval_rows("pack-1"),
            present_surrogate_rows=eval_rows("pack-1"),
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=False,
        )

        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["blockers"])
        self.assertTrue(report["warnings"])

    def test_readiness_blocks_missing_writing_eval_target(self) -> None:
        report = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=skill_rows("pack-1"),
            writing_eval_rows=[],
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertTrue(
            any("missing WritingBench eval rows" in item for item in report["blockers"])
        )

    def test_readiness_blocks_duplicate_eval_cell_with_any_failure(self) -> None:
        rows = eval_rows("pack-1", modes=MVP_EVAL_MODES)
        rows.append(
            {
                **rows[0],
                "status": "judge_parse_error",
                "overall_score": None,
            }
        )

        report = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=skill_rows("pack-1", modes=MVP_SKILL_MODES),
            writing_eval_rows=rows,
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            required_skill_modes=MVP_SKILL_MODES,
            required_eval_modes=MVP_EVAL_MODES,
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertTrue(
            any("incomplete WritingBench eval coverage" in item for item in report["blockers"])
        )
        coverage = next(iter(report["evaluations"]["writingbench"]["coverage"].values()))
        self.assertEqual(
            coverage["duplicate_modes"]["prompt_only"],
            ["success", "judge_parse_error"],
        )
        self.assertIn("prompt_only", coverage["non_success_modes"])

    def test_readiness_blocks_malformed_success_rows_without_scores(self) -> None:
        report = readiness_report(
            packs=[pack("pack-1", source="PresentBench")],
            skill_rows=skill_rows("pack-1"),
            writing_eval_rows=[],
            present_surrogate_rows=eval_rows("pack-1"),
            present_official_rows=[
                {
                    "schema_version": "presentbench-official-score/v1",
                    "pack_id": "pack-1",
                    "task_id": "pack-1::heldout::0",
                    "mode": "prompt_only",
                    "evaluator_kind": "presentbench_official_score_yaml",
                    "status": "success",
                    "overall_score": None,
                },
                {
                    "schema_version": "presentbench-official-score/v1",
                    "pack_id": "pack-1",
                    "task_id": "pack-1::heldout::0",
                    "mode": "auto_skill",
                    "evaluator_kind": "presentbench_official_score_yaml",
                    "status": "success",
                    "overall_score": None,
                },
            ],
            limit_heldout=1,
        )

        self.assertEqual(report["status"], "not_ready")
        self.assertTrue(report["schema_errors"]["presentbench_official"])


class ModelInventoryTests(unittest.TestCase):
    def test_monoculture_when_all_rows_share_one_model(self) -> None:
        inventory = model_inventory(
            skill_rows=[{"solver_model": "qwen-3.5-plus"}],
            eval_rows=[
                {"solver_model": "qwen-3.5-plus", "judge_model": "qwen-3.5-plus"},
            ],
        )
        self.assertTrue(inventory["monoculture"])
        self.assertEqual(inventory["monoculture_model"], "qwen-3.5-plus")

    def test_split_judge_breaks_monoculture(self) -> None:
        inventory = model_inventory(
            skill_rows=[{"solver_model": "qwen-3.5-plus"}],
            eval_rows=[
                {"solver_model": "qwen-3.5-plus", "judge_model": "claude-haiku-4-5"},
            ],
        )
        self.assertFalse(inventory["monoculture"])
        self.assertIn("claude-haiku-4-5", inventory["all_models"])
        self.assertIn("qwen-3.5-plus", inventory["all_models"])

    def test_empty_inventory_does_not_flag_monoculture(self) -> None:
        inventory = model_inventory(skill_rows=[], eval_rows=[])
        self.assertFalse(inventory["monoculture"])
        self.assertIsNone(inventory["monoculture_model"])
        self.assertTrue(inventory["model_identity_complete"])

    def test_inventory_counts_missing_model_identity(self) -> None:
        inventory = model_inventory(
            skill_rows=[{"mode": "old_skill_without_solver_model"}],
            eval_rows=[
                {"solver_model": "qwen-3.5-plus", "judge_model": "qwen-3.5-plus"},
            ],
        )
        self.assertTrue(inventory["monoculture"])
        self.assertFalse(inventory["model_identity_complete"])
        self.assertEqual(
            inventory["missing_model_identity"],
            {
                "skill_solver_model": 1,
                "eval_solver_model": 0,
                "eval_judge_model": 0,
            },
        )

    def test_readiness_emits_monoculture_warning(self) -> None:
        report = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=[
                {**row, "solver_model": "qwen-3.5-plus"} for row in skill_rows("pack-1")
            ],
            writing_eval_rows=[
                {**row, "solver_model": "qwen-3.5-plus", "judge_model": "qwen-3.5-plus"}
                for row in eval_rows("pack-1")
            ],
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=False,
        )
        self.assertTrue(
            any("model_monoculture" in warn for warn in report["warnings"]),
            report["warnings"],
        )
        self.assertTrue(report["model_inventory"]["monoculture"])

    def test_readiness_qualifies_monoculture_warning_when_model_identity_missing(self) -> None:
        report = readiness_report(
            packs=[pack("pack-1")],
            skill_rows=skill_rows("pack-1"),
            writing_eval_rows=[
                {**row, "solver_model": "qwen-3.5-plus", "judge_model": "qwen-3.5-plus"}
                for row in eval_rows("pack-1")
            ],
            present_surrogate_rows=[],
            present_official_rows=[],
            limit_heldout=1,
            require_presentbench_official=False,
        )
        warning = next(warn for warn in report["warnings"] if "model_monoculture" in warn)
        self.assertIn("visible solver/judge model fields", warning)
        self.assertIn("missing top-level model identity", warning)
        self.assertFalse(report["model_inventory"]["model_identity_complete"])


if __name__ == "__main__":
    unittest.main()
