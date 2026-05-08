from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_skill.example_packs import load_jsonl
from auto_skill.mvp import PromptRunResult
from scripts.skills.run_skill_mvp import (
    InductionError,
    StageCallConfig,
    StageLedger,
    checkpoint_rows,
    default_stage_ledger_path,
    existing_success_modes,
    parse_required_json,
    requested_output_modes,
    require_complete_run,
)


class SkillMVPRunnerTests(unittest.TestCase):
    def test_require_complete_run_rejects_truncation(self) -> None:
        with self.assertRaisesRegex(InductionError, "did not finish cleanly"):
            require_complete_run(
                PromptRunResult(text="partial", finish_reason="length"),
                stage="feature_extraction",
            )

    def test_require_complete_run_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(InductionError, "empty text"):
            require_complete_run(
                PromptRunResult(text=" ", finish_reason="stop"),
                stage="skill_compilation",
            )

    def test_parse_required_json_rejects_parse_error(self) -> None:
        with self.assertRaisesRegex(InductionError, "invalid JSON"):
            parse_required_json("not json", stage="validation")

    def test_requested_output_modes_include_full_only_when_loo_possible(self) -> None:
        pack = {
            "train_examples": [
                {
                    "example_id": "ex-1",
                    "task_input": "Task",
                    "desired_output": {"status": "generated", "text": "Output"},
                },
                {
                    "example_id": "ex-2",
                    "task_input": "Task",
                    "desired_output": {"status": "generated", "text": "Output"},
                },
            ]
        }

        self.assertEqual(
            requested_output_modes(
                pack,
                {"one_shot_skill_from_examples", "auto_skill_feature_driven"},
                leave_one_out=True,
            ),
            {
                "one_shot_skill_from_examples",
                "auto_skill_feature_driven_no_validation",
            },
        )

    def test_existing_success_modes_ignores_failures(self) -> None:
        rows = [
            {
                "pack_id": "pack-1",
                "mode": "auto_skill_ours_full",
                "status": "success",
                "skill_md": "# Skill",
            },
            {
                "pack_id": "pack-1",
                "mode": "one_shot_skill_from_examples",
                "status": "induction_error",
                "skill_md": None,
            },
        ]

        self.assertEqual(existing_success_modes(rows), {"pack-1": {"auto_skill_ours_full"}})

    def test_checkpoint_rows_writes_successes_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "skills.jsonl"
            checkpoint_rows(
                out,
                [{"pack_id": "pack-1", "mode": "one_shot", "status": "success"}],
                [{"pack_id": "pack-2", "mode": "pack_induction", "status": "induction_error"}],
            )

            self.assertEqual(
                load_jsonl(out),
                [
                    {"pack_id": "pack-1", "mode": "one_shot", "status": "success"},
                    {
                        "pack_id": "pack-2",
                        "mode": "pack_induction",
                        "status": "induction_error",
                    },
                ],
            )

    def test_stage_ledger_resumes_matching_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "skills.stages.jsonl"
            config = StageCallConfig(
                model="qwen-plus",
                temperature=0.2,
                max_tokens=128,
                enable_thinking=False,
                thinking_budget=None,
                stream=True,
            )
            ledger = StageLedger(path, resume=False)
            ledger.append(
                {
                    "schema_version": "skill-induction-stage/v1",
                    "pack_id": "pack-1",
                    "stage": "feature_extraction",
                    "example_id": "ex-1",
                    "prompt_sha256": "prompt",
                    "config_sha256": config.stable_hash(),
                    "config": config.to_json(),
                    "status": "success",
                    "run": {"text": "{}", "finish_reason": "stop"},
                    "parsed_json": {},
                }
            )

            resumed = StageLedger(path, resume=True)

            self.assertIsNotNone(
                resumed.success(
                    pack_id="pack-1",
                    stage="feature_extraction",
                    example_id="ex-1",
                    prompt_hash="prompt",
                    config_hash=config.stable_hash(),
                )
            )
            self.assertIsNone(
                resumed.success(
                    pack_id="pack-1",
                    stage="feature_extraction",
                    example_id="ex-1",
                    prompt_hash="prompt",
                    config_hash="different",
                )
            )

    def test_default_stage_ledger_path_uses_out_stem(self) -> None:
        self.assertEqual(
            default_stage_ledger_path(Path("runs/skills.jsonl")),
            Path("runs/skills.stages.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
