from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from auto_skill.example_packs import load_jsonl, write_jsonl
from scripts.ops import prepare_three_metric_ablation as helper


class PrepareThreeMetricAblationTests(unittest.TestCase):
    def test_minimal_alias_filters_rows_and_marks_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills = root / "skills.jsonl"
            out = root / "alias.jsonl"
            write_jsonl(
                skills,
                [
                    {
                        "pack_id": "pack-a",
                        "mode": "auto_skill_minimal",
                        "status": "success",
                        "skill_md": "Use concise bullets.",
                    },
                    {
                        "pack_id": "pack-b",
                        "mode": "auto_skill_minimal",
                        "status": "induction_error",
                        "skill_md": "Skip failures.",
                    },
                    {
                        "pack_id": "pack-c",
                        "mode": "one_shot_skill_from_examples",
                        "status": "success",
                        "skill_md": "Skip wrong mode.",
                    },
                ],
            )

            code = helper.cmd_minimal_alias(
                argparse.Namespace(skills=skills, out=out, variant="no_memory")
            )

            rows = load_jsonl(out)
            self.assertEqual(code, 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pack_id"], "pack-a")
            self.assertEqual(rows[0]["mode"], "auto_skill_feature_driven_no_validation")
            self.assertEqual(rows[0]["metadata"]["minimal_memory_variant"], "no_memory")

    def test_minimal_alias_rejects_duplicate_success_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills = root / "skills.jsonl"
            out = root / "alias.jsonl"
            write_jsonl(
                skills,
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

            code = helper.cmd_minimal_alias(
                argparse.Namespace(skills=skills, out=out, variant="no_memory")
            )

            self.assertEqual(code, 3)
            self.assertFalse(out.exists())

    def test_pairwise_input_fails_closed_on_missing_required_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "pairwise.jsonl"

            with self.assertRaises(FileNotFoundError):
                helper.cmd_pairwise_input(
                    argparse.Namespace(
                        base_eval=root / "missing-base.jsonl",
                        within_eval=root / "missing-within.jsonl",
                        xpack_eval=root / "missing-xpack.jsonl",
                        feature_eval=root / "missing-feature.jsonl",
                        out=out,
                    )
                )

    def test_pairwise_input_fails_when_required_mode_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            empty = root / "empty.jsonl"
            out = root / "pairwise.jsonl"
            write_jsonl(
                base,
                [
                    {
                        "pack_id": "p",
                        "task_id": "t",
                        "mode": "prompt_only",
                        "status": "success",
                    }
                ],
            )
            write_jsonl(empty, [])

            with self.assertRaisesRegex(ValueError, "missing mode"):
                helper.cmd_pairwise_input(
                    argparse.Namespace(
                        base_eval=base,
                        within_eval=empty,
                        xpack_eval=empty,
                        feature_eval=empty,
                        out=out,
                    )
                )

    def test_pairwise_input_rejects_duplicate_success_eval_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            empty = root / "empty.jsonl"
            out = root / "pairwise.jsonl"
            write_jsonl(
                base,
                [
                    {
                        "pack_id": "p",
                        "task_id": "t",
                        "mode": "prompt_only",
                        "status": "success",
                    },
                    {
                        "pack_id": "p",
                        "task_id": "t",
                        "mode": "prompt_only",
                        "status": "success",
                    },
                ],
            )
            write_jsonl(empty, [])

            with self.assertRaisesRegex(ValueError, "duplicate prompt_only eval cell"):
                helper.cmd_pairwise_input(
                    argparse.Namespace(
                        base_eval=base,
                        within_eval=empty,
                        xpack_eval=empty,
                        feature_eval=empty,
                        out=out,
                    )
                )

    def test_skill_quality_input_rejects_duplicate_skill_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one_shot = root / "one_shot.jsonl"
            empty = root / "empty.jsonl"
            out = root / "quality.jsonl"
            write_jsonl(
                one_shot,
                [
                    {
                        "pack_id": "p",
                        "mode": "one_shot_skill_from_examples",
                        "status": "success",
                        "skill_md": "First.",
                    },
                    {
                        "pack_id": "p",
                        "mode": "one_shot_skill_from_examples",
                        "status": "success",
                        "skill_md": "Duplicate.",
                    },
                ],
            )
            write_jsonl(empty, [])

            with self.assertRaisesRegex(ValueError, "duplicate one_shot_skill skill row"):
                helper.cmd_skill_quality_input(
                    argparse.Namespace(
                        baseline_skills=one_shot,
                        no_memory_skills=empty,
                        within_skills=empty,
                        xpack_skills=empty,
                        feature_skills=empty,
                        out=out,
                    )
                )

    def test_load_scores_fails_when_required_mode_has_no_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "pack_id": "p",
                        "task_id": "t",
                        "mode": "prompt_only",
                        "status": "judge_parse_error",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "no successful scored rows"):
                helper.load_scores({"prompt_only": (path, "prompt_only")})

    def test_load_scores_rejects_duplicate_success_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            row = {
                "pack_id": "p",
                "task_id": "t",
                "mode": "prompt_only",
                "status": "success",
                "overall_score": 4.0,
            }
            write_jsonl(path, [row, dict(row)])

            with self.assertRaisesRegex(ValueError, "duplicate successful prompt_only"):
                helper.load_scores({"prompt_only": (path, "prompt_only")})

    def test_load_scores_rejects_success_without_numeric_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "pack_id": "p",
                        "task_id": "t",
                        "mode": "prompt_only",
                        "status": "success",
                        "overall_score": True,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "missing numeric overall_score"):
                helper.load_scores({"prompt_only": (path, "prompt_only")})


if __name__ == "__main__":
    unittest.main()
