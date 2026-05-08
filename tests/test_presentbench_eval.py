from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_skill.presentbench_eval import (
    check_presentbench_official_eval_readiness,
    find_score_artifact,
    load_presentbench_score,
)


class PresentBenchEvalTests(unittest.TestCase):
    def test_readiness_requires_slide_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            case_dir = data_root / "education" / "Course" / "Lecture1"
            generation_dir = case_dir / "generation_task"
            generation_dir.mkdir(parents=True)
            (case_dir / "material.md").write_text("source material", encoding="utf-8")
            (generation_dir / "instructions.md").write_text("make slides", encoding="utf-8")
            (generation_dir / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 100",
                encoding="utf-8",
            )

            readiness = check_presentbench_official_eval_readiness(
                task={
                    "task_id": "task-1",
                    "source_task_id": "education/Course/Lecture1",
                },
                data_root=data_root,
                result_root=result_root,
            )

        self.assertEqual(readiness.status, "missing_official_eval_artifacts")
        self.assertTrue(any("slides.pdf or slides.pptx" in item for item in readiness.missing))

    def test_readiness_accepts_slide_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            case_dir = data_root / "education" / "Course" / "Lecture1"
            generation_dir = case_dir / "generation_task"
            result_dir = (
                result_root / "education" / "Course" / "Lecture1" / "generation_task" / "results"
            )
            generation_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            (case_dir / "material.md").write_text("source material", encoding="utf-8")
            (generation_dir / "instructions.md").write_text("make slides", encoding="utf-8")
            (generation_dir / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 100",
                encoding="utf-8",
            )
            (result_dir / "slides.pdf").write_bytes(b"%PDF-1.4\n")

            readiness = check_presentbench_official_eval_readiness(
                task={
                    "task_id": "task-1",
                    "source_task_id": "education/Course/Lecture1",
                },
                data_root=data_root,
                result_root=result_root,
            )

        self.assertEqual(readiness.status, "ready_for_official_judge")
        self.assertEqual(len(readiness.material_files), 1)

    def test_readiness_requires_official_utils_when_code_root_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            code_root = root / "code"
            case_dir = data_root / "education" / "Course" / "Lecture1"
            generation_dir = case_dir / "generation_task"
            result_dir = (
                result_root / "education" / "Course" / "Lecture1" / "generation_task" / "results"
            )
            generation_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            code_root.mkdir()
            (code_root / "judge_all.py").write_text("", encoding="utf-8")
            (code_root / "judge.py").write_text("", encoding="utf-8")
            (code_root / "scoring.py").write_text("", encoding="utf-8")
            (case_dir / "material.md").write_text("source material", encoding="utf-8")
            (generation_dir / "instructions.md").write_text("make slides", encoding="utf-8")
            (generation_dir / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 100",
                encoding="utf-8",
            )
            (result_dir / "slides.pdf").write_bytes(b"%PDF-1.4\n")

            readiness = check_presentbench_official_eval_readiness(
                task={
                    "task_id": "task-1",
                    "source_task_id": "education/Course/Lecture1",
                },
                data_root=data_root,
                result_root=result_root,
                code_root=code_root,
            )

        self.assertEqual(readiness.status, "missing_official_eval_artifacts")
        self.assertTrue(any("presentbench_utils_paths" in item for item in readiness.missing))
        self.assertTrue(any("presentbench_utils_api_base" in item for item in readiness.missing))

    def test_readiness_requires_libreoffice_for_pptx_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            case_dir = data_root / "education" / "Course" / "Lecture1"
            generation_dir = case_dir / "generation_task"
            result_dir = (
                result_root / "education" / "Course" / "Lecture1" / "generation_task" / "results"
            )
            generation_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            (case_dir / "material.md").write_text("source material", encoding="utf-8")
            (generation_dir / "instructions.md").write_text("make slides", encoding="utf-8")
            (generation_dir / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 100",
                encoding="utf-8",
            )
            (result_dir / "slides.pptx").write_bytes(b"pptx")

            with patch("auto_skill.presentbench_eval.shutil.which", return_value=None):
                readiness = check_presentbench_official_eval_readiness(
                    task={
                        "task_id": "task-1",
                        "source_task_id": "education/Course/Lecture1",
                    },
                    data_root=data_root,
                    result_root=result_root,
                )

        self.assertEqual(readiness.status, "missing_official_eval_artifacts")
        self.assertIn("presentbench_binary: libreoffice", readiness.missing)

    def test_readiness_accepts_official_zero_score_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            case_dir = data_root / "education" / "Course" / "Lecture1"
            generation_dir = case_dir / "generation_task"
            result_dir = (
                result_root / "education" / "Course" / "Lecture1" / "generation_task" / "results"
            )
            generation_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            (case_dir / "material.pdf").write_bytes(b"%PDF-1.4\n")
            (generation_dir / "instructions.md").write_text("make slides", encoding="utf-8")
            (generation_dir / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 100",
                encoding="utf-8",
            )
            (result_dir / "slides_generation_failed.txt").write_text("failed", encoding="utf-8")

            readiness = check_presentbench_official_eval_readiness(
                task={
                    "task_id": "task-1",
                    "source_task_id": "education/Course/Lecture1",
                },
                data_root=data_root,
                result_root=result_root,
            )

        self.assertEqual(readiness.status, "ready_for_zero_score")
        self.assertTrue(readiness.failure_flag)

    def test_readiness_marks_existing_score_artifact_as_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            case_dir = data_root / "education" / "Course" / "Lecture1"
            generation_dir = case_dir / "generation_task"
            result_dir = (
                result_root / "education" / "Course" / "Lecture1" / "generation_task" / "results"
            )
            generation_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            (case_dir / "material.md").write_text("source material", encoding="utf-8")
            (generation_dir / "instructions.md").write_text("make slides", encoding="utf-8")
            (generation_dir / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 100",
                encoding="utf-8",
            )
            (result_dir / "slides.pdf").write_bytes(b"%PDF-1.4\n")
            (result_dir / "gemini_score.yaml").write_text(
                "total:\n  weighted_arithmetic_mean_percent: 62.5\n  yes_count: 5\n",
                encoding="utf-8",
            )

            readiness = check_presentbench_official_eval_readiness(
                task={
                    "task_id": "task-1",
                    "source_task_id": "education/Course/Lecture1",
                },
                data_root=data_root,
                result_root=result_root,
            )

        self.assertEqual(readiness.status, "scored")
        self.assertTrue(readiness.score_artifact)

    def test_find_score_artifact_prefers_latest_matching_judge_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            older = result_dir / "gemini-3-flash-preview_2026-05-07_10-00-00_score.yaml"
            newer = result_dir / "gemini-3-flash-preview_2026-05-07_11-00-00_score.yaml"
            other = result_dir / "other-judge_2026-05-07_12-00-00_score.yaml"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            other.write_text("{}", encoding="utf-8")

            selected = find_score_artifact(
                result_dir,
                judge_model="gemini-3-flash-preview",
            )

        self.assertEqual(selected, newer)

    def test_find_score_artifact_rejects_mixed_judges_without_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "gemini-3-flash-preview_2026-05-07_11-00-00_score.yaml").write_text(
                "{}",
                encoding="utf-8",
            )
            (result_dir / "other-judge_2026-05-07_12-00-00_score.yaml").write_text(
                "{}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "multiple judge models"):
                find_score_artifact(result_dir)

    def test_readiness_marks_mixed_judge_scores_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            case_dir = data_root / "education" / "Course" / "Lecture1"
            generation_dir = case_dir / "generation_task"
            result_dir = (
                result_root / "education" / "Course" / "Lecture1" / "generation_task" / "results"
            )
            generation_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            (case_dir / "material.md").write_text("source material", encoding="utf-8")
            (generation_dir / "instructions.md").write_text("make slides", encoding="utf-8")
            (generation_dir / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 100",
                encoding="utf-8",
            )
            (result_dir / "slides.pdf").write_bytes(b"%PDF-1.4\n")
            (result_dir / "gemini-3-flash-preview_2026-05-07_11-00-00_score.yaml").write_text(
                "total:\n  weighted_arithmetic_mean_percent: 62.5\n",
                encoding="utf-8",
            )
            (result_dir / "other-judge_2026-05-07_12-00-00_score.yaml").write_text(
                "total:\n  weighted_arithmetic_mean_percent: 12.5\n",
                encoding="utf-8",
            )

            readiness = check_presentbench_official_eval_readiness(
                task={
                    "task_id": "task-1",
                    "source_task_id": "education/Course/Lecture1",
                },
                data_root=data_root,
                result_root=result_root,
            )

        self.assertEqual(readiness.status, "ambiguous_score_artifacts")
        self.assertIn("multiple judge models", readiness.score_selection_error or "")

    def test_load_presentbench_score_extracts_official_total_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "score.yaml"
            score_path.write_text(
                "total:\n"
                "  weighted_arithmetic_mean_percent: 73.25\n"
                "  yes_count: 10\n"
                "  valid_count: 20\n",
                encoding="utf-8",
            )

            score = load_presentbench_score(score_path)

        self.assertEqual(score["score_percent"], 73.25)
        self.assertEqual(score["yes_count"], 10)

    def test_load_presentbench_score_rejects_boolean_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp) / "score.yaml"
            score_path.write_text(
                "total:\n"
                "  weighted_arithmetic_mean_percent: true\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "weighted_arithmetic_mean_percent"):
                load_presentbench_score(score_path)


if __name__ == "__main__":
    unittest.main()
