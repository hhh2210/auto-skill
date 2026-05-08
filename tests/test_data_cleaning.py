from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from auto_skill.data_cleaning import (
    ValidationError,
    summarize_splits,
    validate_split,
    validate_splits,
)
from scripts.data.build_fewshot_splits import (
    KNOWN_BAD_PRESENTBENCH_CASES,
    summarize_group_selection,
)
from scripts.data.inspect_benchmarks import presentbench_case_dirs


def make_task(source_id: str) -> dict:
    return {
        "source": "WritingBench",
        "source_id": source_id,
        "domain": {"primary": "analysis"},
        "task_input": "Write a short analysis.",
        "supervision": {"type": "rubric"},
        "judge": {"type": "llm"},
    }


class DataCleaningTests(unittest.TestCase):
    def test_validate_split_accepts_minimal_valid_split(self) -> None:
        split = {
            "split_id": "writingbench::analysis",
            "source": "WritingBench",
            "learning_problem": "few_shot_skill_induction",
            "domain": {"primary": "analysis"},
            "train_examples": [make_task("train-1")],
            "heldout_tasks": [make_task("heldout-1")],
        }

        validate_split(split)

    def test_validate_split_rejects_train_heldout_overlap(self) -> None:
        split = {
            "split_id": "writingbench::analysis",
            "source": "WritingBench",
            "learning_problem": "few_shot_skill_induction",
            "domain": {"primary": "analysis"},
            "train_examples": [make_task("same")],
            "heldout_tasks": [make_task("same")],
        }

        with self.assertRaises(ValidationError):
            validate_split(split)

    def test_validate_split_rejects_duplicate_train_tasks(self) -> None:
        split = {
            "split_id": "writingbench::analysis",
            "source": "WritingBench",
            "learning_problem": "few_shot_skill_induction",
            "domain": {"primary": "analysis"},
            "train_examples": [make_task("same"), make_task("same")],
            "heldout_tasks": [make_task("heldout-1")],
        }

        with self.assertRaises(ValidationError):
            validate_split(split)

    def test_summarize_splits_counts_sources_and_tasks(self) -> None:
        rows = [
            {
                "source": "WritingBench",
                "train_examples": [make_task("t1"), make_task("t2")],
                "heldout_tasks": [make_task("h1")],
            }
        ]

        self.assertEqual(
            summarize_splits(rows),
            {
                "num_splits": 1,
                "sources": {"WritingBench": 1},
                "train_examples": 2,
                "heldout_tasks": 1,
            },
        )

    def test_validate_splits_rejects_global_duplicate_tasks(self) -> None:
        rows = [
            {
                "split_id": "split-1",
                "source": "WritingBench",
                "learning_problem": "few_shot_skill_induction",
                "domain": {"primary": "analysis"},
                "train_examples": [make_task("same")],
                "heldout_tasks": [make_task("heldout-1")],
            },
            {
                "split_id": "split-2",
                "source": "WritingBench",
                "learning_problem": "few_shot_skill_induction",
                "domain": {"primary": "analysis"},
                "train_examples": [make_task("other")],
                "heldout_tasks": [make_task("same")],
            },
        ]

        with self.assertRaises(ValidationError):
            validate_splits(rows)

    def test_group_selection_summary_exposes_denominator(self) -> None:
        summary = summarize_group_selection(
            {
                ("a",): [1, 2, 3, 4, 5],
                ("b",): [1, 2, 3],
                ("c",): [1, 2, 3, 4, 5, 6],
            },
            train_size=3,
            heldout_size=2,
            max_groups=1,
            selected_groups=1,
        )

        self.assertEqual(summary["total_groups"], 3)
        self.assertEqual(summary["eligible_groups"], 2)
        self.assertEqual(summary["skipped_too_small"], 1)
        self.assertEqual(summary["truncated_by_max_groups"], 1)
        self.assertGreaterEqual(len(KNOWN_BAD_PRESENTBENCH_CASES), 1)

    def test_presentbench_case_dirs_returns_leaf_cases(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_a = root / "academia" / "CVPR_2023" / "paper_a"
            case_b = root / "academia" / "CVPR_2023" / "paper_b"
            for case_dir in (case_a, case_b):
                generation_dir = case_dir / "generation_task"
                generation_dir.mkdir(parents=True)
                (generation_dir / "instructions.md").write_text("Make slides.", encoding="utf-8")

            self.assertEqual(presentbench_case_dirs(root), [case_a, case_b])


if __name__ == "__main__":
    unittest.main()
