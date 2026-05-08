from __future__ import annotations

import unittest

from auto_skill.example_packs import build_pack, generation_prompt, prompt_sha256, slugify


def make_task(source_id: str) -> dict:
    return {
        "source": "WritingBench",
        "source_id": source_id,
        "domain": {"primary": "Finance", "secondary": "Report"},
        "task_input": "Write an investment report.",
        "supervision": {"type": "rubric", "items": [{"name": "Depth"}]},
        "judge": {"type": "llm"},
        "expected_artifacts": ["written_response"],
    }


class ExamplePackTests(unittest.TestCase):
    def test_slugify_keeps_stable_ascii_ids(self) -> None:
        self.assertEqual(
            slugify("writingbench::Finance & Business::en"),
            "writingbench_Finance_Business_en",
        )

    def test_build_pack_separates_user_visible_and_private_fields(self) -> None:
        split = {
            "split_id": "writingbench::Finance::Report::en",
            "source": "WritingBench",
            "learning_problem": "few_shot_skill_induction",
            "domain": {"primary": "Finance", "secondary": "Report", "language": "en"},
            "train_examples": [make_task("train-1")],
            "heldout_tasks": [make_task("heldout-1")],
        }

        pack, private, jobs = build_pack(split)

        clean_example = pack["train_examples"][0]
        self.assertNotIn("supervision", clean_example)
        self.assertNotIn("judge", clean_example)
        self.assertEqual(clean_example["desired_output"]["status"], "needs_generation")
        self.assertIn("supervision", private["train_private"][0])
        self.assertNotIn("private_builder_context", jobs[0])
        self.assertEqual(
            jobs[0]["private_eval_ref"],
            "artifacts/private/example_private_eval.jsonl",
        )
        self.assertNotIn("Rubric criteria", jobs[0]["prompt"])
        self.assertEqual(jobs[0]["prompt_sha256"], prompt_sha256(jobs[0]["prompt"]))

    def test_generation_prompt_uses_visible_inputs_only(self) -> None:
        prompt = generation_prompt(make_task("train-1"), example_id="ex-1")

        self.assertIn("Write an investment report.", prompt)
        self.assertNotIn("rubric", prompt.lower())
        self.assertNotIn("judge", prompt.lower())
        self.assertNotIn("checklist", prompt.lower())


if __name__ == "__main__":
    unittest.main()
