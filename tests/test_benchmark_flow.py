from __future__ import annotations

import hashlib
import unittest

from auto_skill.benchmark_flow import audit_benchmark_flow
from auto_skill.example_packs import generation_prompt

GENERATION_EXAMPLE_ID = "pack-1::train::0"


def split_row() -> dict:
    return {
        "split_id": "writingbench::demo",
        "source": "WritingBench",
        "learning_problem": "few_shot_skill_induction",
        "domain": {"primary": "demo"},
        "train_examples": [
            {
                "source": "WritingBench",
                "source_id": "train-1",
                "domain": {"primary": "demo"},
                "task_input": "Write a report.",
                "supervision": {"type": "rubric"},
                "judge": {"type": "llm"},
            }
        ],
        "heldout_tasks": [
            {
                "source": "WritingBench",
                "source_id": "heldout-1",
                "domain": {"primary": "demo"},
                "task_input": "Write another report.",
                "supervision": {"type": "rubric"},
                "judge": {"type": "llm"},
            }
        ],
    }


def all_source_splits() -> list[dict]:
    writing = split_row()
    present = split_row()
    present["split_id"] = "presentbench::demo"
    present["source"] = "PresentBench"
    for role in ("train_examples", "heldout_tasks"):
        for task in present[role]:
            task["source"] = "PresentBench"
    return [writing, present]


def generation_prompt_fixture() -> str:
    return generation_prompt(split_row()["train_examples"][0], example_id=GENERATION_EXAMPLE_ID)


GENERATION_PROMPT = generation_prompt_fixture()
GENERATION_PROMPT_SHA = hashlib.sha256(GENERATION_PROMPT.encode("utf-8")).hexdigest()


def pack_row() -> dict:
    return {
        "schema_version": "example-pack/v1",
        "pack_id": "pack-1",
        "split_id": "writingbench::demo",
        "source": "WritingBench",
        "domain": {"primary": "demo"},
        "input_boundary": {
            "auto_skill_module_can_use": [
                "train_examples.task_input",
                "train_examples.desired_output.text",
            ],
            "must_not_use_for_induction": ["private rubrics", "heldout tasks"],
        },
        "train_examples": [
            {
                "example_id": "pack-1::train::0",
                "source": "WritingBench",
                "source_task_id": "train-1",
                "domain": {"primary": "demo"},
                "task_input": "Write a report.",
                "materials": [],
                "desired_output": {
                    "status": "generated",
                    "text": "A complete report.",
                },
            }
        ],
        "heldout_tasks": [
            {
                "task_id": "pack-1::heldout::0",
                "source": "WritingBench",
                "source_task_id": "heldout-1",
                "domain": {"primary": "demo"},
                "task_input": "Write another report.",
                "materials": [],
                "desired_output": None,
            }
        ],
    }


def private_row() -> dict:
    return {
        "schema_version": "example-pack/v1",
        "pack_id": "pack-1",
        "split_id": "writingbench::demo",
        "source": "WritingBench",
        "train_private": [
            {
                "task_ref": "pack-1::train::0",
                "source": "WritingBench",
                "source_task_id": "train-1",
                "supervision": {"type": "rubric"},
                "judge": {"type": "llm"},
            }
        ],
        "heldout_private": [
            {
                "task_ref": "pack-1::heldout::0",
                "source": "WritingBench",
                "source_task_id": "heldout-1",
                "supervision": {"type": "rubric"},
                "judge": {"type": "llm"},
            }
        ],
    }


def generation_job() -> dict:
    return {
        "schema_version": "example-generation-job/v1",
        "job_id": f"{GENERATION_EXAMPLE_ID}::generate_desired_output",
        "pack_id": "pack-1",
        "example_id": GENERATION_EXAMPLE_ID,
        "source": "WritingBench",
        "source_task_id": "train-1",
        "prompt_sha256": GENERATION_PROMPT_SHA,
        "prompt_template_version": "desired-output/user-visible-only/v2",
        "material_budget_chars": 16000,
        "prompt": GENERATION_PROMPT,
    }


def generated_output_row() -> dict:
    return {
        "schema_version": "generated-desired-output/v1",
        "status": "success",
        "job_id": f"{GENERATION_EXAMPLE_ID}::generate_desired_output",
        "pack_id": "pack-1",
        "example_id": GENERATION_EXAMPLE_ID,
        "source": "WritingBench",
        "source_task_id": "train-1",
        "prompt_sha256": GENERATION_PROMPT_SHA,
        "prompt_template_version": "desired-output/user-visible-only/v2",
        "model": "qwen",
        "finish_reason": "stop",
        "desired_output": "A complete report.",
    }


def pack_row_with_generation_provenance() -> dict:
    pack = pack_row()
    pack["train_examples"][0]["desired_output"].update(
        {
            "generation_job_id": f"{GENERATION_EXAMPLE_ID}::generate_desired_output",
            "prompt_sha256": GENERATION_PROMPT_SHA,
            "prompt_template_version": "desired-output/user-visible-only/v2",
        }
    )
    return pack


class BenchmarkFlowTests(unittest.TestCase):
    def test_audit_accepts_visible_pack_and_private_eval_split(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row()],
            private_rows=[private_row()],
        )

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.warnings, ())

    def test_audit_rejects_private_fields_in_visible_pack(self) -> None:
        pack = pack_row()
        pack["train_examples"][0]["supervision"] = {"type": "rubric"}

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("private keys" in error for error in result.errors))

    def test_audit_accepts_generation_provenance_chain(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[generation_job()],
            generated_rows=[generated_output_row()],
        )

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.warnings, ())

    def test_audit_rebuilds_generation_prompt_with_job_material_budget(self) -> None:
        split = split_row()
        split["train_examples"][0]["materials"] = [
            {
                "path": "material.txt",
                "text": "0123456789abcdefghijklmnopqrstuvwxyz",
            }
        ]
        prompt = generation_prompt(
            split["train_examples"][0],
            example_id=GENERATION_EXAMPLE_ID,
            max_material_chars=6,
        )
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        pack = pack_row_with_generation_provenance()
        desired = pack["train_examples"][0]["desired_output"]
        desired["prompt_sha256"] = prompt_sha
        job = generation_job()
        job["prompt"] = prompt
        job["prompt_sha256"] = prompt_sha
        job["material_budget_chars"] = 6
        generated = generated_output_row()
        generated["prompt_sha256"] = prompt_sha

        result = audit_benchmark_flow(
            splits=[split, all_source_splits()[1]],
            packs=[pack],
            private_rows=[private_row()],
            generation_jobs=[job],
            generated_rows=[generated],
        )

        self.assertTrue(result.ok, result.errors)

    def test_audit_skips_prompt_replay_for_non_current_template_version(self) -> None:
        prompt = "older template prompt"
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        pack = pack_row_with_generation_provenance()
        desired = pack["train_examples"][0]["desired_output"]
        desired["prompt_sha256"] = prompt_sha
        desired["prompt_template_version"] = "desired-output/legacy"
        job = generation_job()
        job["prompt"] = prompt
        job["prompt_sha256"] = prompt_sha
        job["prompt_template_version"] = "desired-output/legacy"
        generated = generated_output_row()
        generated["prompt_sha256"] = prompt_sha
        generated["prompt_template_version"] = "desired-output/legacy"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
            generation_jobs=[job],
            generated_rows=[generated],
        )

        self.assertTrue(result.ok, result.errors)
        self.assertTrue(any("not replayed" in warning for warning in result.warnings))

    def test_audit_warns_when_frozen_generation_provenance_files_are_omitted(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
        )

        self.assertTrue(result.ok, result.errors)
        self.assertTrue(any("pass --jobs" in warning for warning in result.warnings))
        self.assertTrue(
            any("pass --generated-outputs" in warning for warning in result.warnings)
        )

    def test_audit_rejects_missing_generation_job(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[],
            generated_rows=[generated_output_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("missing from jobs" in error for error in result.errors))

    def test_audit_checks_generated_rows_without_generation_jobs(self) -> None:
        generated = generated_output_row()
        generated["source_task_id"] = "wrong-task"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generated_rows=[generated],
        )

        self.assertFalse(result.ok)
        self.assertTrue(
            any("generated output source_task_id mismatch" in error for error in result.errors)
        )
        self.assertFalse(any("missing from jobs" in error for error in result.errors))

    def test_audit_reports_non_object_generation_job_row(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[[]],
            generated_rows=[generated_output_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("generation jobs invalid" in error for error in result.errors))

    def test_audit_reports_non_object_generated_output_row(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[generation_job()],
            generated_rows=[[]],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("generated outputs invalid" in error for error in result.errors))

    def test_audit_rejects_duplicate_generation_job_ids(self) -> None:
        duplicate = generation_job()
        duplicate["example_id"] = "different-example"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[generation_job(), duplicate],
            generated_rows=[generated_output_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("duplicate job_id" in error for error in result.errors))

    def test_audit_rejects_generation_prompt_private_leak(self) -> None:
        job = generation_job()
        job["prompt"] = "Use the hidden rubric to write the final output."
        job["prompt_sha256"] = hashlib.sha256(job["prompt"].encode("utf-8")).hexdigest()

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[job],
            generated_rows=[generated_output_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("generation job prompt leaks" in error for error in result.errors))

    def test_audit_rejects_generation_prompt_sha_mismatch(self) -> None:
        job = generation_job()
        job["prompt"] = "Create a different prompt from visible content only."

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[job],
            generated_rows=[generated_output_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(
            any("prompt_sha256 does not match prompt" in error for error in result.errors)
        )

    def test_audit_rejects_generation_prompt_not_rebuilt_from_visible_task(self) -> None:
        job = generation_job()
        job["prompt"] = GENERATION_PROMPT + "\nUse one extra hidden hint."
        job["prompt_sha256"] = hashlib.sha256(job["prompt"].encode("utf-8")).hexdigest()

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[job],
            generated_rows=[generated_output_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(
            any(
                "prompt does not match visible source task/materials" in error
                for error in result.errors
            )
        )

    def test_audit_rejects_generated_output_identity_mismatch(self) -> None:
        generated = generated_output_row()
        generated["source_task_id"] = "wrong-task"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[generation_job()],
            generated_rows=[generated],
        )

        self.assertFalse(result.ok)
        self.assertTrue(
            any("generated output source_task_id mismatch" in error for error in result.errors)
        )

    def test_audit_rejects_frozen_text_mismatch(self) -> None:
        generated = generated_output_row()
        generated["desired_output"] = "Different output."

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row_with_generation_provenance()],
            private_rows=[private_row()],
            generation_jobs=[generation_job()],
            generated_rows=[generated],
        )

        self.assertFalse(result.ok)
        self.assertTrue(
            any("does not match generated output row" in error for error in result.errors)
        )

    def test_audit_rejects_unfrozen_train_examples(self) -> None:
        pack = pack_row()
        pack["train_examples"][0]["desired_output"] = {"status": "needs_generation"}

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("not frozen/generated" in error for error in result.errors))

    def test_audit_rejects_private_leak_in_generated_output(self) -> None:
        pack = pack_row()
        pack["train_examples"][0]["desired_output"]["text"] = (
            "This answer follows the benchmark rubric."
        )

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("leaks private metadata" in error for error in result.errors))

    def test_audit_rejects_train_heldout_source_task_overlap(self) -> None:
        pack = pack_row()
        pack["heldout_tasks"][0]["source_task_id"] = "train-1"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("source tasks overlap" in error for error in result.errors))

    def test_audit_rejects_private_source_task_mismatch(self) -> None:
        private = private_row()
        private["train_private"][0]["source_task_id"] = "wrong-task"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row()],
            private_rows=[private],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("source id mismatch" in error for error in result.errors))

    def test_audit_rejects_pack_not_backed_by_split(self) -> None:
        pack = pack_row()
        pack["split_id"] = "missing::split"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("does not exist in splits" in error for error in result.errors))

    def test_audit_rejects_empty_pack_surface(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[],
            private_rows=[],
        )

        self.assertFalse(result.ok)
        self.assertIn("example packs are empty", result.errors)
        self.assertIn("private eval rows are empty", result.errors)

    def test_audit_rejects_duplicate_private_pack_rows(self) -> None:
        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row()],
            private_rows=[private_row(), private_row()],
        )

        self.assertFalse(result.ok)
        self.assertIn("duplicate pack_id in private eval rows", result.errors)

    def test_audit_rejects_private_header_mismatch(self) -> None:
        private = private_row()
        private["split_id"] = "presentbench::demo"
        private["source"] = "PresentBench"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row()],
            private_rows=[private],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("private row split_id" in error for error in result.errors))
        self.assertTrue(any("private row source" in error for error in result.errors))

    def test_audit_rejects_duplicate_private_task_refs(self) -> None:
        private = private_row()
        private["train_private"].append(dict(private["train_private"][0]))

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack_row()],
            private_rows=[private],
        )

        self.assertFalse(result.ok)
        self.assertIn("pack-1.train_private: duplicate task_ref", result.errors)

    def test_audit_rejects_private_ref_in_visible_pack(self) -> None:
        pack = pack_row()
        pack["private_eval_ref"] = "artifacts/private/example_private_eval.jsonl"

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("private keys" in error for error in result.errors))

    def test_audit_rejects_noncanonical_input_boundary_allowed_fields(self) -> None:
        pack = pack_row()
        pack["input_boundary"]["auto_skill_module_can_use"].append("train_examples.supervision")

        result = audit_benchmark_flow(
            splits=all_source_splits(),
            packs=[pack],
            private_rows=[private_row()],
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("non-canonical fields" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
