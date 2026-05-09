from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from auto_skill.example_packs import generation_prompt
from scripts.ops.report_expanded_cleaning_status import coverage_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "report_expanded_cleaning_status.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def split_row(source: str, split_id: str) -> dict:
    return {
        "split_id": split_id,
        "source": source,
        "learning_problem": "few_shot_skill_induction",
        "domain": {"primary": source},
        "train_examples": [
            {
                "source": source,
                "source_id": f"{source}-train",
                "domain": {"primary": source},
                "task_input": "Task",
                "supervision": {"items": [{"name": "Quality"}]},
                "judge": {"type": "llm"},
            }
        ],
        "heldout_tasks": [
            {
                "source": source,
                "source_id": f"{source}-heldout",
                "domain": {"primary": source},
                "task_input": "Heldout",
                "supervision": {"items": [{"name": "Quality"}]},
                "judge": {"type": "llm"},
            }
        ],
    }


def job_id_for_pack(pack_id: str) -> str:
    return f"{pack_id}::train::0::generate_desired_output"


def generation_prompt_for(source: str, pack_id: str) -> str:
    return generation_prompt(
        split_row(source, f"{source.lower()}::demo")["train_examples"][0],
        example_id=f"{pack_id}::train::0",
    )


def generation_prompt_sha_for(source: str, pack_id: str) -> str:
    prompt = generation_prompt_for(source, pack_id)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def pack_row(source: str, split_id: str, pack_id: str, *, job_id: str | None = None) -> dict:
    generation_job_id = job_id or job_id_for_pack(pack_id)
    prompt_sha = generation_prompt_sha_for(source, pack_id)
    return {
        "schema_version": "example-pack/v1",
        "pack_id": pack_id,
        "split_id": split_id,
        "source": source,
        "domain": {"primary": source},
        "input_boundary": {
            "auto_skill_module_can_use": [
                "train_examples.task_input",
                "train_examples.desired_output.text",
            ],
            "must_not_use_for_induction": ["private rubrics", "heldout tasks"],
        },
        "train_examples": [
            {
                "example_id": f"{pack_id}::train::0",
                "source": source,
                "source_task_id": f"{source}-train",
                "domain": {"primary": source},
                "task_input": "Task",
                "materials": [],
                "desired_output": {
                    "status": "generated",
                    "text": "Output",
                    "generation_job_id": generation_job_id,
                    "prompt_sha256": prompt_sha,
                    "prompt_template_version": "desired-output/user-visible-only/v2",
                },
            }
        ],
        "heldout_tasks": [
            {
                "task_id": f"{pack_id}::heldout::0",
                "source": source,
                "source_task_id": f"{source}-heldout",
                "domain": {"primary": source},
                "task_input": "Heldout",
                "materials": [],
                "desired_output": None,
            }
        ],
    }


def private_row(source: str, split_id: str, pack_id: str) -> dict:
    return {
        "schema_version": "example-pack/v1",
        "pack_id": pack_id,
        "split_id": split_id,
        "source": source,
        "train_private": [
            {
                "task_ref": f"{pack_id}::train::0",
                "source": source,
                "source_task_id": f"{source}-train",
                "supervision": {"items": [{"name": "Quality"}]},
                "judge": {"type": "llm"},
            }
        ],
        "heldout_private": [
            {
                "task_ref": f"{pack_id}::heldout::0",
                "source": source,
                "source_task_id": f"{source}-heldout",
                "supervision": {"items": [{"name": "Quality"}]},
                "judge": {"type": "llm"},
            }
        ],
    }


def generation_job_row(source: str, pack_id: str, *, job_id: str | None = None) -> dict:
    prompt = generation_prompt_for(source, pack_id)
    return {
        "job_id": job_id or job_id_for_pack(pack_id),
        "pack_id": pack_id,
        "example_id": f"{pack_id}::train::0",
        "source": source,
        "source_task_id": f"{source}-train",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_template_version": "desired-output/user-visible-only/v2",
        "prompt": prompt,
    }


def generation_row(job_id: str, pack_id: str, source: str = "WritingBench") -> dict:
    return {
        "schema_version": "generated-desired-output/v1",
        "status": "success",
        "job_id": job_id,
        "pack_id": pack_id,
        "example_id": f"{pack_id}::train::0",
        "source": source,
        "source_task_id": f"{source}-train",
        "prompt_sha256": generation_prompt_sha_for(source, pack_id),
        "prompt_template_version": "desired-output/user-visible-only/v2",
        "desired_output": "Output",
        "finish_reason": "stop",
    }


def audit_row(pack_id: str, source: str, status: str = "success") -> dict:
    return {
        "schema_version": "train-example-quality-audit/v1",
        "pack_id": pack_id,
        "task_id": f"{pack_id}::train::0",
        "source": source,
        "mode": "desired_output",
        "evaluator_kind": "private_train_example_quality_audit",
        "status": status,
        "overall_score": 8 if status == "success" else None,
    }


class ExpandedCleaningStatusCliTests(unittest.TestCase):
    def test_coverage_summary_handles_dict_string_and_missing_domains(self) -> None:
        self.assertEqual(
            coverage_summary(
                [
                    {
                        "source": "WritingBench",
                        "domain": {
                            "primary": "Finance & Business",
                            "secondary": "Tender Document",
                            "language": "zh",
                        },
                    },
                    {"source": "LegacyBench", "domain": "General"},
                    {"source": "UnknownBench"},
                ]
            ),
            {
                "LegacyBench": {"primary_domains": {"General": 1}},
                "UnknownBench": {},
                "WritingBench": {
                    "languages": {"zh": 1},
                    "primary_domains": {"Finance & Business": 1},
                    "secondary_domains": {"Tender Document": 1},
                },
            },
        )

    def test_reports_ready_for_complete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            splits = [
                split_row("WritingBench", "writing::demo"),
                split_row("PresentBench", "present::demo"),
            ]
            packs = [
                pack_row("WritingBench", "writing::demo", "pack-writing"),
                pack_row("PresentBench", "present::demo", "pack-present"),
            ]
            private_rows = [
                private_row("WritingBench", "writing::demo", "pack-writing"),
                private_row("PresentBench", "present::demo", "pack-present"),
            ]
            jobs = [
                generation_job_row("WritingBench", "pack-writing"),
                generation_job_row("PresentBench", "pack-present"),
            ]
            write_jsonl(tmp / "splits.jsonl", splits)
            write_jsonl(tmp / "packs.jsonl", packs)
            write_jsonl(tmp / "private.jsonl", private_rows)
            write_jsonl(tmp / "jobs.jsonl", jobs)
            write_jsonl(
                tmp / "generated.jsonl",
                [
                    generation_row(job_id_for_pack("pack-writing"), "pack-writing"),
                    generation_row(job_id_for_pack("pack-present"), "pack-present", "PresentBench"),
                ],
            )
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])
            write_jsonl(tmp / "audit-present.jsonl", [audit_row("pack-present", "PresentBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--required-audit",
                    "audit-present.jsonl",
                    "--expect-packs",
                    "2",
                    "--expect-train-examples",
                    "2",
                    "--expect-heldout-tasks",
                    "2",
                    "--expect-generation-jobs",
                    "2",
                    "--expect-status",
                    "ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(
                report["artifacts"]["packs"]["coverage"]["WritingBench"][
                    "primary_domains"
                ],
                {"WritingBench": 1},
            )
            self.assertEqual(
                report["artifacts"]["splits"]["coverage"]["PresentBench"][
                    "primary_domains"
                ],
                {"PresentBench": 1},
            )

    def test_reports_optional_mimo_subset_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            splits = [
                split_row("WritingBench", "writing::demo"),
                split_row("PresentBench", "present::demo"),
            ]
            packs = [
                pack_row("WritingBench", "writing::demo", "pack-writing"),
                pack_row("PresentBench", "present::demo", "pack-present"),
            ]
            private_rows = [
                private_row("WritingBench", "writing::demo", "pack-writing"),
                private_row("PresentBench", "present::demo", "pack-present"),
            ]
            jobs = [
                generation_job_row("WritingBench", "pack-writing"),
                generation_job_row("PresentBench", "pack-present"),
            ]
            generated = [
                generation_row(job_id_for_pack("pack-writing"), "pack-writing"),
                generation_row(job_id_for_pack("pack-present"), "pack-present", "PresentBench"),
            ]
            write_jsonl(tmp / "splits.jsonl", splits)
            write_jsonl(tmp / "packs.jsonl", packs)
            write_jsonl(tmp / "private.jsonl", private_rows)
            write_jsonl(tmp / "jobs.jsonl", jobs)
            write_jsonl(tmp / "generated.jsonl", generated)
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])
            write_jsonl(tmp / "audit-present.jsonl", [audit_row("pack-present", "PresentBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--required-audit",
                    "audit-present.jsonl",
                    "--mimo-subset-packs",
                    "packs.jsonl",
                    "--mimo-subset-private-eval",
                    "private.jsonl",
                    "--mimo-subset-jobs",
                    "jobs.jsonl",
                    "--mimo-subset-generated-outputs",
                    "generated.jsonl",
                    "--expect-mimo-subset-packs",
                    "2",
                    "--expect-mimo-subset-train-examples",
                    "2",
                    "--expect-mimo-subset-heldout-tasks",
                    "2",
                    "--expect-mimo-subset-generation-jobs",
                    "2",
                    "--expect-mimo-subset-writing-packs",
                    "1",
                    "--expect-mimo-subset-present-packs",
                    "1",
                    "--expect-packs",
                    "2",
                    "--expect-train-examples",
                    "2",
                    "--expect-heldout-tasks",
                    "2",
                    "--expect-generation-jobs",
                    "2",
                    "--expect-status",
                    "ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["artifacts"]["mimo_subset"]["status"], "ok")
            self.assertEqual(report["artifacts"]["mimo_subset"]["packs"]["packs"], 2)
            self.assertEqual(
                report["artifacts"]["mimo_subset"]["packs"]["coverage"]["WritingBench"][
                    "primary_domains"
                ],
                {"WritingBench": 1},
            )
            self.assertEqual(
                report["artifacts"]["mimo_subset"]["generated_outputs"]["latest_status_counts"],
                {"success": 2},
            )

    def test_require_mimo_subset_fails_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            split = split_row("WritingBench", "writing::demo")
            pack = pack_row("WritingBench", "writing::demo", "pack-writing")
            write_jsonl(tmp / "splits.jsonl", [split, split_row("PresentBench", "present::demo")])
            write_jsonl(tmp / "packs.jsonl", [pack])
            write_jsonl(
                tmp / "private.jsonl",
                [private_row("WritingBench", "writing::demo", "pack-writing")],
            )
            write_jsonl(tmp / "jobs.jsonl", [generation_job_row("WritingBench", "pack-writing")])
            write_jsonl(
                tmp / "generated.jsonl",
                [generation_row(job_id_for_pack("pack-writing"), "pack-writing")],
            )
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--expect-packs",
                    "1",
                    "--expect-train-examples",
                    "1",
                    "--expect-heldout-tasks",
                    "1",
                    "--expect-generation-jobs",
                    "1",
                    "--require-mimo-subset",
                    "--expect-status",
                    "not_ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "not_ready")
            self.assertTrue(
                any("MIMO subset artifacts missing" in error for error in report["errors"])
            )

    def test_require_mimo_subset_fails_when_subset_is_self_consistent_but_incomplete(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            split = split_row("WritingBench", "writing::demo")
            pack = pack_row("WritingBench", "writing::demo", "pack-writing")
            write_jsonl(tmp / "splits.jsonl", [split, split_row("PresentBench", "present::demo")])
            write_jsonl(tmp / "packs.jsonl", [pack])
            write_jsonl(
                tmp / "private.jsonl",
                [private_row("WritingBench", "writing::demo", "pack-writing")],
            )
            write_jsonl(tmp / "jobs.jsonl", [generation_job_row("WritingBench", "pack-writing")])
            write_jsonl(
                tmp / "generated.jsonl",
                [generation_row(job_id_for_pack("pack-writing"), "pack-writing")],
            )
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--mimo-subset-packs",
                    "packs.jsonl",
                    "--mimo-subset-private-eval",
                    "private.jsonl",
                    "--mimo-subset-jobs",
                    "jobs.jsonl",
                    "--mimo-subset-generated-outputs",
                    "generated.jsonl",
                    "--expect-packs",
                    "1",
                    "--expect-train-examples",
                    "1",
                    "--expect-heldout-tasks",
                    "1",
                    "--expect-generation-jobs",
                    "1",
                    "--require-mimo-subset",
                    "--expect-status",
                    "not_ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            errors = json.loads(result.stdout)["errors"]
            self.assertTrue(any("MIMO subset expected 15 packs" in error for error in errors))
            self.assertTrue(any("MIMO subset source counts mismatch" in error for error in errors))

    def test_optional_mimo_subset_invalid_file_is_structured_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            splits = [
                split_row("WritingBench", "writing::demo"),
                split_row("PresentBench", "present::demo"),
            ]
            packs = [
                pack_row("WritingBench", "writing::demo", "pack-writing"),
                pack_row("PresentBench", "present::demo", "pack-present"),
            ]
            private_rows = [
                private_row("WritingBench", "writing::demo", "pack-writing"),
                private_row("PresentBench", "present::demo", "pack-present"),
            ]
            jobs = [
                generation_job_row("WritingBench", "pack-writing"),
                generation_job_row("PresentBench", "pack-present"),
            ]
            generated = [
                generation_row(job_id_for_pack("pack-writing"), "pack-writing"),
                generation_row(job_id_for_pack("pack-present"), "pack-present", "PresentBench"),
            ]
            write_jsonl(tmp / "splits.jsonl", splits)
            write_jsonl(tmp / "packs.jsonl", packs)
            write_jsonl(tmp / "private.jsonl", private_rows)
            write_jsonl(tmp / "jobs.jsonl", jobs)
            write_jsonl(tmp / "generated.jsonl", generated)
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])
            write_jsonl(tmp / "audit-present.jsonl", [audit_row("pack-present", "PresentBench")])
            (tmp / "bad.jsonl").write_text("{not-json}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--required-audit",
                    "audit-present.jsonl",
                    "--mimo-subset-packs",
                    "bad.jsonl",
                    "--mimo-subset-private-eval",
                    "private.jsonl",
                    "--mimo-subset-jobs",
                    "jobs.jsonl",
                    "--mimo-subset-generated-outputs",
                    "generated.jsonl",
                    "--expect-packs",
                    "2",
                    "--expect-train-examples",
                    "2",
                    "--expect-heldout-tasks",
                    "2",
                    "--expect-generation-jobs",
                    "2",
                    "--expect-status",
                    "ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "ready")
            self.assertTrue(
                any("MIMO subset artifacts invalid" in item for item in report["warnings"])
            )

    def test_skip_mimo_subset_ignores_misaligned_subset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            splits = [
                split_row("WritingBench", "writing::demo"),
                split_row("PresentBench", "present::demo"),
            ]
            packs = [
                pack_row("WritingBench", "writing::demo", "pack-writing"),
                pack_row("PresentBench", "present::demo", "pack-present"),
            ]
            private_rows = [
                private_row("WritingBench", "writing::demo", "pack-writing"),
                private_row("PresentBench", "present::demo", "pack-present"),
            ]
            jobs = [
                generation_job_row("WritingBench", "pack-writing"),
                generation_job_row("PresentBench", "pack-present"),
            ]
            generated = [
                generation_row(job_id_for_pack("pack-writing"), "pack-writing"),
                generation_row(job_id_for_pack("pack-present"), "pack-present", "PresentBench"),
            ]
            write_jsonl(tmp / "splits.jsonl", splits)
            write_jsonl(tmp / "packs.jsonl", packs)
            write_jsonl(tmp / "private.jsonl", private_rows)
            write_jsonl(tmp / "jobs.jsonl", jobs)
            write_jsonl(tmp / "generated.jsonl", generated)
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])
            write_jsonl(tmp / "audit-present.jsonl", [audit_row("pack-present", "PresentBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--required-audit",
                    "audit-present.jsonl",
                    "--optional-audit",
                    "audit-present.jsonl",
                    "--mimo-subset-packs",
                    "does-not-exist.jsonl",
                    "--skip-mimo-subset",
                    "--expect-packs",
                    "2",
                    "--expect-train-examples",
                    "2",
                    "--expect-heldout-tasks",
                    "2",
                    "--expect-generation-jobs",
                    "2",
                    "--expect-status",
                    "ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["warnings"], [])
            self.assertNotIn("mimo_subset", report["artifacts"])

    def test_optional_mimo_subset_bad_row_shape_is_structured_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            splits = [
                split_row("WritingBench", "writing::demo"),
                split_row("PresentBench", "present::demo"),
            ]
            packs = [
                pack_row("WritingBench", "writing::demo", "pack-writing"),
                pack_row("PresentBench", "present::demo", "pack-present"),
            ]
            private_rows = [
                private_row("WritingBench", "writing::demo", "pack-writing"),
                private_row("PresentBench", "present::demo", "pack-present"),
            ]
            jobs = [
                generation_job_row("WritingBench", "pack-writing"),
                generation_job_row("PresentBench", "pack-present"),
            ]
            generated = [
                generation_row(job_id_for_pack("pack-writing"), "pack-writing"),
                generation_row(job_id_for_pack("pack-present"), "pack-present", "PresentBench"),
            ]
            write_jsonl(tmp / "splits.jsonl", splits)
            write_jsonl(tmp / "packs.jsonl", packs)
            write_jsonl(tmp / "private.jsonl", private_rows)
            write_jsonl(tmp / "jobs.jsonl", jobs)
            write_jsonl(tmp / "generated.jsonl", generated)
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])
            write_jsonl(tmp / "audit-present.jsonl", [audit_row("pack-present", "PresentBench")])
            (tmp / "bad-shape.jsonl").write_text("[]\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--required-audit",
                    "audit-present.jsonl",
                    "--mimo-subset-packs",
                    "bad-shape.jsonl",
                    "--mimo-subset-private-eval",
                    "private.jsonl",
                    "--mimo-subset-jobs",
                    "jobs.jsonl",
                    "--mimo-subset-generated-outputs",
                    "generated.jsonl",
                    "--expect-packs",
                    "2",
                    "--expect-train-examples",
                    "2",
                    "--expect-heldout-tasks",
                    "2",
                    "--expect-generation-jobs",
                    "2",
                    "--expect-status",
                    "ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "ready")
            self.assertTrue(any("row 1 must be an object" in item for item in report["warnings"]))

    def test_optional_mimo_subset_allows_zero_generation_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            splits = [
                split_row("WritingBench", "writing::demo"),
                split_row("PresentBench", "present::demo"),
            ]
            packs = [
                pack_row("WritingBench", "writing::demo", "pack-writing"),
                pack_row("PresentBench", "present::demo", "pack-present"),
            ]
            private_rows = [
                private_row("WritingBench", "writing::demo", "pack-writing"),
                private_row("PresentBench", "present::demo", "pack-present"),
            ]
            jobs = [
                generation_job_row("WritingBench", "pack-writing"),
                generation_job_row("PresentBench", "pack-present"),
            ]
            generated = [
                generation_row(job_id_for_pack("pack-writing"), "pack-writing"),
                generation_row(job_id_for_pack("pack-present"), "pack-present", "PresentBench"),
            ]
            mimo_packs = [
                pack_row("WritingBench", "writing::demo", "mimo-writing"),
                pack_row("PresentBench", "present::demo", "mimo-present"),
            ]
            for pack in mimo_packs:
                desired = pack["train_examples"][0]["desired_output"]
                desired.pop("generation_job_id")
                desired.pop("prompt_sha256")
            mimo_private_rows = [
                private_row("WritingBench", "writing::demo", "mimo-writing"),
                private_row("PresentBench", "present::demo", "mimo-present"),
            ]
            write_jsonl(tmp / "splits.jsonl", splits)
            write_jsonl(tmp / "packs.jsonl", packs)
            write_jsonl(tmp / "private.jsonl", private_rows)
            write_jsonl(tmp / "jobs.jsonl", jobs)
            write_jsonl(tmp / "generated.jsonl", generated)
            write_jsonl(tmp / "mimo-packs.jsonl", mimo_packs)
            write_jsonl(tmp / "mimo-private.jsonl", mimo_private_rows)
            write_jsonl(tmp / "mimo-jobs.jsonl", [])
            write_jsonl(tmp / "mimo-generated.jsonl", [])
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])
            write_jsonl(tmp / "audit-present.jsonl", [audit_row("pack-present", "PresentBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--required-audit",
                    "audit-present.jsonl",
                    "--mimo-subset-packs",
                    "mimo-packs.jsonl",
                    "--mimo-subset-private-eval",
                    "mimo-private.jsonl",
                    "--mimo-subset-jobs",
                    "mimo-jobs.jsonl",
                    "--mimo-subset-generated-outputs",
                    "mimo-generated.jsonl",
                    "--expect-mimo-subset-packs",
                    "2",
                    "--expect-mimo-subset-train-examples",
                    "2",
                    "--expect-mimo-subset-heldout-tasks",
                    "2",
                    "--expect-mimo-subset-generation-jobs",
                    "0",
                    "--expect-mimo-subset-writing-packs",
                    "1",
                    "--expect-mimo-subset-present-packs",
                    "1",
                    "--expect-packs",
                    "2",
                    "--expect-train-examples",
                    "2",
                    "--expect-heldout-tasks",
                    "2",
                    "--expect-generation-jobs",
                    "2",
                    "--expect-status",
                    "ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(
                report["artifacts"]["mimo_subset"]["generated_outputs"][
                    "latest_status_counts"
                ],
                {},
            )

    def test_reports_not_ready_when_latest_generation_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            split = split_row("WritingBench", "writing::demo")
            pack = pack_row("WritingBench", "writing::demo", "pack-writing")
            write_jsonl(tmp / "splits.jsonl", [split, split_row("PresentBench", "present::demo")])
            write_jsonl(tmp / "packs.jsonl", [pack])
            write_jsonl(
                tmp / "private.jsonl",
                [private_row("WritingBench", "writing::demo", "pack-writing")],
            )
            write_jsonl(
                tmp / "jobs.jsonl",
                [generation_job_row("WritingBench", "pack-writing")],
            )
            failed = generation_row(job_id_for_pack("pack-writing"), "pack-writing") | {
                "status": "rejected_incomplete_generation",
                "finish_reason": "length",
            }
            failed.pop("desired_output")
            write_jsonl(tmp / "generated.jsonl", [failed])
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--expect-packs",
                    "1",
                    "--expect-train-examples",
                    "1",
                    "--expect-heldout-tasks",
                    "1",
                    "--expect-generation-jobs",
                    "1",
                    "--expect-status",
                    "not_ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "not_ready")
            self.assertTrue(any("not all success" in error for error in report["errors"]))

    def test_reports_not_ready_when_generation_keys_do_not_match_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            split = split_row("WritingBench", "writing::demo")
            pack = pack_row(
                "WritingBench",
                "writing::demo",
                "pack-writing",
                job_id="expected-job",
            )
            write_jsonl(tmp / "splits.jsonl", [split, split_row("PresentBench", "present::demo")])
            write_jsonl(tmp / "packs.jsonl", [pack])
            write_jsonl(
                tmp / "private.jsonl",
                [private_row("WritingBench", "writing::demo", "pack-writing")],
            )
            write_jsonl(
                tmp / "jobs.jsonl",
                [generation_job_row("WritingBench", "pack-writing", job_id="expected-job")],
            )
            write_jsonl(tmp / "generated.jsonl", [generation_row("wrong-job", "pack-writing")])
            write_jsonl(tmp / "audit-writing.jsonl", [audit_row("pack-writing", "WritingBench")])

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-writing.jsonl",
                    "--expect-packs",
                    "1",
                    "--expect-train-examples",
                    "1",
                    "--expect-heldout-tasks",
                    "1",
                    "--expect-generation-jobs",
                    "1",
                    "--expect-status",
                    "not_ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            errors = json.loads(result.stdout)["errors"]
            self.assertTrue(
                any("keys do not match jobs" in error for error in errors)
            )

    def test_required_audit_spec_enforces_source_and_min_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            splits = [
                split_row("WritingBench", "writing::demo"),
                split_row("PresentBench", "present::demo"),
            ]
            packs = [
                pack_row("WritingBench", "writing::demo", "pack-writing"),
                pack_row("PresentBench", "present::demo", "pack-present"),
            ]
            private_rows = [
                private_row("WritingBench", "writing::demo", "pack-writing"),
                private_row("PresentBench", "present::demo", "pack-present"),
            ]
            write_jsonl(tmp / "splits.jsonl", splits)
            write_jsonl(tmp / "packs.jsonl", packs)
            write_jsonl(tmp / "private.jsonl", private_rows)
            write_jsonl(
                tmp / "jobs.jsonl",
                [
                    generation_job_row("WritingBench", "pack-writing"),
                    generation_job_row("PresentBench", "pack-present"),
                ],
            )
            write_jsonl(
                tmp / "generated.jsonl",
                [
                    generation_row(job_id_for_pack("pack-writing"), "pack-writing"),
                    generation_row(job_id_for_pack("pack-present"), "pack-present", "PresentBench"),
                ],
            )
            write_jsonl(
                tmp / "audit-wrong-source.jsonl",
                [audit_row("pack-writing", "WritingBench")],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--splits",
                    "splits.jsonl",
                    "--packs",
                    "packs.jsonl",
                    "--private-eval",
                    "private.jsonl",
                    "--jobs",
                    "jobs.jsonl",
                    "--generated-outputs",
                    "generated.jsonl",
                    "--required-audit",
                    "audit-wrong-source.jsonl:PresentBench:2",
                    "--expect-packs",
                    "2",
                    "--expect-train-examples",
                    "2",
                    "--expect-heldout-tasks",
                    "2",
                    "--expect-generation-jobs",
                    "2",
                    "--expect-status",
                    "not_ready",
                ],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            errors = json.loads(result.stdout)["errors"]
            self.assertTrue(any("source mismatch" in error for error in errors))
            self.assertTrue(any("too few success rows" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
