from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from auto_skill.grounding import (
    build_grounding_prompt,
    heldout_evidence_text,
    parse_grounding_report,
    summarize_grounding_rows,
    truncate_text,
)
from auto_skill.llm import ChatCompletionConfig
from scripts.eval.run_heldout_eval import append_checkpoint_row
from scripts.metrics.run_grounding_eval import (
    build_jobs,
    load_compatible_resume_success_rows,
    main,
    row_matches_runtime,
    runtime_metadata,
)


class GroundingTests(unittest.TestCase):
    def test_prompt_marks_examples_as_not_factual_evidence(self) -> None:
        prompt = build_grounding_prompt(
            heldout_task={
                "task_id": "task-1",
                "task_input": "Write a report about project Alpha.",
                "materials": [{"path": "alpha.md", "text": "Alpha uses Method X."}],
            },
            candidate_output="Alpha uses Method Y.",
            mode="task_first_operational_anchors",
        )

        self.assertIn("User examples or induced skills are not factual evidence", prompt)
        self.assertIn("Alpha uses Method X", prompt)
        self.assertIn("Alpha uses Method Y", prompt)

    def test_parse_rejects_boolean_scores(self) -> None:
        report = parse_grounding_report(
            '{"grounding_score": true, "hallucination_risk": 3, '
            '"unsupported_claim_count": 1, "unsupported_claims": []}'
        )

        self.assertIn("grounding_score_not_numeric", report["parse_error"])

    def test_parse_accepts_valid_report(self) -> None:
        report = parse_grounding_report(
            '{"grounding_score": 8, "hallucination_risk": 2, '
            '"unsupported_claim_count": 0, "unsupported_claims": [], '
            '"supported_detail_examples": ["Method X"], "summary": "grounded"}'
        )

        self.assertNotIn("parse_error", report)

    def test_truncation_marker_is_visible(self) -> None:
        self.assertEqual(truncate_text("abc", max_chars=10), "abc")
        self.assertIn("[TRUNCATED after 3 chars]", truncate_text("abcdef", max_chars=3))

    def test_heldout_evidence_includes_material_path_and_text(self) -> None:
        text = heldout_evidence_text(
            {
                "task_input": "Use the source.",
                "materials": [{"path": "source.md", "text": "Verified fact."}],
            },
            max_chars=1000,
        )

        self.assertIn("source.md", text)
        self.assertIn("Verified fact.", text)

    def test_summary_groups_success_rows_by_mode(self) -> None:
        summary = summarize_grounding_rows(
            [
                {
                    "mode": "a",
                    "status": "success",
                    "grounding_report": {
                        "grounding_score": 8,
                        "hallucination_risk": 2,
                        "unsupported_claim_count": 1,
                    },
                },
                {
                    "mode": "a",
                    "status": "judge_parse_error",
                    "grounding_report": None,
                },
            ]
        )

        self.assertEqual(summary["status_counts"]["success"], 1)
        self.assertEqual(summary["modes"]["a"]["count"], 2)
        self.assertEqual(summary["modes"]["a"]["success"], 1)
        self.assertEqual(summary["modes"]["a"]["mean_grounding_score"], 8)

    def test_build_jobs_reports_missing_context(self) -> None:
        jobs, skipped = build_jobs(
            packs=[],
            candidate_rows=[
                {
                    "pack_id": "pack-1",
                    "task_id": "task-1",
                    "mode": "m",
                    "status": "success",
                    "generation": {"text": "candidate"},
                }
            ],
            modes=None,
            pack_ids=None,
        )

        self.assertEqual(jobs, [])
        self.assertEqual(skipped[0].reason, "missing_pack")

    def test_build_jobs_reports_empty_generation(self) -> None:
        jobs, skipped = build_jobs(
            packs=[{"pack_id": "pack-1", "heldout_tasks": [{"task_id": "task-1"}]}],
            candidate_rows=[
                {
                    "pack_id": "pack-1",
                    "task_id": "task-1",
                    "mode": "m",
                    "status": "success",
                    "generation": {"text": ""},
                }
            ],
            modes=None,
            pack_ids=None,
        )

        self.assertEqual(jobs, [])
        self.assertEqual(skipped[0].reason, "empty_generation")

    def test_runtime_metadata_tracks_grounding_resume_knobs(self) -> None:
        config = ChatCompletionConfig(
            base_url="https://example.test/v1",
            api_key="key",
            model="judge-a",
            enable_thinking=True,
            thinking_budget=1024,
        )

        metadata = runtime_metadata(
            config=config,
            judge_config_prefix="JUDGE",
            max_tokens=2048,
            parse_max_attempts=3,
            max_evidence_chars=60000,
            max_candidate_chars=40000,
        )

        self.assertEqual(
            metadata,
            {
                "judge_config_prefix": "JUDGE",
                "judge_enable_thinking": True,
                "judge_thinking_budget": 1024,
                "judge_max_tokens": 2048,
                "grounding_parse_max_attempts": 3,
                "max_evidence_chars": 60000,
                "max_candidate_chars": 40000,
            },
        )

    def test_resume_requires_matching_grounding_runtime_metadata(self) -> None:
        metadata = {
            "judge_config_prefix": "JUDGE",
            "judge_enable_thinking": True,
            "judge_thinking_budget": 1024,
            "judge_max_tokens": 2048,
            "grounding_parse_max_attempts": 3,
            "max_evidence_chars": 60000,
            "max_candidate_chars": 40000,
        }

        self.assertTrue(
            row_matches_runtime(
                {"judge_model": "judge-a", **metadata},
                expected_judge_model="judge-a",
                metadata=metadata,
            )
        )
        self.assertFalse(
            row_matches_runtime(
                {"judge_model": "judge-a", **metadata, "max_candidate_chars": 100},
                expected_judge_model="judge-a",
                metadata=metadata,
            )
        )
        self.assertFalse(
            row_matches_runtime(
                {"judge_model": "judge-b", **metadata},
                expected_judge_model="judge-a",
                metadata=metadata,
            )
        )

    def test_resume_filters_stale_grounding_success_rows(self) -> None:
        metadata = {
            "judge_config_prefix": "JUDGE",
            "judge_enable_thinking": True,
            "judge_thinking_budget": 1024,
            "judge_max_tokens": 2048,
            "grounding_parse_max_attempts": 3,
            "max_evidence_chars": 60000,
            "max_candidate_chars": 40000,
        }
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "grounding.jsonl"
            rows = []
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "prompt_only",
                    "status": "success",
                    "judge_model": "judge-a",
                    **metadata,
                },
            )
            append_checkpoint_row(
                out,
                rows,
                {
                    "pack_id": "pack",
                    "task_id": "task",
                    "mode": "auto_skill",
                    "status": "success",
                    "judge_model": "judge-a",
                    **metadata,
                    "grounding_parse_max_attempts": 1,
                },
            )

            resumed = load_compatible_resume_success_rows(
                out,
                {("pack", "task", "prompt_only"), ("pack", "task", "auto_skill")},
                expected_judge_model="judge-a",
                metadata=metadata,
            )

        self.assertEqual([row["mode"] for row in resumed], ["prompt_only"])

    def test_cli_returns_nonzero_for_non_success_without_allow_partial(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            packs = root / "packs.jsonl"
            candidate_eval = root / "eval.jsonl"
            out = root / "grounding.jsonl"
            summary = root / "summary.json"
            env_file = root / ".env"
            packs.write_text(
                (
                    '{"pack_id":"pack","heldout_tasks":['
                    '{"task_id":"task","task_input":"Do it","materials":[]}]}'
                    "\n"
                ),
                encoding="utf-8",
            )
            candidate_eval.write_text(
                (
                    '{"pack_id":"pack","task_id":"task","mode":"prompt_only",'
                    '"status":"success","generation":{"text":"answer"}}\n'
                ),
                encoding="utf-8",
            )
            env_file.write_text(
                "BAILIAN_BASE_URL=https://example.test/v1\n"
                "BAILIAN_API_KEY=key\n"
                "BAILIAN_MODEL=judge-a\n",
                encoding="utf-8",
            )

            argv = [
                "run_grounding_eval.py",
                "--packs",
                str(packs),
                "--eval",
                str(candidate_eval),
                "--out",
                str(out),
                "--summary-out",
                str(summary),
                "--env-file",
                str(env_file),
            ]
            with patch.object(sys, "argv", argv):
                with patch(
                    "scripts.metrics.run_grounding_eval.evaluate_job",
                    return_value=(
                        ("pack", "task", "prompt_only"),
                        {
                            "pack_id": "pack",
                            "task_id": "task",
                            "mode": "prompt_only",
                            "status": "judge_parse_error",
                        },
                    ),
                ):
                    with redirect_stdout(StringIO()):
                        self.assertEqual(main(), 3)


if __name__ == "__main__":
    unittest.main()
