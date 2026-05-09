from __future__ import annotations

import unittest

from auto_skill.grounding import (
    build_grounding_prompt,
    heldout_evidence_text,
    parse_grounding_report,
    summarize_grounding_rows,
    truncate_text,
)
from scripts.metrics.run_grounding_eval import build_jobs


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


if __name__ == "__main__":
    unittest.main()
