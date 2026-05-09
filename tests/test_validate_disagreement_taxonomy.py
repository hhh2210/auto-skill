from __future__ import annotations

import unittest

from scripts.metrics.validate_disagreement_taxonomy import validate_taxonomy


def packet(
    mode: str = "skill",
    *,
    pack_id: str = "pack-1",
    task_id: str = "task-1",
    candidate_hash: str = "candidate-hash",
    baseline_hash: str = "baseline-hash",
) -> dict:
    return {
        "pack_id": pack_id,
        "task_id": task_id,
        "mode": mode,
        "candidate_output_stats": {"sha256": candidate_hash},
        "baseline_output_stats": {"sha256": baseline_hash},
    }


def annotation(
    mode: str = "skill",
    *,
    pack_id: str = "pack-1",
    task_id: str = "task-1",
    candidate_hash: str = "candidate-hash",
    baseline_hash: str = "baseline-hash",
    label: str = "candidate_error",
    more_credible_signal: str = "left",
) -> dict:
    return {
        "schema_version": "judge-disagreement-taxonomy/v1",
        "packet_ref": {
            "pack_id": pack_id,
            "task_id": task_id,
            "mode": mode,
            "candidate_sha256": candidate_hash,
            "baseline_sha256": baseline_hash,
        },
        "label": label,
        "more_credible_signal": more_credible_signal,
        "rationale": "Candidate misses the requested deliverable.",
    }


class ValidateDisagreementTaxonomyTests(unittest.TestCase):
    def test_validates_complete_taxonomy(self) -> None:
        report = validate_taxonomy([packet()], [annotation()])

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["label_counts"], {"candidate_error": 1})
        self.assertEqual(report["errors"], [])

    def test_requires_exactly_one_annotation_per_packet(self) -> None:
        report = validate_taxonomy([packet()], [])

        self.assertEqual(report["status"], "error")
        self.assertIn("expected exactly one annotation", report["errors"][0])

    def test_rejects_duplicate_annotations(self) -> None:
        report = validate_taxonomy([packet()], [annotation(), annotation()])

        self.assertEqual(report["status"], "error")
        self.assertIn("expected exactly one annotation, found 2", report["errors"][0])

    def test_rejects_duplicate_packet_rows(self) -> None:
        report = validate_taxonomy([packet(), packet()], [annotation()])

        self.assertEqual(report["status"], "error")
        self.assertTrue(any("duplicate packet row" in error for error in report["errors"]))

    def test_rejects_hash_mismatch(self) -> None:
        report = validate_taxonomy(
            [packet(candidate_hash="expected")],
            [annotation(candidate_hash="actual")],
        )

        self.assertEqual(report["status"], "error")
        self.assertIn("packet hash mismatch", report["errors"][0])

    def test_rejects_unknown_label_and_signal(self) -> None:
        report = validate_taxonomy(
            [packet()],
            [
                annotation(
                    label="looks_bad",
                    more_credible_signal="qwen",
                )
            ],
        )

        self.assertEqual(report["status"], "error")
        self.assertTrue(any("invalid label" in error for error in report["errors"]))
        self.assertTrue(
            any("invalid more_credible_signal" in error for error in report["errors"])
        )

    def test_empty_inputs_fail_closed(self) -> None:
        report = validate_taxonomy([], [])

        self.assertEqual(report["status"], "error")
        self.assertIn("no packets were provided", report["errors"])
        self.assertIn("no annotations were provided", report["errors"])


if __name__ == "__main__":
    unittest.main()
