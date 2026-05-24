from __future__ import annotations

import unittest

from auto_skill.metrics.author_style_reference import ReferenceCandidate, ReferenceRetrievalJob
from auto_skill.probes.author_style.stylometric import feature_vector, run_baseline, summarize


class AuthorStyleStylometricTests(unittest.TestCase):
    def test_feature_extraction_and_argmax_tie_break_on_two_row_fixture(self) -> None:
        features = feature_vector(
            "I am here! I'm still here. Don't go. Are you there?",
            ("i", "am", "i'm", "don't", "you"),
        )

        self.assertGreater(features["fw:i"], 0)
        self.assertGreater(features["fw:i'm"], 0)
        self.assertGreater(features["fw:don't"], 0)
        self.assertGreater(features["punct:!"], 0)
        self.assertIn("sent:mean", features)
        self.assertIn("sent:p90", features)
        self.assertIn("char: he", features)

        jobs = [
            _job(
                "p1",
                "same text.",
                [
                    ReferenceCandidate("z-ref", "hard_negative", "same text."),
                    ReferenceCandidate("a-ref", "same_author_reference", "same text."),
                    ReferenceCandidate("b-ref", "hard_negative", "different words."),
                ],
                expected="a-ref",
            ),
            _job(
                "p2",
                "question? answer!",
                [
                    ReferenceCandidate("same", "same_author_reference", "question? answer!"),
                    ReferenceCandidate("other", "hard_negative", "plain sentence."),
                ],
                expected="same",
            ),
        ]

        rows = run_baseline(jobs)

        self.assertEqual(rows[0]["selected_candidate_id"], "a-ref")
        self.assertTrue(rows[0]["correct"])
        self.assertTrue(rows[1]["correct"])

        summary = summarize(rows, function_words=("i", "don't"))
        self.assertEqual(summary["feature_config"]["function_words"], ["i", "don't"])


def _job(
    pack_id: str,
    target: str,
    candidates: list[ReferenceCandidate],
    *,
    expected: str,
) -> ReferenceRetrievalJob:
    return ReferenceRetrievalJob(
        pack_id=pack_id,
        task_id=f"{pack_id}::heldout::0",
        source="fixture",
        source_task_id=None,
        probe_metadata={},
        target_text=target,
        expected_candidate_id=expected,
        candidates=candidates,
    )


if __name__ == "__main__":
    unittest.main()
