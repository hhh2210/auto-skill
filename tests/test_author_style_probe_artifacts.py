from __future__ import annotations

import argparse
import unittest

from auto_skill.probes.author_style.artifacts import build_probe
from auto_skill.probes.author_style.negatives import build_negative_pool, select_negatives
from auto_skill.probes.author_style.pairwise_jobs import add_cross_pairwise_for_corpus
from auto_skill.probes.author_style.pairwise_separation import summarize_pairwise_rows


def _post(author: str, index: int, *, corpus: str, words: int, topic: str) -> dict[str, object]:
    return {
        "source": "BlogAuthorship" if corpus == "blog" else "MendeleyRedditCrossTopic",
        "source_id": f"{corpus}-post-{index}-{sum(ord(char) for char in author)}",
        "author_hash": author,
        "text": f"post {index} " + "word " * max(words, 1),
        "word_count": words,
        "private_topic": topic,
        "private_date": f"2004-01-{index + 1:02d}",
        "content_tags": [topic, f"tag-{index % 2}"],
        "style_features": {
            "avg_sentence_words": float(words),
            "question_per_100w": float(index % 2),
            "exclamation_per_100w": 0.0,
        },
    }


def _posts(corpus: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for author in (f"{corpus}-a", f"{corpus}-b"):
        for index, words in enumerate((25, 75, 150, 300, 600)):
            rows.append(_post(author, index, corpus=corpus, words=words, topic=f"topic-{index}"))
    return rows


class AuthorStyleProbeArtifactsTests(unittest.TestCase):
    def test_build_probe_keeps_author_hashes_out_of_public_oracle_artifacts(self) -> None:
        args = argparse.Namespace(
            seed=7,
            blog_authors=2,
            reddit_authors=2,
            posts_per_author=5,
            blog_min_buckets=2,
            triples_per_author=1,
            triples_per_bucket=1,
            negatives_per_triple=1,
            cross_pairs_per_corpus=2,
            crosscheck_rate=0.1,
        )

        packs, _, negatives, pairwise_jobs, manifest, private_manifest = build_probe(
            blog_posts=_posts("blog"),
            reddit_posts=_posts("reddit"),
            args=args,
        )

        self.assertEqual(manifest["styles"], 4)
        self.assertNotIn("author_hash", str(packs))
        self.assertNotIn("author_hash", str(negatives))
        self.assertNotIn("author_hash", str(manifest))
        self.assertNotIn("blog-a", str(packs))
        self.assertNotIn("blog-a", str(negatives))
        self.assertNotIn("blog-a", str(manifest))
        self.assertIn("blog-a", str(private_manifest))
        self.assertEqual(len(pairwise_jobs), 44)

    def test_style_surface_uses_best_post_per_other_author_for_target(self) -> None:
        target = _post("target-author", 0, corpus="blog", words=100, topic="same")
        close = _post("candidate-author", 0, corpus="blog", words=100, topic="same")
        far = _post("candidate-author", 1, corpus="blog", words=900, topic="other")
        distractor = _post("distractor-author", 0, corpus="blog", words=850, topic="other")
        pool = build_negative_pool([target, far, close, distractor], seed=7)

        selected = select_negatives(
            target,
            pool,
            variant="style_surface",
            count=1,
            seed=7,
            salt="style-test",
        )

        self.assertEqual(selected[0]["source_id"], close["source_id"])

    def test_cross_pairwise_caps_at_available_unique_pairs(self) -> None:
        jobs: list[dict[str, object]] = []
        grouped = {
            "left-author": [_post("left-author", 0, corpus="blog", words=100, topic="x")],
            "right-author": [
                _post("right-author", 0, corpus="blog", words=100, topic="x"),
                _post("right-author", 1, corpus="blog", words=120, topic="x"),
            ],
        }

        add_cross_pairwise_for_corpus(
            jobs,
            corpus="blog",
            grouped=grouped,
            seed=7,
            target=100,
        )

        self.assertEqual(len(jobs), 2)

    def test_pairwise_summary_uses_latest_resume_status(self) -> None:
        summary = summarize_pairwise_rows(
            [
                {
                    "job_id": "job-1",
                    "corpus": "blog",
                    "pair_kind": "within_author",
                    "status": "model_error",
                },
                {
                    "job_id": "job-1",
                    "corpus": "blog",
                    "pair_kind": "within_author",
                    "status": "success",
                    "similarity_score": 8,
                },
                {
                    "job_id": "job-2",
                    "corpus": "blog",
                    "pair_kind": "cross_author",
                    "status": "success",
                    "similarity_score": 2,
                },
            ]
        )

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["raw_attempt_rows"], 3)
        self.assertEqual(summary["status_counts"], {"success": 2})
        self.assertEqual(summary["by_corpus"]["blog"]["within_n"], 1)
        self.assertEqual(summary["by_corpus"]["blog"]["cross_n"], 1)


if __name__ == "__main__":
    unittest.main()
