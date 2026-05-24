from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from auto_skill.probes.author_style.embedding_cache import (
    assert_unit_norm,
    embedding_cache_slug,
    embedding_path,
    embedding_sha,
    model_slug,
    truncate_for_embedding,
)
from auto_skill.probes.author_style.topic_nearest import (
    TopicNearestConfig,
    build_manifest,
    build_negative_rows,
    build_summary,
    filter_artifacts,
    rewrite_pack_metadata,
)


class AuthorStyleTopicNearestTests(unittest.TestCase):
    def test_embedding_cache_uses_truncated_text_and_sharded_paths(self) -> None:
        text = "abcdef"
        truncated = truncate_for_embedding(text, 3)
        sha = embedding_sha(truncated)

        self.assertEqual(truncated, "abc")
        self.assertEqual(model_slug("Qwen/Qwen3-Embedding-0.6B"), "Qwen--Qwen3-Embedding-0.6B")
        self.assertEqual(
            embedding_cache_slug(backend="openai", model_id="Qwen/Qwen3-Embedding-0.6B", dim=1024),
            "openai--Qwen--Qwen3-Embedding-0.6B--dim1024",
        )
        self.assertEqual(
            embedding_path(Path("cache"), "model", sha),
            Path("cache") / "model" / sha[:2] / f"{sha}.npy",
        )

    def test_embedding_norm_check_fails_loud_for_dot_product_cosine(self) -> None:
        if find_spec("numpy") is None:
            self.skipTest("numpy is only required for embedding runtime")
        import numpy as np

        with self.assertRaisesRegex(ValueError, "L2-normalized"):
            assert_unit_norm(np.asarray([[2.0, 0.0]], dtype="float32"))

    def test_filter_artifacts_keeps_only_requested_tasks(self) -> None:
        packs = [{"pack_id": "p1"}, {"pack_id": "p2"}]
        private_eval = [
            {
                "pack_id": "p1",
                "heldout_private": [
                    {"task_ref": "p1::heldout::0"},
                    {"task_ref": "p1::heldout::1"},
                ],
            },
            {"pack_id": "p2", "heldout_private": [{"task_ref": "p2::heldout::0"}]},
        ]

        filtered_packs, filtered_private = filter_artifacts(
            packs,
            private_eval,
            {("p1", "p1::heldout::1")},
        )

        self.assertEqual(filtered_packs, [{"pack_id": "p1"}])
        self.assertEqual(filtered_private[0]["heldout_private"], [{"task_ref": "p1::heldout::1"}])

    def test_negative_rows_preserve_task_mapping_and_embedding_score(self) -> None:
        target = _post("target", "author-a", 100)
        negative = _post("negative", "author-b", 120)
        private_eval = [
            {
                "pack_id": "p1",
                "heldout_private": [{"task_ref": "p1::heldout::0"}],
            }
        ]

        rows = build_negative_rows(
            private_eval,
            {("p1", "p1::heldout::0"): target},
            {("p1", "p1::heldout::0"): [(negative, 0.9876543)]},
        )

        self.assertEqual(rows[0]["target_task_ref"], "p1::heldout::0")
        self.assertEqual(rows[0]["negative_type"], "topic_nearest_qwen3_other_author")
        self.assertEqual(rows[0]["match_features"]["embedding_cosine"], 0.987654)
        self.assertTrue(rows[0]["must_not_use_for_induction"])

    def test_manifest_records_base_manifest_hash_and_no_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_probe = root / "probe"
            source_probe.mkdir()
            sample_manifest = source_probe / "sample_manifest.json"
            sample_manifest.write_text(json.dumps({"sample": "fixed"}), encoding="utf-8")

            manifest = build_manifest(
                TopicNearestConfig(
                    source_probe_dir=source_probe,
                    blog_clean_posts=root / "blog.jsonl",
                    reddit_clean_posts=root / "reddit.jsonl",
                    out_dir=root / "out",
                    embedding_backend="openai",
                    embedding_base_url="http://127.0.0.1:18001/v1",
                    max_candidate_chars=3000,
                ),
                packs=[{"pack_id": "p1"}],
                pools={"blog": [_post("b", "a", 10)], "reddit": []},
                hard_negatives=[{"negative_id": "n1"}],
                embedding_summary={
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "embedding_dim": 1024,
                    "truncate_chars": 3000,
                    "package_versions": {"torch": "x"},
                },
            )

        self.assertFalse(manifest["resampled_styles"])
        self.assertEqual(manifest["base_sample_manifest"], str(sample_manifest))
        self.assertIsNotNone(manifest["base_sample_manifest_sha256"])
        self.assertEqual(manifest["pool_source_paths"]["blog"], str(root / "blog.jsonl"))
        self.assertEqual(manifest["embedding_backend"], "openai")
        self.assertEqual(manifest["embedding_base_url"], "http://127.0.0.1:18001/v1")
        self.assertEqual(manifest["embedding_expected_dim"], 1024)
        self.assertEqual(manifest["embedding_workers"], 1)
        self.assertEqual(manifest["embedding"]["embedding_dim"], 1024)

        summary = build_summary(
            TopicNearestConfig(
                source_probe_dir=source_probe,
                blog_clean_posts=root / "blog.jsonl",
                reddit_clean_posts=root / "reddit.jsonl",
                out_dir=root / "out",
                embedding_backend="openai",
                max_candidate_chars=3000,
            ),
            manifest,
            packs=[{"pack_id": "p1"}],
            pools={"blog": [_post("b", "a", 10)], "reddit": []},
            hard_negatives=[{"negative_id": "n1"}],
            embedding_summary={
                "model_slug": "openai--Qwen--dim1024",
                "embedding_dim": 1024,
                "cache_namespace_policy": "backend+model_id+dim",
                "package_versions": {},
            },
        )

        self.assertEqual(summary["schema_version"], "author-style-topic-nearest-summary/v1")
        self.assertEqual(summary["sample_manifest"], str(root / "out" / "sample_manifest.json"))
        self.assertEqual(summary["embedding_dim"], 1024)
        self.assertEqual(summary["cache_namespace_policy"], "backend+model_id+dim")

    def test_pack_metadata_marks_topic_nearest_variant(self) -> None:
        rewritten = rewrite_pack_metadata(
            {"pack_id": "p1", "probe_metadata": {"hard_neg_variant": "style_surface"}}
        )

        self.assertEqual(rewritten["probe_metadata"]["source_hard_neg_variant"], "style_surface")
        self.assertEqual(rewritten["probe_metadata"]["hard_neg_variant"], "topic_nearest_qwen3")


def _post(source_id: str, author_hash: str, word_count: int) -> dict[str, object]:
    return {
        "source": "BlogAuthorship",
        "source_id": source_id,
        "author_hash": author_hash,
        "text": f"text for {source_id}",
        "word_count": word_count,
        "private_topic": "topic",
        "private_date": "2004-01-01",
        "content_tags": ["x"],
        "style_features": {},
    }


if __name__ == "__main__":
    unittest.main()
