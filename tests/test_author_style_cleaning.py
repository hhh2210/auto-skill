from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from auto_skill.author_style_cleaning import (
    CleanPost,
    audit_passes_thresholds,
    build_gpt_audit_prompt,
    build_public_pack,
    choose_authors,
    clean_blog_row,
    clean_longlamp_product_profile_item,
    clean_longlamp_topic_output,
    filter_negatives_by_pack_ids,
    find_hard_negatives,
    find_mendeley_native_impostor_negatives,
    load_mendeley_reddit_posts,
    merge_negative_rows,
    pack_ids_with_negative_coverage,
    pack_prefix_for_source_choice,
    parse_mendeley_filename,
    profile_items,
    run_gpt_negative_rerank,
    source_native_impostor_rows,
)
from auto_skill.mvp import user_examples_from_pack


def sample_text(seed: str, repeat: int = 18) -> str:
    variants = [
        "morning",
        "notebook",
        "window",
        "habit",
        "memory",
        "project",
        "coffee",
        "question",
        "garden",
        "signal",
        "decision",
        "pattern",
        "weather",
        "conversation",
        "rhythm",
        "revision",
        "detail",
        "threshold",
    ]
    sentences = []
    for index in range(repeat):
        marker = variants[index % len(variants)]
        sentences.append(
            f"I keep coming back to {seed} and {marker} because the day feels easier "
            "when I can name the small tradeoffs, compare what changed, and admit "
            f"what still feels unresolved around {marker}."
        )
    return " ".join(sentences)


def clean_post(
    *,
    author_hash: str,
    source_id: str,
    topic: str,
    date: str,
    text_seed: str,
    words: int = 120,
) -> CleanPost:
    text = sample_text(text_seed, max(4, words // 28))
    return CleanPost(
        source="BlogAuthorship",
        source_id=source_id,
        author_hash=author_hash,
        raw_author_id=author_hash.replace("author_", "raw_"),
        private_topic=topic,
        private_date=date,
        text=text,
        word_count=len(text.split()),
        content_tags=(text_seed, "tradeoffs", "changed", "unresolved"),
        style_features={
            "avg_sentence_words": 22.0,
            "question_per_100w": 0.0,
            "exclamation_per_100w": 0.0,
            "ellipsis_per_100w": 0.0,
            "paren_per_100w": 0.0,
            "first_person_per_100w": 4.0,
            "contraction_per_100w": 0.0,
            "hedge_marker": 0.0,
        },
    )


class AuthorStyleCleaningTests(unittest.TestCase):
    def test_clean_blog_row_keeps_author_and_private_fields_out_of_public_id(self) -> None:
        row = {
            "id": "12345",
            "date": "02,August,2004",
            "topic": "Technology",
            "text": sample_text("laptops", repeat=3),
        }

        post = clean_blog_row(row, min_words=80, max_words=900)

        self.assertIsNotNone(post)
        assert post is not None
        self.assertEqual(post.private_date, "2004-08-02")
        self.assertEqual(post.private_topic, "Technology")
        self.assertTrue(post.author_hash.startswith("author_"))
        self.assertNotIn("12345", post.author_hash)

    def test_public_pack_is_runner_readable_and_hides_heldout_output(self) -> None:
        train = [
            clean_post(
                author_hash="author_a",
                source_id=f"train-{index}",
                topic="Tech",
                date=f"2004-08-0{index + 1}",
                text_seed="laptops",
            )
            for index in range(3)
        ]
        heldout = [
            clean_post(
                author_hash="author_a",
                source_id="heldout-0",
                topic="Tech",
                date="2004-09-01",
                text_seed="laptops",
            )
        ]

        pack = build_public_pack("blog_author_style_0001", train, heldout)
        examples = user_examples_from_pack(pack)

        self.assertEqual(len(examples), 3)
        self.assertEqual(pack["train_examples"][0]["desired_output"]["status"], "generated")
        self.assertIsNone(pack["heldout_tasks"][0]["desired_output"])
        rendered_ids = "\n".join(
            [pack["pack_id"]]
            + [row["example_id"] for row in pack["train_examples"]]
            + [row["task_id"] for row in pack["heldout_tasks"]]
        )
        self.assertNotIn("author_a", rendered_ids)

    def test_hard_negatives_are_different_author_and_matched(self) -> None:
        target_heldout = clean_post(
            author_hash="author_a",
            source_id="heldout-0",
            topic="Tech",
            date="2004-09-01",
            text_seed="laptops",
        )
        selected = [("author_a", [], [target_heldout])]
        pool = [
            target_heldout,
            clean_post(
                author_hash="author_b",
                source_id="neg-0",
                topic="Tech",
                date="2004-09-03",
                text_seed="laptops",
            ),
        ]

        negatives = find_hard_negatives(
            selected,
            pool,
            negatives_per_heldout=1,
            pack_ids_by_author={"author_a": "blog_author_style_0001"},
        )

        self.assertEqual(len(negatives), 1)
        self.assertEqual(negatives[0]["private_label"], "different_author")
        self.assertEqual(negatives[0]["negative_author_hash"], "author_b")
        self.assertTrue(negatives[0]["match_features"]["same_topic"])
        self.assertIn("style_similarity", negatives[0]["match_features"])
        self.assertEqual(negatives[0]["target_task_ref"], "blog_author_style_0001::heldout::0")

    def test_hard_negative_author_cap_preserves_diversity(self) -> None:
        target_heldout = clean_post(
            author_hash="author_a",
            source_id="heldout-0",
            topic="Tech",
            date="2004-09-01",
            text_seed="laptops",
        )
        selected = [("author_a", [], [target_heldout])]
        pool = [target_heldout]
        pool.extend(
            clean_post(
                author_hash="author_b",
                source_id=f"neg-b-{index}",
                topic="Tech",
                date=f"2004-09-0{index + 2}",
                text_seed="laptops",
            )
            for index in range(3)
        )
        pool.append(
            clean_post(
                author_hash="author_c",
                source_id="neg-c-0",
                topic="Tech",
                date="2004-09-06",
                text_seed="laptops",
            )
        )

        negatives = find_hard_negatives(
            selected,
            pool,
            negatives_per_heldout=3,
            negative_author_cap=1,
            pack_ids_by_author={"author_a": "blog_author_style_0001"},
        )

        negative_authors = [row["negative_author_hash"] for row in negatives]
        self.assertEqual(len(negative_authors), 2)
        self.assertEqual(len(set(negative_authors)), 2)

    def test_gpt_audit_prompt_uses_private_heldout_without_public_pack_leak(self) -> None:
        train = [
            clean_post(
                author_hash="author_a",
                source_id=f"train-{index}",
                topic="Tech",
                date=f"2004-08-0{index + 1}",
                text_seed="laptops",
            )
            for index in range(3)
        ]
        heldout = [
            clean_post(
                author_hash="author_a",
                source_id="heldout-0",
                topic="Tech",
                date="2004-09-01",
                text_seed="laptops",
            )
        ]
        pack = build_public_pack("blog_author_style_0001", train, heldout)
        private_eval = {
            "pack_id": pack["pack_id"],
            "heldout_private": [
                {
                    "task_ref": pack["heldout_tasks"][0]["task_id"],
                    "reference_output_private": "This private heldout sentence verifies style.",
                }
            ],
        }

        prompt = build_gpt_audit_prompt(pack, [], private_eval)

        self.assertIn("This private heldout sentence verifies style.", prompt)
        self.assertIsNone(pack["heldout_tasks"][0]["desired_output"])

    def test_filter_negatives_can_require_gpt_selected_rows(self) -> None:
        rows = [
            {
                "target_task_ref": "blog_author_style_0001::heldout::0",
                "gpt_rerank": {"status": "selected"},
            },
            {
                "target_task_ref": "blog_author_style_0001::heldout::1",
                "gpt_rerank": {"status": "fallback_prefilter"},
            },
            {
                "target_task_ref": "blog_author_style_0002::heldout::0",
                "gpt_rerank": {"status": "selected"},
            },
        ]

        filtered = filter_negatives_by_pack_ids(
            rows,
            {"blog_author_style_0001"},
            require_gpt_selected=True,
        )

        self.assertEqual(filtered, [rows[0]])

    def test_gpt_negative_rerank_transport_error_falls_back(self) -> None:
        pack = {
            "pack_id": "blog_author_style_0001",
            "heldout_tasks": [{"task_id": "blog_author_style_0001::heldout::0"}],
            "train_examples": [],
        }
        private = {
            "pack_id": "blog_author_style_0001",
            "heldout_private": [
                {
                    "task_ref": "blog_author_style_0001::heldout::0",
                    "reference_output_private": "Reference text.",
                }
            ],
        }
        candidate = {
            "negative_id": "prefilter-1",
            "target_task_ref": "blog_author_style_0001::heldout::0",
            "public_negative_text": "Candidate text.",
        }
        args = Namespace(
            gpt_rerank_candidates=1,
            negatives_per_heldout=1,
            gpt_rerank_candidate_chars=100,
            gpt_max_attempts=2,
            gpt_model="test-model",
            auth=None,
            service_tier=None,
            gpt_timeout_seconds=1,
            gpt_rerank_num_threads=1,
        )

        with patch(
            "auto_skill.cleaning.author_style.gpt.codex_response_text",
            side_effect=RuntimeError("boom"),
        ):
            rows = run_gpt_negative_rerank([pack], [private], [candidate], args)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prefilter_negative_id"], "prefilter-1")
        self.assertEqual(rows[0]["gpt_rerank"]["status"], "error_fallback_prefilter")
        self.assertEqual(rows[0]["gpt_rerank"]["error_type"], "RuntimeError")
        self.assertEqual(rows[0]["gpt_rerank"]["attempts"], 2)

    def test_merge_negative_rows_preserves_source_native_marker(self) -> None:
        ordinary = {
            "target_task_ref": "mendeley_author_style_0001::heldout::0",
            "negative_author_hash": "author_native",
            "negative_type": "original_av_impostor_cross_topic",
            "public_negative_text": "Same native impostor text.",
            "gpt_rerank": {"status": "selected"},
        }
        native = source_native_impostor_rows([ordinary])[0]

        merged = merge_negative_rows([ordinary, native])

        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0]["gpt_rerank"]["status"],
            "source_native_impostor",
        )

    def test_audit_threshold_policy_none_accepts_successful_audit_without_scores(self) -> None:
        row = {
            "status": "success",
            "pack_id": "pack_a",
            "audit": {
                "usable": False,
                "should_use_for_smoke": False,
                "style_extractability_1_to_5": 1,
            },
        }

        self.assertTrue(audit_passes_thresholds(row, Namespace(audit_thresholds="none")))

    def test_audit_threshold_policy_smoke_uses_quality_cutoffs(self) -> None:
        args = Namespace(
            audit_thresholds="smoke",
            min_style_extractability=4.0,
            min_negative_strength=3.0,
            max_topic_leakage=4.0,
            max_model_familiarity=3.0,
        )
        weak = {
            "status": "success",
            "pack_id": "pack_a",
            "audit": {
                "usable": True,
                "should_use_for_smoke": True,
                "style_extractability_1_to_5": 2,
                "negative_strength_1_to_5": 4,
                "topic_leakage_risk_1_to_5": 3,
                "model_familiarity_risk_1_to_5": 2,
            },
        }
        strong = {
            **weak,
            "audit": {
                **weak["audit"],
                "style_extractability_1_to_5": 4,
            },
        }

        self.assertFalse(audit_passes_thresholds(weak, args))
        self.assertTrue(audit_passes_thresholds(strong, args))

    def test_negative_coverage_gate_drops_accepted_pack_with_too_few_negatives(self) -> None:
        private_rows = [
            {
                "pack_id": "pack_a",
                "heldout_private": [{"task_ref": "pack_a::heldout::0"}],
            },
            {
                "pack_id": "pack_b",
                "heldout_private": [{"task_ref": "pack_b::heldout::0"}],
            },
        ]
        negative_rows = [
            {"target_task_ref": "pack_a::heldout::0"},
            {"target_task_ref": "pack_a::heldout::0"},
            {"target_task_ref": "pack_b::heldout::0"},
        ]

        covered = pack_ids_with_negative_coverage(
            private_rows,
            negative_rows,
            {"pack_a", "pack_b"},
            min_negatives_per_heldout=2,
        )

        self.assertEqual(covered, {"pack_a"})

    def test_longlamp_topic_output_sanitizes_public_task_input(self) -> None:
        post = clean_longlamp_topic_output(
            {
                "author": "--Caius--",
                "input": "Generate the content for a reddit post about jungle queue anxiety.",
                "output": sample_text("jungle", repeat=3),
                "profile": [],
            },
            row_index=7,
            min_words=40,
            max_words=900,
        )

        self.assertIsNotNone(post)
        assert post is not None
        self.assertEqual(post.source, "LongLaMPTopicWritingTemporal")
        self.assertNotIn("--Caius--", post.task_input or "")
        self.assertNotIn("reddit", (post.task_input or "").lower())
        self.assertTrue(post.private_metadata["date_is_synthetic"])
        pack = build_public_pack("longlamp_topic_author_style_0001", [post], [post])
        self.assertEqual(pack["domain"]["kind"], "personal_topic_writing")
        self.assertNotIn("source_config", str(pack))
        self.assertNotIn("LongLaMP", str(pack))
        self.assertEqual(pack["source"], "personal_topic_writing_history")
        self.assertEqual(pack["train_examples"][0]["source"], "personal_topic_writing_history")
        self.assertEqual(pack_prefix_for_source_choice("longlamp-topic"), "author_style_topic_post")
        self.assertNotIn("longlamp", pack_prefix_for_source_choice("longlamp-topic").lower())

    def test_longlamp_product_profile_accepts_json_string_profiles(self) -> None:
        profile = json.dumps(
            [
                {
                    "reviewText": sample_text("toaster", repeat=3),
                    "description": "A toaster oven",
                    "overall": "4.0",
                    "summary": "Good oven",
                }
            ]
        )
        items = profile_items(profile)

        self.assertEqual(len(items), 1)
        post = clean_longlamp_product_profile_item(
            items[0],
            raw_author_id="A1REVIEWER",
            row_index=11,
            item_index=0,
            min_words=40,
            max_words=900,
        )

        self.assertIsNotNone(post)
        assert post is not None
        self.assertEqual(post.source, "LongLaMPProductReviewTemporal")
        self.assertIn("rating", post.task_input or "")
        self.assertNotIn("reviewerId", post.task_input or "")
        self.assertEqual(post.private_metadata["source_config"], "product_review_temporal")

    def test_mendeley_reddit_loader_reindexes_unknown_by_true_doc_author(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reddit_av.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("truth.txt", "authorA N\nauthorB N\n")
                for topic in ["alpha", "beta", "gamma", "delta"]:
                    archive.writestr(
                        f"authorA/known - authorA - [{topic}][2015].txt",
                        sample_text(f"a_{topic}", repeat=3),
                    )
                    archive.writestr(
                        f"authorB/known - authorB - [{topic}][2015].txt",
                        sample_text(f"b_{topic}", repeat=3),
                    )
                archive.writestr(
                    "authorA/unknown - authorB - [epsilon][2016].txt",
                    sample_text("b_heldout", repeat=3),
                )
                archive.writestr(
                    "authorB/unknown - authorA - [zeta][2016].txt",
                    sample_text("a_heldout", repeat=3),
                )

            posts = load_mendeley_reddit_posts(
                Namespace(mendeley_path=path, min_words=40, max_words=2_500)
            )
            selected = choose_authors(posts, authors=2, train_posts=4, heldout_posts=1)

        self.assertEqual(len(posts), 10)
        self.assertEqual(len(selected), 2)
        for author_hash, train, heldout in selected:
            self.assertEqual(len(train), 4)
            self.assertEqual(len(heldout), 1)
            self.assertTrue(all(post.author_hash == author_hash for post in train + heldout))
            self.assertEqual(
                {post.private_metadata["doc_role_private"] for post in train},
                {"known"},
            )
            self.assertEqual(heldout[0].private_metadata["doc_role_private"], "unknown")

        pack = build_public_pack(
            "author_style_reddit_cross_topic_0001",
            selected[0][1],
            selected[0][2],
        )
        self.assertEqual(pack["source"], "cross_topic_online_comment_history")
        self.assertEqual(pack["domain"]["split_kind"], "known_unknown_cross_topic")
        self.assertIn("authentic voice", pack["train_examples"][0]["task_input"])
        self.assertIn("authentic voice", pack["heldout_tasks"][0]["task_input"])
        self.assertNotIn("authorA", str(pack))
        self.assertNotIn("authorB", str(pack))
        self.assertEqual(
            pack_prefix_for_source_choice("mendeley-reddit"),
            "author_style_reddit_cross_topic",
        )

    def test_mendeley_original_negative_problem_yields_native_impostor(self) -> None:
        target = clean_post(
            author_hash="author_a",
            source_id="heldout-a",
            topic="zeta",
            date="2027-05-18",
            text_seed="a_heldout",
        )
        known = clean_post(
            author_hash="author_a",
            source_id="known-a",
            topic="alpha",
            date="2000-01-01",
            text_seed="a_known",
        )
        known = CleanPost(
            **{
                **known.__dict__,
                "source": "MendeleyRedditCrossTopic",
                "private_metadata": {
                    "doc_role_private": "known",
                    "mendeley_problem_id_private": "authorA",
                },
            }
        )
        impostor = clean_post(
            author_hash="author_b",
            source_id="unknown-b",
            topic="epsilon",
            date="2027-05-18",
            text_seed="b_heldout",
        )
        impostor = CleanPost(
            **{
                **impostor.__dict__,
                "source": "MendeleyRedditCrossTopic",
                "private_metadata": {
                    "doc_role_private": "unknown",
                    "truth_label_private": "N",
                    "mendeley_problem_id_private": "authorA",
                },
            }
        )

        rows = find_mendeley_native_impostor_negatives(
            [("author_a", [known], [target])],
            [known, target, impostor],
            pack_ids_by_author={"author_a": "author_style_reddit_cross_topic_0001"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["negative_type"], "original_av_impostor_cross_topic")
        self.assertEqual(rows[0]["negative_author_hash"], "author_b")
        self.assertEqual(
            rows[0]["target_task_ref"],
            "author_style_reddit_cross_topic_0001::heldout::0",
        )

    def test_parse_mendeley_filename_extracts_role_author_topic_year(self) -> None:
        parsed = parse_mendeley_filename(
            "SomeUser/unknown - OtherUser - [AskHistorians][2015].txt"
        )

        self.assertEqual(
            parsed,
            {
                "problem_id": "SomeUser",
                "role": "unknown",
                "doc_author": "OtherUser",
                "subreddit": "AskHistorians",
                "year": "2015",
            },
        )


if __name__ == "__main__":
    unittest.main()
