from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from auto_skill.author_style_audit import audit_author_style_run


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_run(root: Path, *, public_source: str = "personal_topic_writing_history") -> None:
    pack = {
        "schema_version": "author-style-pack/v1",
        "pack_id": "author_style_topic_post_0001",
        "split_id": "author_style_topic_post_0001::temporal",
        "source": public_source,
        "learning_problem": "personal_author_style_induction",
        "domain": {"kind": "personal_topic_writing", "language": "en"},
        "input_boundary": {
            "auto_skill_module_can_use": [
                "train_examples.task_input",
                "train_examples.desired_output.text",
            ],
            "must_not_use_for_induction": ["raw author ids", "heldout outputs"],
        },
        "train_examples": [
            {
                "schema_version": "author-style-pack/v1",
                "example_id": "author_style_topic_post_0001::train::0",
                "source": public_source,
                "source_task_id": "post_1",
                "domain": {"kind": "personal_topic_writing", "language": "en"},
                "task_input": "Write a short online post about a routine argument.",
                "metadata": {"public_content_tags": ["routine", "argument"], "word_count": 98},
                "desired_output": {
                    "status": "generated",
                    "text": (
                        "I mean, sure, this whole routine feels absurd, but that is exactly "
                        "why I keep circling back to it in these short, clipped paragraphs."
                    ),
                    "source": "original_user_history",
                    "provenance": "source_provided",
                },
            }
        ],
        "heldout_tasks": [
            {
                "schema_version": "author-style-pack/v1",
                "task_id": "author_style_topic_post_0001::heldout::0",
                "source": public_source,
                "source_task_id": "post_2",
                "domain": {"kind": "personal_topic_writing", "language": "en"},
                "task_input": "Write a short online post about an annoying habit.",
                "metadata": {"public_content_tags": ["habit", "argument"], "word_count": 92},
                "desired_output": None,
            }
        ],
    }
    private_eval = {
        "schema_version": "author-style-private-eval/v1",
        "pack_id": "author_style_topic_post_0001",
        "raw_author_id_private": "author-a",
        "heldout_private": [
            {
                "task_ref": "author_style_topic_post_0001::heldout::0",
                "reference_output_private": (
                    "Sure, the habit is tiny, but the rhythm of complaining about it "
                    "has become the whole point."
                ),
            }
        ],
    }
    audit = {
        "schema_version": "author-style-gpt-audit/v1",
        "status": "success",
        "pack_id": "author_style_topic_post_0001",
        "model": "gpt-5.5",
        "audit": {
            "usable": True,
            "should_use_for_smoke": True,
            "style_extractability_1_to_5": 4,
            "negative_strength_1_to_5": 4,
            "topic_leakage_risk_1_to_5": 3,
            "model_familiarity_risk_1_to_5": 2,
            "style_cluster_tags": ["clipped_argumentative", "concessive_sure"],
        },
    }
    negative = {
        "schema_version": "author-style-hard-negative/v1",
        "negative_id": "author_style_topic_post_0001::heldout::0::negative::1",
        "target_task_ref": "author_style_topic_post_0001::heldout::0",
        "target_author_hash": "author_a",
        "negative_author_hash": "author_b",
        "negative_type": "topic_time_length_style_matched_impostor",
        "public_negative_text": (
            "Sure, this is annoying, but the punchline is that everyone keeps doing it anyway."
        ),
        "public_negative_content_tags": ["annoying", "habit"],
        "match_features": {"style_similarity": 0.91, "score": 4.2},
        "private_label": "different_author",
        "gpt_rerank": {
            "status": "selected",
            "style_confusability_1_to_5": 5,
            "topic_shortcut_risk_1_to_5": 2,
        },
    }

    write_jsonl(root / "accepted_author_style_packs.jsonl", [pack])
    write_jsonl(root / "accepted_author_style_private_eval.jsonl", [private_eval])
    write_jsonl(root / "gpt55_author_audits.jsonl", [audit])
    write_jsonl(root / "accepted_hard_negatives.jsonl", [negative])
    (root / "author_style_smoke_summary.json").write_text(
        json.dumps({"source": "longlamp-topic", "clean_posts": 100, "packs": 1}),
        encoding="utf-8",
    )


class AuthorStyleArtifactAuditTests(unittest.TestCase):
    def test_audit_accepts_clean_selected_only_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_run(root)

            report = audit_author_style_run(run_dir=root, min_negatives_per_heldout=1)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["hard_negatives"]["gpt_rerank_status_counts"], {"selected": 1})
            self.assertEqual(
                report["gpt_audits"]["accepted_style_cluster_tag_counts"],
                {"author_style_topic_post_0001": 2},
            )

    def test_audit_allows_deterministic_negatives_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_run(root)
            negative_path = root / "accepted_hard_negatives.jsonl"
            row = json.loads(negative_path.read_text(encoding="utf-8").splitlines()[0])
            row.pop("gpt_rerank")
            write_jsonl(negative_path, [row])

            report = audit_author_style_run(run_dir=root, min_negatives_per_heldout=1)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(
                report["hard_negatives"]["gpt_rerank_status_counts"],
                {"not_reranked": 1},
            )

    def test_audit_rejects_public_source_label_leakage_and_strict_fallback_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_run(root, public_source="LongLaMPTopicWritingTemporal")
            negative_path = root / "accepted_hard_negatives.jsonl"
            row = json.loads(negative_path.read_text(encoding="utf-8").splitlines()[0])
            row["gpt_rerank"]["status"] = "fallback_prefilter"
            write_jsonl(negative_path, [row])

            report = audit_author_style_run(
                run_dir=root,
                min_negatives_per_heldout=1,
                require_gpt_selected=True,
            )

            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("forbidden terms" in issue for issue in report["issues"]),
                report["issues"],
            )
            self.assertTrue(
                any("not gpt-selected" in issue for issue in report["issues"]),
                report["issues"],
            )

    def test_audit_can_allow_source_native_impostor_without_allowing_all_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_run(root)
            negative_path = root / "accepted_hard_negatives.jsonl"
            selected = json.loads(negative_path.read_text(encoding="utf-8").splitlines()[0])
            native = {
                **selected,
                "negative_id": "author_style_topic_post_0001::heldout::0::negative::native",
                "negative_type": "original_av_impostor_cross_topic",
                "gpt_rerank": {"status": "source_native_impostor"},
            }
            write_jsonl(negative_path, [selected, native])

            report = audit_author_style_run(
                run_dir=root,
                min_negatives_per_heldout=2,
                require_gpt_selected=True,
                allow_source_native_negatives=True,
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(
                report["hard_negatives"]["gpt_rerank_status_counts"],
                {"selected": 1, "source_native_impostor": 1},
            )
            self.assertEqual(
                report["hard_negatives"]["negative_type_counts"]["original_av_impostor_cross_topic"],
                1,
            )

            native["negative_type"] = "topic_time_length_style_matched_impostor"
            write_jsonl(negative_path, [selected, native])
            rejected = audit_author_style_run(
                run_dir=root,
                min_negatives_per_heldout=2,
                require_gpt_selected=True,
                allow_source_native_negatives=True,
            )

            self.assertEqual(rejected["status"], "failed")
            self.assertTrue(
                any("not gpt-selected" in issue for issue in rejected["issues"]),
                rejected["issues"],
            )


if __name__ == "__main__":
    unittest.main()
