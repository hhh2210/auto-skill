from __future__ import annotations

import unittest

from auto_skill.author_style_negative_transfer import build_negative_transfer_packs


def pack(pack_id: str, tags: list[str], words: int, heldout_tags: list[str]) -> dict:
    return {
        "schema_version": "author-style-pack/v1",
        "pack_id": pack_id,
        "source": "cross_topic_online_comment_history",
        "train_examples": [
            {
                "example_id": f"{pack_id}::train::0",
                "task_input": "write",
                "metadata": {"public_content_tags": tags, "word_count": words},
                "desired_output": {"text": f"{pack_id} text"},
            }
        ],
        "heldout_tasks": [
            {
                "task_id": f"{pack_id}::heldout::0",
                "task_input": "write heldout",
                "metadata": {"public_content_tags": heldout_tags, "word_count": words},
            }
        ],
    }


class AuthorStyleNegativeTransferTests(unittest.TestCase):
    def test_build_negative_transfer_packs_uses_public_wrong_author_examples(self) -> None:
        packs = [
            pack("target", ["alpha", "beta"], 100, ["topic_a"]),
            pack("near_length_disjoint", ["gamma", "delta"], 110, ["topic_b"]),
            pack("topic_overlap", ["topic_a", "delta"], 100, ["topic_c"]),
        ]

        output_packs, manifest = build_negative_transfer_packs(packs, seed=7)
        target_output = next(row for row in output_packs if row["pack_id"] == "target")

        self.assertEqual(target_output["heldout_tasks"][0]["task_id"], "target::heldout::0")
        self.assertEqual(
            target_output["negative_transfer"]["impostor_pack_id"],
            "near_length_disjoint",
        )
        self.assertEqual(
            target_output["train_examples"][0]["desired_output"]["text"],
            "near_length_disjoint text",
        )
        self.assertTrue(target_output["negative_transfer"]["must_not_use_for_induction"])
        self.assertFalse(manifest["artifact_boundary"]["uses_private_eval"])
        self.assertFalse(manifest["artifact_boundary"]["uses_hard_negative_text"])
        self.assertEqual(
            manifest["selection_algorithm"],
            "greedy_sorted_by_source_tag_length_reuse_tiebreak",
        )
        self.assertIn("p90_length_log_ratio", manifest)
        self.assertIn("reuse_histogram", manifest)

    def test_build_negative_transfer_packs_can_cap_impostor_reuse(self) -> None:
        packs = [
            pack("p1", ["alpha"], 100, ["topic_a"]),
            pack("p2", ["beta"], 110, ["topic_b"]),
            pack("p3", ["gamma"], 120, ["topic_c"]),
            pack("p4", ["delta"], 130, ["topic_d"]),
        ]

        output_packs, manifest = build_negative_transfer_packs(
            packs,
            seed=7,
            max_impostor_reuse_count=1,
        )
        impostor_ids = [
            row["negative_transfer"]["impostor_pack_id"] for row in output_packs
        ]

        self.assertEqual(len(set(impostor_ids)), len(output_packs))
        self.assertEqual(manifest["unique_impostor_pack_count"], len(output_packs))
        self.assertEqual(manifest["max_impostor_reuse_count"], 1)
        self.assertEqual(manifest["max_impostor_reuse_count_allowed"], 1)
        self.assertEqual(manifest["reuse_histogram"], {"1": 4})

    def test_build_negative_transfer_packs_rejects_non_positive_reuse_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_negative_transfer_packs(
                [pack("p1", ["alpha"], 100, ["topic_a"])],
                max_impostor_reuse_count=0,
            )


if __name__ == "__main__":
    unittest.main()
