from __future__ import annotations

import unittest

from auto_skill.cleaning.author_style.profile import (
    ProfileRecord,
    percentile,
    profile_records,
    record_from_blog_row,
)


class AuthorStyleProfileTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([10, 20, 30], 50), 20)
        self.assertEqual(percentile([10, 20, 30, 40], 50), 25)
        self.assertEqual(percentile([10, 20, 30, 40], 90), 37)

    def test_record_from_blog_row_keeps_raw_empty_signals(self) -> None:
        record = record_from_blog_row(
            {
                "id": " 123 ",
                "date": "02,August,2004",
                "topic": "Technology",
                "text": "I have a small post with the same words.",
            }
        )

        self.assertEqual(record.raw_author_id, "123")
        self.assertEqual(record.date, "2004-08-02")
        self.assertEqual(record.topic, "Technology")

    def test_profile_counts_duplicates_and_author_feasibility(self) -> None:
        records = [
            ProfileRecord(
                source="blog",
                raw_author_id="a",
                text="The same post has enough words and with a simple ending.",
                topic="x",
            ),
            ProfileRecord(
                source="blog",
                raw_author_id="a",
                text="The same post has enough words and with a simple ending.",
                topic="x",
            ),
            ProfileRecord(
                source="blog",
                raw_author_id="b",
                text="A different post can be short.",
                topic="y",
            ),
            ProfileRecord(source="blog", raw_author_id="", text="", topic=""),
        ]

        profile = profile_records(records, source="blog", train_posts=1, heldout_posts=1)

        self.assertEqual(profile["profiled_rows"], 4)
        self.assertEqual(profile["empty_author_rows"], 1)
        self.assertEqual(profile["empty_text_rows"], 1)
        self.assertEqual(profile["exact_text_duplicate_rows"], 1)
        self.assertEqual(profile["author_count"], 2)
        self.assertEqual(profile["authors_with_at_least_2_posts"], 1)
        self.assertEqual(profile["triple_feasibility"]["feasible_author_count"], 1)
        self.assertEqual(profile["word_count"]["p50"], 7.5)


if __name__ == "__main__":
    unittest.main()
