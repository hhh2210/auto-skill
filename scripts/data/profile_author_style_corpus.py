#!/usr/bin/env python3
"""Profile raw blog/reddit author-style corpora without dropping rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_skill.cleaning.author_style.profile import (  # noqa: E402
    iter_blog_records,
    iter_mendeley_records,
    profile_records,
    write_profile_json,
    write_profile_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile raw author-style source corpora before choosing filters.",
    )
    parser.add_argument("--source", choices=("blog", "mendeley-reddit"), required=True)
    parser.add_argument("--max-rows", type=int, default=200_000, help="0 means no row limit.")
    parser.add_argument("--hf-dataset", default="paoramen/blog-authorship-corpus")
    parser.add_argument(
        "--mendeley-path",
        type=Path,
        default=Path("data/mendeley_reddit_cross_topic/Reddit_Cross-Topic-AV-Corpus_1000_users.zip"),
    )
    parser.add_argument("--train-posts", type=int, default=1)
    parser.add_argument("--heldout-posts", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    if args.source == "blog":
        records = iter_blog_records(args)
    else:
        records = iter_mendeley_records(args.mendeley_path, max_rows=max_rows)

    profile = profile_records(
        records,
        source=args.source,
        train_posts=args.train_posts,
        heldout_posts=args.heldout_posts,
    )

    json_out = args.json_out or Path("runs/author_style") / f"{args.source}_corpus_profile.json"
    md_out = args.md_out or Path("runs/author_style") / f"{args.source}_corpus_profile.md"
    write_profile_json(json_out, profile)
    write_profile_markdown(md_out, profile)
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    print(
        f"rows={profile['profiled_rows']} authors={profile['author_count']} "
        f"word_p50={profile['word_count']['p50']} word_p90={profile['word_count']['p90']} "
        f"exact_dup_rate={profile['exact_text_duplicate_rate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
