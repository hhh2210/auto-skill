#!/usr/bin/env python3
"""Build target-nearest Qwen3 embedding hard negatives for T2 probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.probes.author_style.topic_nearest import (  # noqa: E402
    TopicNearestConfig,
    write_topic_nearest_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-probe-dir", type=Path, required=True)
    parser.add_argument("--blog-clean-posts", type=Path, required=True)
    parser.add_argument("--reddit-clean-posts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--embedding-backend",
        choices=("sentence-transformers", "openai"),
        default="sentence-transformers",
    )
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-api-key", default="EMPTY")
    parser.add_argument("--embedding-expected-dim", type=int, default=1024)
    parser.add_argument("--embedding-max-retries", type=int, default=3)
    parser.add_argument("--embedding-workers", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-candidate-chars", type=int, default=3000)
    parser.add_argument("--negatives-per-target", type=int, default=4)
    parser.add_argument("--cache-root", type=Path, default=Path("runs/cache/embeddings"))
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--limit-corpus-rows", type=int)
    parser.add_argument("--max-scan", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    manifest = write_topic_nearest_artifacts(TopicNearestConfig(**vars(parse_args())))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
