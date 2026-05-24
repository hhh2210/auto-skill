"""Build JSONL artifacts for author-style T2 probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_skill.probes.author_style.artifact_writer import (
    ProbeArtifactConfig,
    manifest_summary,
    write_probe_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blog-clean-posts", type=Path, required=True)
    parser.add_argument("--reddit-clean-posts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--blog-authors", type=int, default=50)
    parser.add_argument("--reddit-authors", type=int, default=50)
    parser.add_argument("--posts-per-author", type=int, default=5)
    parser.add_argument("--blog-min-buckets", type=int, default=3)
    parser.add_argument("--triples-per-author", type=int, default=4)
    parser.add_argument("--triples-per-bucket", type=int, default=80)
    parser.add_argument("--negatives-per-triple", type=int, default=4)
    parser.add_argument("--cross-pairs-per-corpus", type=int, default=500)
    parser.add_argument("--crosscheck-rate", type=float, default=0.10)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ProbeArtifactConfig:
    return ProbeArtifactConfig(**vars(args))


def main() -> int:
    manifest = write_probe_artifacts(config_from_args(parse_args()))
    print(json.dumps(manifest_summary(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
