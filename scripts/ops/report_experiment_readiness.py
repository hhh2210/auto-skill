#!/usr/bin/env python3
"""Report whether generated packs, skills, and eval rows form a usable experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.readiness import readiness_report  # noqa: E402


def load_readiness_jsonl(
    path: Path,
    *,
    label: str,
    missing_messages: list[str],
) -> list[dict]:
    if not path.exists():
        missing_messages.append(f"{label}: file not found: {path}")
        return []
    return load_jsonl(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument("--skills", type=Path, default=Path("runs/skill_mvp.qwen.smoke.jsonl"))
    parser.add_argument(
        "--writing-eval",
        type=Path,
        default=Path("runs/writingbench_official_eval.qwen.smoke.jsonl"),
    )
    parser.add_argument(
        "--present-surrogate-eval",
        type=Path,
        default=Path("runs/presentbench_surrogate_eval.qwen.smoke.jsonl"),
    )
    parser.add_argument(
        "--present-official-scores",
        type=Path,
        default=Path("runs/presentbench_official_scores.jsonl"),
    )
    parser.add_argument("--limit-heldout", type=int)
    parser.add_argument("--out", type=Path, default=Path("runs/experiment_readiness.json"))
    parser.add_argument(
        "--allow-not-ready",
        action="store_true",
        help="Write the report and exit 0 even when blockers remain.",
    )
    parser.add_argument(
        "--allow-missing-presentbench-official",
        action="store_true",
        help=(
            "Downgrade missing PresentBench official score coverage to warnings. "
            "Use only for smoke inspection, not full experiment gating."
        ),
    )
    parser.add_argument(
        "--expect-status",
        choices=["ready", "not_ready"],
        help="Assert the computed status and exit non-zero if it differs.",
    )
    args = parser.parse_args()

    artifact_blockers: list[str] = []
    artifact_warnings: list[str] = []
    present_official_missing_messages = (
        artifact_warnings if args.allow_missing_presentbench_official else artifact_blockers
    )
    packs = load_readiness_jsonl(
        args.packs,
        label="packs",
        missing_messages=artifact_blockers,
    )
    skill_rows = load_readiness_jsonl(
        args.skills,
        label="skills",
        missing_messages=artifact_blockers,
    )
    writing_eval_rows = load_readiness_jsonl(
        args.writing_eval,
        label="writingbench_eval",
        missing_messages=artifact_blockers,
    )
    present_surrogate_rows = load_readiness_jsonl(
        args.present_surrogate_eval,
        label="presentbench_surrogate_eval",
        missing_messages=artifact_blockers,
    )
    present_official_rows = load_readiness_jsonl(
        args.present_official_scores,
        label="presentbench_official_scores",
        missing_messages=present_official_missing_messages,
    )

    report = readiness_report(
        packs=packs,
        skill_rows=skill_rows,
        writing_eval_rows=writing_eval_rows,
        present_surrogate_rows=present_surrogate_rows,
        present_official_rows=present_official_rows,
        limit_heldout=args.limit_heldout,
        require_presentbench_official=not args.allow_missing_presentbench_official,
        artifact_blockers=artifact_blockers,
        artifact_warnings=artifact_warnings,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.expect_status is not None:
        if report["status"] != args.expect_status:
            print(
                f"error: expected status {args.expect_status}, got {report['status']}",
                file=sys.stderr,
            )
            return 4
        return 0
    if report["status"] != "ready" and not args.allow_not_ready:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
