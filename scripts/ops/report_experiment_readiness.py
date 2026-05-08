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
from auto_skill.readiness import (  # noqa: E402
    READINESS_PROFILES,
    readiness_profile,
    readiness_report,
)

PROFILE_DEFAULT_PATHS = {
    "smoke": {
        "skills": [Path("runs/skill_mvp.qwen.mvp.jsonl")],
        "writing_eval": [Path("runs/writingbench_official_eval.qwen.mvp.jsonl")],
        "present_surrogate_eval": [Path("runs/presentbench_surrogate_eval.qwen.mvp.jsonl")],
    },
    "mvp": {
        "skills": [Path("runs/skill_mvp.qwen.mvp.jsonl")],
        "writing_eval": [Path("runs/writingbench_official_eval.qwen.mvp.jsonl")],
        "present_surrogate_eval": [Path("runs/presentbench_surrogate_eval.qwen.mvp.jsonl")],
    },
    "full": {
        "skills": [
            Path("runs/skill_mvp.qwen.mvp.jsonl"),
            Path("runs/skill_mvp.qwen.ours_full.writingbench.jsonl"),
        ],
        "writing_eval": [
            Path("runs/writingbench_official_eval.qwen.five_modes.no_thinking_auto_skill.jsonl")
        ],
        "present_surrogate_eval": [Path("runs/presentbench_surrogate_eval.qwen.mvp.jsonl")],
    },
}


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


def load_readiness_jsonls(
    paths: list[Path],
    *,
    label: str,
    missing_messages: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(
            load_readiness_jsonl(
                path,
                label=label,
                missing_messages=missing_messages,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(READINESS_PROFILES),
        default="mvp",
        help=(
            "Readiness contract to apply. smoke permits partial old/small artifacts, "
            "mvp requires current MVP modes, and full requires ours_full/auto_skill "
            "plus PresentBench official scores."
        ),
    )
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument(
        "--skills",
        type=Path,
        action="append",
        help="Skill artifact JSONL. Repeat to merge multiple skill row files.",
    )
    parser.add_argument(
        "--writing-eval",
        type=Path,
        action="append",
        help="WritingBench eval JSONL. Repeat to merge multiple eval row files.",
    )
    parser.add_argument(
        "--present-surrogate-eval",
        type=Path,
        action="append",
        help="PresentBench surrogate eval JSONL. Repeat to merge multiple eval row files.",
    )
    parser.add_argument(
        "--present-official-scores",
        type=Path,
        action="append",
        help=(
            "PresentBench official score JSONL. Repeat to merge multiple score row files. "
            "Defaults to runs/presentbench_official_scores.jsonl."
        ),
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
    profile = readiness_profile(args.profile)
    profile_paths = PROFILE_DEFAULT_PATHS[args.profile]
    skills_paths = args.skills or profile_paths["skills"]
    writing_eval_paths = args.writing_eval or profile_paths["writing_eval"]
    present_surrogate_eval_paths = (
        args.present_surrogate_eval or profile_paths["present_surrogate_eval"]
    )
    present_official_paths = args.present_official_scores or [
        Path("runs/presentbench_official_scores.jsonl")
    ]
    require_presentbench_official = (
        profile.require_presentbench_official and not args.allow_missing_presentbench_official
    )

    artifact_blockers: list[str] = []
    artifact_warnings: list[str] = []
    present_official_missing_messages = artifact_blockers
    if not require_presentbench_official:
        present_official_missing_messages = artifact_warnings
    packs = load_readiness_jsonl(
        args.packs,
        label="packs",
        missing_messages=artifact_blockers,
    )
    skill_rows = load_readiness_jsonls(
        skills_paths,
        label="skills",
        missing_messages=artifact_blockers,
    )
    writing_eval_rows = load_readiness_jsonls(
        writing_eval_paths,
        label="writingbench_eval",
        missing_messages=artifact_blockers,
    )
    present_surrogate_rows = load_readiness_jsonls(
        present_surrogate_eval_paths,
        label="presentbench_surrogate_eval",
        missing_messages=artifact_blockers,
    )
    present_official_rows = load_readiness_jsonls(
        present_official_paths,
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
        require_presentbench_official=require_presentbench_official,
        required_skill_modes=profile.skill_modes,
        required_eval_modes=profile.eval_modes,
        require_complete_coverage=profile.require_complete_coverage,
        profile=profile.name,
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
