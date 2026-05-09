#!/usr/bin/env python3
"""Validate judge-disagreement taxonomy annotations against packet JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl  # noqa: E402

ALLOWED_LABELS = {
    "judge_error",
    "candidate_error",
    "baseline_error",
    "small_delta_noise",
    "ambiguous",
}
ALLOWED_CREDIBLE_SIGNALS = {
    "left",
    "right",
    "both",
    "neither",
    "unresolved",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def packet_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("pack_id") or ""),
        str(row.get("task_id") or ""),
        str(row.get("mode") or ""),
    )


def annotation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    ref = row.get("packet_ref")
    if not isinstance(ref, dict):
        return ("", "", "")
    return (
        str(ref.get("pack_id") or ""),
        str(ref.get("task_id") or ""),
        str(ref.get("mode") or ""),
    )


def expected_hashes(packet: dict[str, Any]) -> dict[str, Any]:
    candidate_stats = packet.get("candidate_output_stats")
    baseline_stats = packet.get("baseline_output_stats")
    return {
        "candidate_sha256": (
            candidate_stats.get("sha256") if isinstance(candidate_stats, dict) else None
        ),
        "baseline_sha256": (
            baseline_stats.get("sha256") if isinstance(baseline_stats, dict) else None
        ),
    }


def annotation_hashes(annotation: dict[str, Any]) -> dict[str, Any]:
    ref = annotation.get("packet_ref")
    if not isinstance(ref, dict):
        return {"candidate_sha256": None, "baseline_sha256": None}
    return {
        "candidate_sha256": ref.get("candidate_sha256"),
        "baseline_sha256": ref.get("baseline_sha256"),
    }


def validate_taxonomy(
    packets: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    packet_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    annotation_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for packet in packets:
        key = packet_key(packet)
        if not all(key):
            errors.append("packet missing pack_id/task_id/mode")
            continue
        packet_hashes = expected_hashes(packet)
        for name, value in packet_hashes.items():
            if not valid_sha256(value):
                errors.append(f"{key}: packet {name} must be non-empty 64-hex sha256")
        if key in packet_index:
            errors.append(f"{key}: duplicate packet row")
            continue
        packet_index[key] = packet

    for annotation in annotations:
        key = annotation_key(annotation)
        if not all(key):
            errors.append("annotation missing packet_ref pack_id/task_id/mode")
            continue
        annot_hashes = annotation_hashes(annotation)
        for name, value in annot_hashes.items():
            if not valid_sha256(value):
                errors.append(f"{key}: annotation {name} must be non-empty 64-hex sha256")
        annotation_index.setdefault(key, []).append(annotation)
        if annotation.get("label") not in ALLOWED_LABELS:
            errors.append(f"{key}: invalid label {annotation.get('label')!r}")
        if annotation.get("more_credible_signal") not in ALLOWED_CREDIBLE_SIGNALS:
            errors.append(
                f"{key}: invalid more_credible_signal "
                f"{annotation.get('more_credible_signal')!r}"
            )
        rationale = annotation.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{key}: rationale is required")

    for key, packet in sorted(packet_index.items()):
        matching = annotation_index.get(key, [])
        if len(matching) != 1:
            errors.append(f"{key}: expected exactly one annotation, found {len(matching)}")
            continue
        expected = expected_hashes(packet)
        actual = annotation_hashes(matching[0])
        if expected != actual:
            errors.append(
                f"{key}: packet hash mismatch expected={expected} actual={actual}"
            )

    extra_keys = sorted(set(annotation_index) - set(packet_index))
    for key in extra_keys:
        errors.append(f"{key}: annotation has no matching packet")

    if not packets:
        errors.append("no packets were provided")
    if not annotations:
        errors.append("no annotations were provided")

    label_counts: dict[str, int] = {}
    for annotation in annotations:
        label = annotation.get("label")
        if isinstance(label, str):
            label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "schema_version": "judge-disagreement-taxonomy-validation/v1",
        "status": "ok" if not errors else "error",
        "packet_rows": len(packets),
        "annotation_rows": len(annotations),
        "label_counts": dict(sorted(label_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--expect-status", choices=("ok", "error"))
    args = parser.parse_args()

    report = validate_taxonomy(load_jsonl(args.packets), load_jsonl(args.taxonomy))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.expect_status and report["status"] != args.expect_status:
        return 3
    return 0 if report["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
