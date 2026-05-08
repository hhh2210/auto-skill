#!/usr/bin/env python3
"""Report readiness of the local expanded cleaned-example artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.benchmark_flow import audit_benchmark_flow  # noqa: E402
from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.schemas import SchemaValidationError, validate_artifact_rows  # noqa: E402


@dataclass(frozen=True)
class AuditSpec:
    path: Path
    source: str | None = None
    min_success: int = 1


def parse_audit_spec(raw: str) -> AuditSpec:
    """Parse PATH[:SOURCE[:MIN_SUCCESS]] for audit coverage checks."""

    parts = raw.rsplit(":", 2)
    if len(parts) == 1:
        return AuditSpec(path=Path(raw))
    if len(parts) == 2:
        path, source = parts
        return AuditSpec(path=Path(path), source=source or None)
    path, source, min_success_raw = parts
    try:
        min_success = int(min_success_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"audit spec min success must be an integer: {raw!r}"
        ) from exc
    if min_success < 1:
        raise argparse.ArgumentTypeError(f"audit spec min success must be positive: {raw!r}")
    return AuditSpec(path=Path(path), source=source or None, min_success=min_success)


def latest_generation_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        prompt_sha = str(row.get("prompt_sha256") or "")
        if job_id and prompt_sha:
            latest[(job_id, prompt_sha)] = row
    return latest


def generation_job_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys = set()
    for row in rows:
        job_id = str(row.get("job_id") or "")
        prompt_sha = str(row.get("prompt_sha256") or "")
        if job_id and prompt_sha:
            keys.add((job_id, prompt_sha))
    return keys


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("source") or "unknown") for row in rows).items()))


def frozen_pack_summary(packs: list[dict[str, Any]]) -> dict[str, Any]:
    train_examples = 0
    heldout_tasks = 0
    frozen_examples = 0
    for pack in packs:
        examples = pack.get("train_examples", [])
        tasks = pack.get("heldout_tasks", [])
        train_examples += len(examples)
        heldout_tasks += len(tasks)
        frozen_examples += sum(
            1
            for example in examples
            if isinstance(example.get("desired_output"), dict)
            and example["desired_output"].get("status") == "generated"
            and isinstance(example["desired_output"].get("text"), str)
            and example["desired_output"]["text"].strip()
        )
    return {
        "packs": len(packs),
        "sources": source_counts(packs),
        "train_examples": train_examples,
        "heldout_tasks": heldout_tasks,
        "frozen_train_examples": frozen_examples,
    }


def audit_file_summary(spec: AuditSpec) -> dict[str, Any]:
    rows = load_jsonl(spec.path)
    validate_artifact_rows(rows, kind="eval", label=str(spec.path))
    return {
        "path": str(spec.path),
        "required_source": spec.source,
        "min_success": spec.min_success,
        "rows": len(rows),
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "source_counts": source_counts(rows),
        "success_rows": sum(1 for row in rows if row.get("status") == "success"),
    }


def require_dict_rows(rows: list[Any], *, label: str) -> None:
    """Raise a structured error when a JSONL artifact contains non-object rows."""

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"{label}: row {index + 1} must be an object, got {type(row).__name__}")


def maybe_mimo_subset_summary(
    *,
    args: argparse.Namespace,
    splits: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Validate and summarize an optional local MIMO frozen subset."""

    errors: list[str] = []
    warnings: list[str] = []
    paths = [
        args.mimo_subset_packs,
        args.mimo_subset_private_eval,
        args.mimo_subset_jobs,
        args.mimo_subset_generated_outputs,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        message = f"MIMO subset artifacts missing: {missing}"
        if args.require_mimo_subset:
            errors.append(message)
        else:
            warnings.append(message)
        return None, errors, warnings

    try:
        packs = load_jsonl(args.mimo_subset_packs)
        private_rows = load_jsonl(args.mimo_subset_private_eval)
        jobs = load_jsonl(args.mimo_subset_jobs)
        generated_rows = load_jsonl(args.mimo_subset_generated_outputs)
        require_dict_rows(packs, label=str(args.mimo_subset_packs))
        require_dict_rows(private_rows, label=str(args.mimo_subset_private_eval))
        require_dict_rows(jobs, label=str(args.mimo_subset_jobs))
        require_dict_rows(generated_rows, label=str(args.mimo_subset_generated_outputs))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"MIMO subset artifacts invalid: {exc}")
        return None, errors, warnings

    try:
        flow = audit_benchmark_flow(
            splits=splits,
            packs=packs,
            private_rows=private_rows,
            generation_jobs=jobs,
            generated_rows=generated_rows,
        )
        if flow.errors:
            errors.extend(f"MIMO subset benchmark_flow: {error}" for error in flow.errors)
        warnings.extend(f"MIMO subset benchmark_flow: {warning}" for warning in flow.warnings)

        pack_summary = frozen_pack_summary(packs)
        latest_rows = latest_generation_rows(generated_rows)
        latest_status_counts = Counter(str(row.get("status")) for row in latest_rows.values())
        if len(latest_rows) != len(jobs):
            errors.append(
                f"MIMO subset latest generation rows {len(latest_rows)} "
                f"do not match jobs {len(jobs)}"
            )
        if latest_status_counts != {"success": len(jobs)}:
            errors.append(
                "MIMO subset latest generation rows are not all success: "
                f"{dict(latest_status_counts)}"
            )
        if pack_summary["frozen_train_examples"] != pack_summary["train_examples"]:
            errors.append(
                "MIMO subset not all train examples are frozen: "
                f"{pack_summary['frozen_train_examples']}/{pack_summary['train_examples']}"
            )
        expected_source_counts = {
            "PresentBench": args.expect_mimo_subset_present_packs,
            "WritingBench": args.expect_mimo_subset_writing_packs,
        }
        if pack_summary["packs"] != args.expect_mimo_subset_packs:
            errors.append(
                "MIMO subset expected "
                f"{args.expect_mimo_subset_packs} packs, got {pack_summary['packs']}"
            )
        if pack_summary["train_examples"] != args.expect_mimo_subset_train_examples:
            errors.append(
                "MIMO subset expected "
                f"{args.expect_mimo_subset_train_examples} train examples, "
                f"got {pack_summary['train_examples']}"
            )
        if pack_summary["heldout_tasks"] != args.expect_mimo_subset_heldout_tasks:
            errors.append(
                "MIMO subset expected "
                f"{args.expect_mimo_subset_heldout_tasks} heldout tasks, "
                f"got {pack_summary['heldout_tasks']}"
            )
        if len(jobs) != args.expect_mimo_subset_generation_jobs:
            errors.append(
                "MIMO subset expected "
                f"{args.expect_mimo_subset_generation_jobs} generation jobs, got {len(jobs)}"
            )
        if pack_summary["sources"] != expected_source_counts:
            errors.append(
                "MIMO subset source counts mismatch: "
                f"expected {expected_source_counts}, got {pack_summary['sources']}"
            )
    except (SchemaValidationError, TypeError, AttributeError) as exc:
        errors.append(f"MIMO subset artifacts invalid: {exc}")
        return None, errors, warnings

    summary = {
        "status": "ok" if not errors else "failed",
        "packs": {"path": str(args.mimo_subset_packs), **pack_summary},
        "private_eval": {
            "path": str(args.mimo_subset_private_eval),
            "rows": len(private_rows),
            "sources": source_counts(private_rows),
        },
        "generation_jobs": {"path": str(args.mimo_subset_jobs), "rows": len(jobs)},
        "generated_outputs": {
            "path": str(args.mimo_subset_generated_outputs),
            "rows": len(generated_rows),
            "latest_rows": len(latest_rows),
            "latest_status_counts": dict(sorted(latest_status_counts.items())),
        },
        "benchmark_flow": {
            "status": "ok" if flow.ok else "failed",
            "errors": list(flow.errors),
            "warnings": list(flow.warnings),
        },
    }
    return summary, errors, warnings


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    splits = load_jsonl(args.splits)
    packs = load_jsonl(args.packs)
    private_rows = load_jsonl(args.private_eval)
    jobs = load_jsonl(args.jobs)
    generated_rows = load_jsonl(args.generated_outputs)

    try:
        validate_artifact_rows(
            generated_rows,
            kind="generated_outputs",
            label=str(args.generated_outputs),
        )
    except SchemaValidationError as exc:
        errors.append(f"generated outputs schema invalid: {exc}")

    flow = audit_benchmark_flow(
        splits=splits,
        packs=packs,
        private_rows=private_rows,
        generation_jobs=jobs,
        generated_rows=generated_rows,
    )
    errors.extend(f"benchmark_flow: {error}" for error in flow.errors)
    warnings.extend(f"benchmark_flow: {warning}" for warning in flow.warnings)

    pack_summary = frozen_pack_summary(packs)
    if pack_summary["packs"] != args.expect_packs:
        errors.append(f"expected {args.expect_packs} packs, got {pack_summary['packs']}")
    if pack_summary["train_examples"] != args.expect_train_examples:
        errors.append(
            "expected "
            f"{args.expect_train_examples} train examples, got {pack_summary['train_examples']}"
        )
    if pack_summary["heldout_tasks"] != args.expect_heldout_tasks:
        errors.append(
            "expected "
            f"{args.expect_heldout_tasks} heldout tasks, got {pack_summary['heldout_tasks']}"
        )
    if pack_summary["frozen_train_examples"] != pack_summary["train_examples"]:
        errors.append(
            "not all train examples are frozen: "
            f"{pack_summary['frozen_train_examples']}/{pack_summary['train_examples']}"
        )

    expected_job_keys = generation_job_keys(jobs)
    latest_rows = latest_generation_rows(generated_rows)
    latest_keys = set(latest_rows)
    latest_status_counts = Counter(str(row.get("status")) for row in latest_rows.values())
    if len(jobs) != args.expect_generation_jobs:
        errors.append(f"expected {args.expect_generation_jobs} generation jobs, got {len(jobs)}")
    if len(latest_rows) != len(jobs):
        errors.append(f"latest generation rows {len(latest_rows)} do not match jobs {len(jobs)}")
    if latest_keys != expected_job_keys:
        missing = sorted(expected_job_keys - latest_keys)[:5]
        extra = sorted(latest_keys - expected_job_keys)[:5]
        errors.append(
            "latest generation row keys do not match jobs: "
            f"missing={missing}, extra={extra}"
        )
    if latest_status_counts != {"success": len(jobs)}:
        errors.append(f"latest generation rows are not all success: {dict(latest_status_counts)}")

    required_audits = []
    for spec in args.required_audit:
        try:
            summary = audit_file_summary(spec)
        except (OSError, SchemaValidationError) as exc:
            errors.append(f"required audit invalid {spec.path}: {exc}")
            continue
        required_audits.append(summary)
        if spec.source is not None and set(summary["source_counts"]) != {spec.source}:
            errors.append(
                f"required audit source mismatch: {spec.path}: "
                f"expected {spec.source}, got {summary['source_counts']}"
            )
        if summary["success_rows"] < spec.min_success:
            errors.append(
                f"required audit has too few success rows: {spec.path}: "
                f"{summary['success_rows']} < {spec.min_success}"
            )
        if set(summary["status_counts"]) != {"success"}:
            errors.append(
                f"required audit has non-success rows: {spec.path}: {summary['status_counts']}"
            )

    optional_audits = []
    for spec in args.optional_audit:
        try:
            summary = audit_file_summary(spec)
        except (OSError, SchemaValidationError) as exc:
            warnings.append(f"optional audit invalid {spec.path}: {exc}")
            continue
        optional_audits.append(summary)
        if spec.source is not None and set(summary["source_counts"]) != {spec.source}:
            warnings.append(
                f"optional audit source mismatch: {spec.path}: "
                f"expected {spec.source}, got {summary['source_counts']}"
            )
        if summary["success_rows"] < spec.min_success:
            warnings.append(
                f"optional audit has too few success rows: {spec.path}: "
                f"{summary['success_rows']} < {spec.min_success}"
            )
        if set(summary["status_counts"]) != {"success"}:
            warnings.append(
                f"optional audit has non-success rows: {spec.path}: {summary['status_counts']}"
            )

    mimo_subset, mimo_errors, mimo_warnings = maybe_mimo_subset_summary(
        args=args,
        splits=splits,
    )
    if args.require_mimo_subset:
        errors.extend(mimo_errors)
    else:
        warnings.extend(mimo_errors)
    warnings.extend(mimo_warnings)

    status = "ready" if not errors else "not_ready"
    artifacts = {
        "splits": {
            "path": str(args.splits),
            "rows": len(splits),
            "sources": source_counts(splits),
        },
        "packs": {"path": str(args.packs), **pack_summary},
        "private_eval": {
            "path": str(args.private_eval),
            "rows": len(private_rows),
            "sources": source_counts(private_rows),
        },
        "generation_jobs": {"path": str(args.jobs), "rows": len(jobs)},
        "generated_outputs": {
            "path": str(args.generated_outputs),
            "rows": len(generated_rows),
            "latest_rows": len(latest_rows),
            "latest_status_counts": dict(sorted(latest_status_counts.items())),
        },
        "required_audits": required_audits,
        "optional_audits": optional_audits,
    }
    if mimo_subset is not None:
        artifacts["mimo_subset"] = mimo_subset

    return {
        "schema_version": "expanded-cleaning-status/v1",
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("runs/expanded/fewshot_splits.30wb_20pb.jsonl"),
    )
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("runs/expanded/example_packs.30wb_20pb.qwen.v1.jsonl"),
    )
    parser.add_argument(
        "--private-eval",
        type=Path,
        default=Path("runs/expanded/example_private_eval.30wb_20pb.jsonl"),
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("runs/expanded/example_generation_jobs.30wb_20pb.jsonl"),
    )
    parser.add_argument(
        "--generated-outputs",
        type=Path,
        default=Path("runs/expanded/generated_desired_outputs.30wb_20pb.qwen.jsonl"),
    )
    parser.add_argument(
        "--required-audit",
        type=parse_audit_spec,
        action="append",
        help="Audit spec PATH[:SOURCE[:MIN_SUCCESS]].",
    )
    parser.add_argument(
        "--optional-audit",
        type=parse_audit_spec,
        action="append",
        help="Audit spec PATH[:SOURCE[:MIN_SUCCESS]].",
    )
    parser.add_argument("--expect-packs", type=int, default=50)
    parser.add_argument("--expect-train-examples", type=int, default=150)
    parser.add_argument("--expect-heldout-tasks", type=int, default=100)
    parser.add_argument("--expect-generation-jobs", type=int, default=150)
    parser.add_argument(
        "--mimo-subset-packs",
        type=Path,
        default=Path("runs/expanded/example_packs.30wb_20pb.mimo.sample.v1.jsonl"),
    )
    parser.add_argument(
        "--mimo-subset-private-eval",
        type=Path,
        default=Path("runs/expanded/example_private_eval.30wb_20pb.mimo.sample.jsonl"),
    )
    parser.add_argument(
        "--mimo-subset-jobs",
        type=Path,
        default=Path("runs/expanded/example_generation_jobs.30wb_20pb.mimo.sample.jsonl"),
    )
    parser.add_argument(
        "--mimo-subset-generated-outputs",
        type=Path,
        default=Path(
            "runs/expanded/generated_desired_outputs.30wb_20pb.mimo.sample.latest_success.jsonl"
        ),
    )
    parser.add_argument(
        "--require-mimo-subset",
        action="store_true",
        help="Fail when the optional local MIMO frozen subset is missing or invalid.",
    )
    parser.add_argument("--expect-mimo-subset-packs", type=int, default=15)
    parser.add_argument("--expect-mimo-subset-train-examples", type=int, default=45)
    parser.add_argument("--expect-mimo-subset-heldout-tasks", type=int, default=30)
    parser.add_argument("--expect-mimo-subset-generation-jobs", type=int, default=45)
    parser.add_argument("--expect-mimo-subset-writing-packs", type=int, default=12)
    parser.add_argument("--expect-mimo-subset-present-packs", type=int, default=3)
    parser.add_argument("--expect-status", choices=["ready", "not_ready"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.required_audit is None:
        args.required_audit = [
            AuditSpec(
                path=Path(
                    "runs/expanded/"
                    "train_example_quality_audit.30wb_20pb.mimo.writingbench.sample6.jsonl"
                ),
                source="WritingBench",
                min_success=6,
            ),
            AuditSpec(
                path=Path(
                    "runs/expanded/"
                    "train_example_quality_audit.30wb_20pb.qwen.presentbench.sample2.jsonl"
                ),
                source="PresentBench",
                min_success=2,
            ),
        ]
    if args.optional_audit is None:
        args.optional_audit = [
            AuditSpec(
                path=Path(
                    "runs/expanded/"
                    "train_example_quality_audit.30wb_20pb.mimo.presentbench.sample2.jsonl"
                ),
                source="PresentBench",
                min_success=1,
            ),
        ]

    try:
        report = build_report(args)
    except OSError as exc:
        report = {
            "schema_version": "expanded-cleaning-status/v1",
            "status": "not_ready",
            "errors": [str(exc)],
            "warnings": [],
            "artifacts": {},
        }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.expect_status and report["status"] != args.expect_status:
        print(
            f"error: expected status {args.expect_status}, got {report['status']}",
            file=sys.stderr,
        )
        return 1
    if args.expect_status:
        return 0
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
