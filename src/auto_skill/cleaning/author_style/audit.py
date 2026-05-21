"""Audit personal author-style cleaning artifacts before freezing a run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from auto_skill.cleaning.packs import load_jsonl
from auto_skill.mvp import user_examples_from_pack

NATIVE_IMPOSTOR_NEGATIVE_TYPE = "original_av_impostor_cross_topic"
SOURCE_NATIVE_STATUS = "source_native_impostor"
DEFAULT_PUBLIC_FORBIDDEN = (
    "LongLaMP",
    "longlamp",
    "BlogAuthorship",
    "paoramen",
    "Mendeley",
    "hppkn5kbg8",
    "truth.txt",
    "truth_label",
    "mendeley_problem",
    "claimed_problem_author",
    "actual_doc_author",
    "reviewerId",
    "raw_author",
    "author_hash",
    "private_date",
    "private_label",
    "reference_output_private",
    "gpt_rerank",
    "source_config",
    "style_summary",
    "cluster_tags",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_passes_thresholds(
    row: dict[str, Any],
    *,
    audit_thresholds: str = "none",
    min_style_extractability: float,
    min_negative_strength: float,
    max_topic_leakage: float,
    max_model_familiarity: float,
) -> bool:
    if row.get("status") != "success":
        return False
    audit = row.get("audit")
    if not isinstance(audit, dict):
        return False
    if audit_thresholds == "none":
        return True
    if audit_thresholds != "smoke":
        raise ValueError(f"unknown audit threshold policy: {audit_thresholds}")
    if not (audit.get("usable") and audit.get("should_use_for_smoke")):
        return False
    return (
        float(audit.get("style_extractability_1_to_5") or 0) >= min_style_extractability
        and float(audit.get("negative_strength_1_to_5") or 0) >= min_negative_strength
        and float(audit.get("topic_leakage_risk_1_to_5") or 6) <= max_topic_leakage
        and float(audit.get("model_familiarity_risk_1_to_5") or 6) <= max_model_familiarity
    )


def task_refs_by_pack(packs: list[dict[str, Any]]) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        refs[pack_id] = {
            str(task.get("task_id"))
            for task in pack.get("heldout_tasks", [])
            if isinstance(task, dict) and task.get("task_id")
        }
    return refs


def public_pack_text(packs: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(pack, ensure_ascii=False, sort_keys=True) for pack in packs)


def check_public_packs(
    packs: list[dict[str, Any]],
    *,
    forbidden_terms: tuple[str, ...],
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    pack_ids = [str(pack.get("pack_id") or "") for pack in packs]
    duplicate_pack_ids = sorted(
        pack_id for pack_id, count in Counter(pack_ids).items() if count > 1
    )
    if duplicate_pack_ids:
        issues.append(f"duplicate accepted pack ids: {duplicate_pack_ids[:5]}")

    if not packs:
        issues.append("no accepted public packs")

    text = public_pack_text(packs)
    leaked_terms = [term for term in forbidden_terms if term and term in text]
    if leaked_terms:
        issues.append(f"public pack contains forbidden terms: {leaked_terms}")

    example_counts = []
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        examples = user_examples_from_pack(pack)
        train_rows = [
            row
            for row in pack.get("train_examples", [])
            if isinstance(row, dict)
            and isinstance(row.get("desired_output"), dict)
            and row["desired_output"].get("status") == "generated"
        ]
        if len(examples) != len(train_rows):
            issues.append(
                f"{pack_id}: user_examples_from_pack returned {len(examples)} "
                f"for {len(train_rows)} generated train rows"
            )
        if not examples:
            issues.append(f"{pack_id}: no usable train examples")
        for task in pack.get("heldout_tasks", []):
            if isinstance(task, dict) and task.get("desired_output") is not None:
                issues.append(f"{pack_id}: heldout task exposes desired_output")
        example_counts.append(len(examples))

    summary = {
        "packs": len(packs),
        "train_examples_min": min(example_counts) if example_counts else 0,
        "train_examples_max": max(example_counts) if example_counts else 0,
        "forbidden_terms": leaked_terms,
    }
    return issues, summary


def check_private_eval(
    private_rows: list[dict[str, Any]],
    *,
    accepted_pack_ids: set[str],
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    private_pack_ids = {str(row.get("pack_id") or "") for row in private_rows}
    missing = sorted(accepted_pack_ids - private_pack_ids)
    extra = sorted(private_pack_ids - accepted_pack_ids)
    if missing:
        issues.append(f"missing private eval rows for packs: {missing[:5]}")
    if extra:
        issues.append(f"private eval has non-accepted packs: {extra[:5]}")
    for row in private_rows:
        pack_id = str(row.get("pack_id") or "")
        heldout = row.get("heldout_private")
        if not isinstance(heldout, list) or not heldout:
            issues.append(f"{pack_id}: private eval has no heldout_private rows")
            continue
        for task in heldout:
            if (
                not isinstance(task, dict)
                or not str(task.get("reference_output_private") or "").strip()
            ):
                issues.append(f"{pack_id}: heldout private row missing reference output")
                break
    return issues, {"private_eval_rows": len(private_rows)}


def check_gpt_audits(
    audits: list[dict[str, Any]],
    *,
    accepted_pack_ids: set[str],
    audit_thresholds: str,
    min_cluster_tags: int,
    min_style_extractability: float,
    min_negative_strength: float,
    max_topic_leakage: float,
    max_model_familiarity: float,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    by_pack = {str(row.get("pack_id") or ""): row for row in audits if isinstance(row, dict)}
    passed_pack_ids = {
        pack_id
        for pack_id, row in by_pack.items()
        if audit_passes_thresholds(
            row,
            audit_thresholds=audit_thresholds,
            min_style_extractability=min_style_extractability,
            min_negative_strength=min_negative_strength,
            max_topic_leakage=max_topic_leakage,
            max_model_familiarity=max_model_familiarity,
        )
    }
    missing = sorted(accepted_pack_ids - set(by_pack))
    not_passing = sorted(accepted_pack_ids - passed_pack_ids)
    if audit_thresholds != "none" and missing:
        issues.append(f"missing gpt audit rows for accepted packs: {missing[:5]}")
    if audit_thresholds != "none" and not_passing:
        issues.append(f"accepted packs do not pass gpt audit thresholds: {not_passing[:5]}")

    tag_counter: Counter[str] = Counter()
    tag_counts: dict[str, int] = {}
    for pack_id in accepted_pack_ids:
        row = by_pack.get(pack_id, {})
        audit = row.get("audit") if isinstance(row, dict) else None
        tags = audit.get("style_cluster_tags") if isinstance(audit, dict) else None
        clean_tags = (
            [str(tag) for tag in tags if str(tag).strip()] if isinstance(tags, list) else []
        )
        tag_counts[pack_id] = len(clean_tags)
        tag_counter.update(clean_tags)
        if audit_thresholds != "none" and len(clean_tags) < min_cluster_tags:
            issues.append(
                f"{pack_id}: only {len(clean_tags)} style_cluster_tags, "
                f"expected at least {min_cluster_tags}"
            )
    summary = {
        "threshold_policy": audit_thresholds,
        "audit_rows": len(audits),
        "passed_rows": len(passed_pack_ids),
        "accepted_style_cluster_tag_counts": tag_counts,
        "top_style_cluster_tags": tag_counter.most_common(20),
    }
    return issues, summary


def check_hard_negatives(
    negatives: list[dict[str, Any]],
    *,
    accepted_task_refs: dict[str, set[str]],
    require_gpt_selected: bool,
    allow_source_native_negatives: bool,
    require_style_similarity: bool,
    min_negatives_per_heldout: int,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    accepted_pack_ids = set(accepted_task_refs)
    accepted_refs = set().union(*accepted_task_refs.values()) if accepted_task_refs else set()
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    confusability_scores: list[float] = []
    for row in negatives:
        target_ref = str(row.get("target_task_ref") or "")
        pack_id = target_ref.split("::heldout::", maxsplit=1)[0]
        if pack_id not in accepted_pack_ids:
            issues.append(f"negative targets non-accepted pack: {target_ref}")
            continue
        if target_ref not in accepted_refs:
            issues.append(f"negative targets unknown heldout task: {target_ref}")
        by_target[target_ref].append(row)
        if row.get("target_author_hash") == row.get("negative_author_hash"):
            issues.append(f"{target_ref}: negative has same author hash as target")
        if (
            require_style_similarity
            and row.get("match_features", {}).get("style_similarity") is None
        ):
            issues.append(f"{target_ref}: negative missing match_features.style_similarity")
        status = str(row.get("gpt_rerank", {}).get("status") or "not_reranked")
        status_counts[status] += 1
        negative_type = str(row.get("negative_type") or "")
        type_counts[negative_type] += 1
        is_allowed_source_native = (
            allow_source_native_negatives
            and status == SOURCE_NATIVE_STATUS
            and negative_type == NATIVE_IMPOSTOR_NEGATIVE_TYPE
        )
        if require_gpt_selected and status != "selected" and not is_allowed_source_native:
            issues.append(f"{target_ref}: negative is not gpt-selected ({status})")
        score = row.get("gpt_rerank", {}).get("style_confusability_1_to_5")
        if isinstance(score, (int, float)):
            confusability_scores.append(float(score))

    missing_or_weak = sorted(
        target_ref
        for target_ref in accepted_refs
        if len(by_target.get(target_ref, [])) < min_negatives_per_heldout
    )
    if missing_or_weak:
        issues.append(
            f"{len(missing_or_weak)} heldout tasks have fewer than "
            f"{min_negatives_per_heldout} accepted hard negatives; examples: {missing_or_weak[:5]}"
        )

    summary = {
        "hard_negatives": len(negatives),
        "target_refs_with_negatives": len(by_target),
        "required_target_refs": len(accepted_refs),
        "gpt_rerank_status_counts": dict(status_counts),
        "negative_type_counts": dict(type_counts),
        "style_confusability_mean": (
            round(sum(confusability_scores) / len(confusability_scores), 3)
            if confusability_scores
            else None
        ),
    }
    return issues, summary


def audit_author_style_run(
    *,
    run_dir: Path,
    accepted_packs_path: Path | None = None,
    accepted_private_eval_path: Path | None = None,
    accepted_hard_negatives_path: Path | None = None,
    gpt_audits_path: Path | None = None,
    forbidden_terms: tuple[str, ...] = DEFAULT_PUBLIC_FORBIDDEN,
    require_gpt_selected: bool = False,
    allow_source_native_negatives: bool = False,
    require_style_similarity: bool = True,
    min_negatives_per_heldout: int = 1,
    audit_thresholds: str = "none",
    min_cluster_tags: int = 2,
    min_style_extractability: float = 4.0,
    min_negative_strength: float = 3.0,
    max_topic_leakage: float = 4.0,
    max_model_familiarity: float = 3.0,
) -> dict[str, Any]:
    accepted_packs_path = accepted_packs_path or run_dir / "accepted_author_style_packs.jsonl"
    accepted_private_eval_path = (
        accepted_private_eval_path or run_dir / "accepted_author_style_private_eval.jsonl"
    )
    accepted_hard_negatives_path = (
        accepted_hard_negatives_path or run_dir / "accepted_hard_negatives.jsonl"
    )
    gpt_audits_path = gpt_audits_path or run_dir / "gpt55_author_audits.jsonl"
    summary_path = run_dir / "author_style_smoke_summary.json"

    packs = load_jsonl(accepted_packs_path)
    private_rows = load_jsonl(accepted_private_eval_path)
    negatives = load_jsonl(accepted_hard_negatives_path)
    audits = load_jsonl(gpt_audits_path) if gpt_audits_path.exists() else []
    source_summary = read_json(summary_path) if summary_path.exists() else {}

    issues: list[str] = []
    public_issues, public_summary = check_public_packs(packs, forbidden_terms=forbidden_terms)
    issues.extend(public_issues)
    accepted_pack_ids = {str(pack.get("pack_id") or "") for pack in packs}
    task_refs = task_refs_by_pack(packs)
    summary_pack_ids = source_summary.get("accepted_pack_ids")
    if isinstance(summary_pack_ids, list):
        summary_pack_set = {str(pack_id) for pack_id in summary_pack_ids}
        if summary_pack_set != accepted_pack_ids:
            missing_from_summary = sorted(accepted_pack_ids - summary_pack_set)
            extra_in_summary = sorted(summary_pack_set - accepted_pack_ids)
            issues.append(
                "summary accepted_pack_ids mismatch actual accepted packs: "
                f"missing={missing_from_summary[:5]} extra={extra_in_summary[:5]}"
            )
    summary_audits = source_summary.get("gpt_audits")
    if isinstance(summary_audits, dict):
        accepted_count = summary_audits.get("accepted")
        if isinstance(accepted_count, int) and accepted_count != len(packs):
            issues.append(
                f"summary gpt_audits.accepted={accepted_count} but accepted pack rows={len(packs)}"
            )

    private_issues, private_summary = check_private_eval(
        private_rows,
        accepted_pack_ids=accepted_pack_ids,
    )
    issues.extend(private_issues)
    audit_issues, audit_summary = check_gpt_audits(
        audits,
        accepted_pack_ids=accepted_pack_ids,
        audit_thresholds=audit_thresholds,
        min_cluster_tags=min_cluster_tags,
        min_style_extractability=min_style_extractability,
        min_negative_strength=min_negative_strength,
        max_topic_leakage=max_topic_leakage,
        max_model_familiarity=max_model_familiarity,
    )
    issues.extend(audit_issues)
    negative_issues, negative_summary = check_hard_negatives(
        negatives,
        accepted_task_refs=task_refs,
        require_gpt_selected=require_gpt_selected,
        allow_source_native_negatives=allow_source_native_negatives,
        require_style_similarity=require_style_similarity,
        min_negatives_per_heldout=min_negatives_per_heldout,
    )
    issues.extend(negative_issues)

    status = "ok" if not issues else "failed"
    return {
        "schema_version": "author-style-artifact-audit/v1",
        "status": status,
        "run_dir": str(run_dir),
        "paths": {
            "accepted_packs": str(accepted_packs_path),
            "accepted_private_eval": str(accepted_private_eval_path),
            "accepted_hard_negatives": str(accepted_hard_negatives_path),
            "gpt_audits": str(gpt_audits_path),
        },
        "source_summary": {
            key: source_summary.get(key)
            for key in (
                "source",
                "hf_dataset",
                "clean_posts",
                "packs",
                "selected_authors",
                "gpt_audits",
                "gpt_negative_rerank",
            )
            if key in source_summary
        },
        "public_packs": public_summary,
        "private_eval": private_summary,
        "gpt_audits": audit_summary,
        "hard_negatives": negative_summary,
        "issues": issues,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit for personal author-style cleaning artifacts."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--accepted-packs", type=Path)
    parser.add_argument("--accepted-private-eval", type=Path)
    parser.add_argument("--accepted-hard-negatives", type=Path)
    parser.add_argument("--gpt-audits", type=Path)
    parser.add_argument("--forbidden-public-term", action="append", default=[])
    parser.add_argument(
        "--require-gpt-selected-negatives",
        action="store_true",
        help="fail hard negatives that were not selected by GPT reranking",
    )
    parser.add_argument(
        "--allow-non-selected-negatives",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--allow-source-native-negatives", action="store_true")
    parser.add_argument("--allow-missing-style-similarity", action="store_true")
    parser.add_argument("--min-negatives-per-heldout", type=int, default=1)
    parser.add_argument("--audit-thresholds", choices=("none", "smoke"), default="none")
    parser.add_argument("--min-cluster-tags", type=int, default=2)
    parser.add_argument("--min-style-extractability", type=float, default=4.0)
    parser.add_argument("--min-negative-strength", type=float, default=3.0)
    parser.add_argument("--max-topic-leakage", type=float, default=4.0)
    parser.add_argument("--max-model-familiarity", type=float, default=3.0)
    parser.add_argument("--expect-status", choices=["ok", "failed"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    forbidden_terms = DEFAULT_PUBLIC_FORBIDDEN + tuple(args.forbidden_public_term)
    report = audit_author_style_run(
        run_dir=args.run_dir,
        accepted_packs_path=args.accepted_packs,
        accepted_private_eval_path=args.accepted_private_eval,
        accepted_hard_negatives_path=args.accepted_hard_negatives,
        gpt_audits_path=args.gpt_audits,
        forbidden_terms=forbidden_terms,
        require_gpt_selected=(
            args.require_gpt_selected_negatives and not args.allow_non_selected_negatives
        ),
        allow_source_native_negatives=args.allow_source_native_negatives,
        require_style_similarity=not args.allow_missing_style_similarity,
        min_negatives_per_heldout=args.min_negatives_per_heldout,
        audit_thresholds=args.audit_thresholds,
        min_cluster_tags=args.min_cluster_tags,
        min_style_extractability=args.min_style_extractability,
        min_negative_strength=args.min_negative_strength,
        max_topic_leakage=args.max_topic_leakage,
        max_model_familiarity=args.max_model_familiarity,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.expect_status and report["status"] != args.expect_status:
        print(
            f"error: expected status {args.expect_status}, got {report['status']}",
            file=sys.stderr,
        )
        return 1
    if args.expect_status:
        return 0
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
