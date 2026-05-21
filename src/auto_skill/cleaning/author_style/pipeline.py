"""CLI orchestration for author-style data cleaning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from auto_skill.cleaning.author_style.common import CleanPost, write_jsonl
from auto_skill.cleaning.author_style.gpt import (
    audit_passes_thresholds,
    audit_threshold_policy,
    filter_by_pack_ids,
    filter_negatives_by_pack_ids,
    pack_ids_with_negative_coverage,
    run_gpt_audits,
    run_gpt_negative_rerank,
)
from auto_skill.cleaning.author_style.packs import (
    build_public_pack,
    build_split_row,
    choose_authors,
    clean_post_row,
    find_hard_negatives,
    find_mendeley_native_impostor_negatives,
    merge_negative_rows,
    pack_prefix_for_source_choice,
    private_author_row,
    source_inventory_row,
    source_native_impostor_rows,
)
from auto_skill.cleaning.author_style.sources import load_posts


def build_report(
    *,
    posts: list[CleanPost],
    selected: list[tuple[str, list[CleanPost], list[CleanPost]]],
    packs: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    args: argparse.Namespace,
    accepted_pack_ids: set[str] | None = None,
) -> dict[str, Any]:
    audit_successes = [row for row in audits if row.get("status") == "success"]
    usable_audits = [
        row
        for row in audit_successes
        if row.get("audit", {}).get("usable") and row.get("audit", {}).get("should_use_for_smoke")
    ]
    accepted_audits = [row for row in audit_successes if audit_passes_thresholds(row, args)]
    policy_accepted_pack_ids = (
        {str(row["pack_id"]) for row in accepted_audits}
        if args.gpt_audit
        else {str(pack["pack_id"]) for pack in packs}
    )
    final_accepted_pack_ids = (
        set(accepted_pack_ids) if accepted_pack_ids is not None else policy_accepted_pack_ids
    )
    policy = audit_threshold_policy(args)
    rerank_status_counts = Counter(
        str(row.get("gpt_rerank", {}).get("status", "not_enabled"))
        for row in negatives
        if isinstance(row, dict)
    )
    rerank_scores = [
        float(row["gpt_rerank"]["style_confusability_1_to_5"])
        for row in negatives
        if isinstance(row.get("gpt_rerank"), dict)
        and isinstance(row["gpt_rerank"].get("style_confusability_1_to_5"), (int, float))
    ]
    return {
        "schema_version": "author-style-smoke-report/v1",
        "source": args.source,
        "hf_dataset": args.hf_dataset,
        "scanned_max_rows": args.max_rows,
        "clean_posts": len(posts),
        "selected_authors": len(selected),
        "packs": len(packs),
        "hard_negatives": len(negatives),
        "gpt_audits": {
            "enabled": args.gpt_audit,
            "threshold_policy": policy,
            "rows": len(audits),
            "success": len(audit_successes),
            "usable": len(usable_audits),
            "threshold_accepted": len(accepted_audits),
            "policy_accepted": len(accepted_audits),
            "accepted": len(final_accepted_pack_ids),
            "negative_coverage_rejected": len(
                policy_accepted_pack_ids - final_accepted_pack_ids
            ),
            "thresholds": None
            if policy == "none"
            else {
                "min_style_extractability": args.min_style_extractability,
                "min_negative_strength": args.min_negative_strength,
                "max_topic_leakage": args.max_topic_leakage,
                "max_model_familiarity": args.max_model_familiarity,
            },
        },
        "gpt_negative_rerank": {
            "enabled": args.gpt_rerank_negatives,
            "status_counts": dict(sorted(rerank_status_counts.items())),
            "style_confusability_mean": (
                round(sum(rerank_scores) / len(rerank_scores), 3) if rerank_scores else None
            ),
            "style_confusability_rows": len(rerank_scores),
        },
        "pack_ids": [pack["pack_id"] for pack in packs],
        "accepted_pack_ids": [
            pack["pack_id"] for pack in packs if pack["pack_id"] in final_accepted_pack_ids
        ],
        "artifact_paths": {
            "source_inventory": str(args.out_dir / "source_inventory.json"),
            "clean_posts": str(args.out_dir / "clean_posts.jsonl"),
            "private_eval": str(args.out_dir / "author_style_private_eval.jsonl"),
            "splits": str(args.out_dir / "author_style_splits.jsonl"),
            "packs": str(args.out_dir / "author_style_packs.jsonl"),
            "hard_negative_candidates": str(args.out_dir / "hard_negative_candidates.jsonl"),
            "hard_negatives": str(args.out_dir / "hard_negatives.jsonl"),
            "native_impostor_negatives": str(args.out_dir / "native_impostor_negatives.jsonl"),
            "gpt_audits": str(args.out_dir / "gpt55_author_audits.jsonl"),
            "accepted_packs": str(args.out_dir / "accepted_author_style_packs.jsonl"),
            "accepted_private_eval": str(args.out_dir / "accepted_author_style_private_eval.jsonl"),
            "accepted_hard_negatives": str(args.out_dir / "accepted_hard_negatives.jsonl"),
            "accepted_native_impostor_negatives": str(
                args.out_dir / "accepted_native_impostor_negatives.jsonl"
            ),
            "accepted_hard_negatives_plus_native": str(
                args.out_dir / "accepted_hard_negatives_plus_native.jsonl"
            ),
            "summary": str(args.out_dir / "author_style_smoke_summary.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("blog", "longlamp-topic", "longlamp-product", "mendeley-reddit"),
        default="blog",
    )
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument(
        "--mendeley-path",
        type=Path,
        default=None,
        help="Path to the Mendeley Reddit Cross-Topic zip or extracted root.",
    )
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--authors", type=int, default=6)
    parser.add_argument("--train-posts", type=int, default=6)
    parser.add_argument("--heldout-posts", type=int, default=3)
    parser.add_argument("--negatives-per-heldout", type=int, default=2)
    parser.add_argument("--negative-prefilter-size", type=int, default=24)
    parser.add_argument("--negative-author-cap", type=int, default=2)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--longlamp-profile-items-per-row", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/author_style/smoke"))
    parser.add_argument("--gpt-audit", action="store_true")
    parser.add_argument("--gpt-limit-authors", type=int, default=4)
    parser.add_argument("--gpt-model", default="gpt-5.5")
    parser.add_argument("--auth", type=Path, default=Path("~/.codex/auth.json").expanduser())
    parser.add_argument("--service-tier", default=None)
    parser.add_argument("--gpt-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--gpt-num-threads", type=int, default=32)
    parser.add_argument("--gpt-max-attempts", type=int, default=2)
    parser.add_argument("--gpt-rerank-negatives", action="store_true")
    parser.add_argument("--gpt-rerank-num-threads", type=int, default=48)
    parser.add_argument("--gpt-rerank-candidates", type=int, default=16)
    parser.add_argument("--gpt-rerank-candidate-chars", type=int, default=520)
    parser.add_argument("--allow-fallback-accepted-negatives", action="store_true")
    parser.add_argument(
        "--audit-thresholds",
        choices=("none", "smoke"),
        default="none",
        help=(
            "Audit gate for accepted packs. 'none' disables GPT audit threshold "
            "filtering; 'smoke' uses the style/negative/leak/familiarity thresholds."
        ),
    )
    parser.add_argument("--min-style-extractability", type=float, default=4.0)
    parser.add_argument("--min-negative-strength", type=float, default=3.0)
    parser.add_argument("--max-topic-leakage", type=float, default=4.0)
    parser.add_argument("--max-model-familiarity", type=float, default=3.0)
    args = parser.parse_args()
    if args.source == "mendeley-reddit":
        if args.train_posts == 6:
            args.train_posts = 4
        if args.heldout_posts == 3:
            args.heldout_posts = 1
        if args.max_words == 900:
            args.max_words = 2_500
    if args.gpt_max_attempts < 1:
        parser.error("--gpt-max-attempts must be positive")
    if args.hf_dataset is None:
        if args.source == "blog":
            args.hf_dataset = "paoramen/blog-authorship-corpus"
        elif args.source in {"longlamp-topic", "longlamp-product"}:
            args.hf_dataset = "LongLaMP/LongLaMP"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    posts = load_posts(args)
    selected = choose_authors(
        posts,
        authors=args.authors,
        train_posts=args.train_posts,
        heldout_posts=args.heldout_posts,
    )
    if not selected:
        raise SystemExit(
            "No authors met the clean-post threshold; raise --max-rows or lower thresholds."
        )

    pack_ids_by_author = {
        author: f"{pack_prefix_for_source_choice(args.source)}_{index:04d}"
        for index, (author, _, _) in enumerate(selected, start=1)
    }
    packs = [
        build_public_pack(pack_ids_by_author[author], train, heldout)
        for author, train, heldout in selected
    ]
    split_rows = [
        build_split_row(pack_ids_by_author[author], train, heldout)
        for author, train, heldout in selected
    ]
    private_authors = [
        private_author_row(pack_ids_by_author[author], author, train, heldout)
        for author, train, heldout in selected
    ]
    prefilter_size = (
        max(args.negatives_per_heldout, args.negative_prefilter_size)
        if args.gpt_rerank_negatives
        else args.negatives_per_heldout
    )
    negative_candidates = find_hard_negatives(
        selected,
        posts,
        negatives_per_heldout=prefilter_size,
        negative_author_cap=args.negative_author_cap,
        pack_ids_by_author=pack_ids_by_author,
    )
    if args.source == "mendeley-reddit":
        negative_candidates = merge_negative_rows(
            find_mendeley_native_impostor_negatives(
                selected,
                posts,
                pack_ids_by_author=pack_ids_by_author,
            )
            + negative_candidates
        )
    negatives = negative_candidates
    if args.gpt_rerank_negatives:
        negatives = run_gpt_negative_rerank(packs, private_authors, negative_candidates, args)
    native_negatives = source_native_impostor_rows(negative_candidates)
    clean_post_rows = [clean_post_row(post) for post in posts]

    audits: list[dict[str, Any]] = []
    if args.gpt_audit:
        audits = run_gpt_audits(packs, negatives, private_authors, args)
    if args.gpt_audit:
        audit_candidate_pack_ids = {
            str(row["pack_id"]) for row in audits if audit_passes_thresholds(row, args)
        }
    else:
        audit_candidate_pack_ids = {str(pack["pack_id"]) for pack in packs}
    candidate_accepted_hard_negatives = filter_negatives_by_pack_ids(
        negatives,
        audit_candidate_pack_ids,
        require_gpt_selected=(
            args.gpt_rerank_negatives and not args.allow_fallback_accepted_negatives
        ),
    )
    accepted_pack_ids = pack_ids_with_negative_coverage(
        private_authors,
        candidate_accepted_hard_negatives,
        audit_candidate_pack_ids,
        min_negatives_per_heldout=args.negatives_per_heldout,
    )

    (args.out_dir / "source_inventory.json").write_text(
        json.dumps(source_inventory_row(args, posts), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.out_dir / "clean_posts.jsonl", clean_post_rows)
    write_jsonl(args.out_dir / "author_style_private_eval.jsonl", private_authors)
    write_jsonl(args.out_dir / "author_style_splits.jsonl", split_rows)
    write_jsonl(args.out_dir / "author_style_packs.jsonl", packs)
    write_jsonl(args.out_dir / "hard_negative_candidates.jsonl", negative_candidates)
    write_jsonl(args.out_dir / "hard_negatives.jsonl", negatives)
    write_jsonl(args.out_dir / "native_impostor_negatives.jsonl", native_negatives)
    write_jsonl(args.out_dir / "gpt55_author_audits.jsonl", audits)
    write_jsonl(
        args.out_dir / "accepted_author_style_packs.jsonl",
        filter_by_pack_ids(packs, accepted_pack_ids),
    )
    write_jsonl(
        args.out_dir / "accepted_author_style_private_eval.jsonl",
        filter_by_pack_ids(private_authors, accepted_pack_ids),
    )
    accepted_hard_negatives = filter_negatives_by_pack_ids(
        negatives,
        accepted_pack_ids,
        require_gpt_selected=(
            args.gpt_rerank_negatives and not args.allow_fallback_accepted_negatives
        ),
    )
    accepted_native_negatives = filter_negatives_by_pack_ids(
        native_negatives,
        accepted_pack_ids,
        require_gpt_selected=False,
    )
    write_jsonl(args.out_dir / "accepted_hard_negatives.jsonl", accepted_hard_negatives)
    write_jsonl(
        args.out_dir / "accepted_native_impostor_negatives.jsonl",
        accepted_native_negatives,
    )
    write_jsonl(
        args.out_dir / "accepted_hard_negatives_plus_native.jsonl",
        merge_negative_rows(accepted_hard_negatives + accepted_native_negatives),
    )
    report = build_report(
        posts=posts,
        selected=selected,
        packs=packs,
        negatives=negatives,
        audits=audits,
        args=args,
        accepted_pack_ids=accepted_pack_ids,
    )
    (args.out_dir / "author_style_smoke_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
