"""Stop/go diagnostics for personal coding-style data sources."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "coding-style-source-diagnostic/v1"

FEATURE_NORMALIZERS = {
    "avg_line_length": 80.0,
    "indent_spaces_per_line": 8.0,
    "indent_tabs_per_line": 2.0,
    "comment_line_rate": 0.4,
    "blank_line_rate": 0.4,
    "brace_next_line_rate": 1.0,
    "semicolon_per_line": 1.0,
    "snake_identifier_rate": 1.0,
    "camel_identifier_rate": 1.0,
    "short_identifier_rate": 1.0,
}

DEFAULT_EXCLUDE_PATH_MARKERS = (
    "/vendor/",
    "/third_party/",
    "/generated/",
    "/node_modules/",
    "/dist/",
)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(float(value)):
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 3),
        "mean": mean(values),
        "max": round(max(values), 3),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def identifier_tokens(code: str) -> list[str]:
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code)


def code_style_features(code: str) -> dict[str, float]:
    lines = code.splitlines()
    nonempty = [line for line in lines if line.strip()]
    line_count = max(1, len(lines))
    identifiers = identifier_tokens(code)
    identifier_count = max(1, len(identifiers))
    snake = sum(1 for token in identifiers if "_" in token and token.lower() == token)
    camel = sum(1 for token in identifiers if re.search(r"[a-z][A-Z]", token))
    short = sum(1 for token in identifiers if len(token) <= 2)
    brace_next_line = sum(1 for line in nonempty if line.strip() == "{")
    comments = sum(
        1
        for line in lines
        if line.strip().startswith(("#", "//", "/*", "*", "--"))
    )
    blank = sum(1 for line in lines if not line.strip())
    spaces = sum(len(match.group(0)) for line in lines if (match := re.match(r" +", line)))
    tabs = sum(len(match.group(0)) for line in lines if (match := re.match(r"\t+", line)))
    return {
        "avg_line_length": round(sum(len(line) for line in lines) / line_count, 3),
        "indent_spaces_per_line": round(spaces / line_count, 3),
        "indent_tabs_per_line": round(tabs / line_count, 3),
        "comment_line_rate": round(comments / line_count, 3),
        "blank_line_rate": round(blank / line_count, 3),
        "brace_next_line_rate": round(brace_next_line / max(1, len(nonempty)), 3),
        "semicolon_per_line": round(code.count(";") / line_count, 3),
        "snake_identifier_rate": round(snake / identifier_count, 3),
        "camel_identifier_rate": round(camel / identifier_count, 3),
        "short_identifier_rate": round(short / identifier_count, 3),
    }


def average_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row.get("style_features", {})})
    averaged = {}
    for key in keys:
        values = [
            value
            for row in rows
            if (value := numeric(row.get("style_features", {}).get(key))) is not None
        ]
        if values:
            averaged[key] = float(mean(values))
    return averaged


def style_similarity(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    distances = []
    for key, normalizer in FEATURE_NORMALIZERS.items():
        left_value = numeric(left.get(key))
        right_value = numeric(right.get(key))
        if left_value is None or right_value is None:
            continue
        distances.append(abs(left_value - right_value) / normalizer)
    if not distances:
        return None
    return round(max(0.0, 1.0 - min(1.0, sum(distances) / len(distances))), 3)


def is_clean_fragment(
    row: dict[str, Any],
    *,
    cutoff: date,
    min_lines: int,
    max_lines: int,
) -> tuple[bool, list[str]]:
    flags = []
    code = row.get("code")
    if not isinstance(code, str) or not code.strip():
        flags.append("missing_code")
    line_count = len(code.splitlines()) if isinstance(code, str) else 0
    if line_count < min_lines:
        flags.append("too_short")
    if line_count > max_lines:
        flags.append("too_long")
    author = row.get("author_id")
    if not isinstance(author, str) or not author.strip():
        flags.append("missing_author_id")
    timestamp = parse_date(row.get("date") or row.get("committed_at") or row.get("created_at"))
    if timestamp is None:
        flags.append("missing_or_invalid_date")
    elif timestamp >= cutoff:
        flags.append("post_ai_cutoff")
    path = str(row.get("path") or "")
    normalized_path = f"/{path.strip('/')}"
    if any(marker in normalized_path for marker in DEFAULT_EXCLUDE_PATH_MARKERS):
        flags.append("vendor_or_generated_path")
    for key, flag in (
        ("is_bot", "bot_authored"),
        ("is_merge", "merge_commit"),
        ("is_generated", "generated_code"),
        ("is_formatting_only", "formatting_only"),
        ("is_vendor", "vendor_code"),
    ):
        if row.get(key) is True:
            flags.append(flag)
    if row.get("license_ok") is False:
        flags.append("license_not_ok")
    return not flags, flags


def clean_fragments(
    rows: list[dict[str, Any]],
    *,
    pre_ai_cutoff: str,
    min_lines: int,
    max_lines: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    cutoff = datetime.strptime(pre_ai_cutoff, "%Y-%m-%d").date()
    rejected: Counter[str] = Counter()
    clean = []
    for index, row in enumerate(rows):
        ok, flags = is_clean_fragment(
            row,
            cutoff=cutoff,
            min_lines=min_lines,
            max_lines=max_lines,
        )
        if not ok:
            rejected.update(flags)
            continue
        enriched = dict(row)
        enriched["row_index"] = index
        enriched["parsed_date"] = parse_date(
            row.get("date") or row.get("committed_at") or row.get("created_at")
        )
        enriched["style_features"] = code_style_features(str(row["code"]))
        clean.append(enriched)
    clean.sort(
        key=lambda item: (
            item["parsed_date"],
            str(item.get("source_id") or item["row_index"]),
        )
    )
    return clean, rejected


def temporal_author_splits(
    rows: list[dict[str, Any]],
    *,
    min_train: int,
    min_heldout: int,
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_author[str(row["author_id"])].append(row)
    splits = {}
    for author, author_rows in by_author.items():
        author_rows.sort(
            key=lambda item: (item["parsed_date"], str(item.get("source_id") or item["row_index"]))
        )
        if len(author_rows) < min_train + min_heldout:
            continue
        splits[author] = (
            author_rows[:min_train],
            author_rows[-min_heldout:],
        )
    return splits


def source_pool_metrics(
    splits: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> dict[str, Any]:
    margins = []
    win_rates = []
    same_author_similarities = []
    source_pool_similarities = []
    author_reports = {}
    heldout_pool = [
        (author, row)
        for author, (_, heldout_rows) in splits.items()
        for row in heldout_rows
    ]
    for author, (train_rows, heldout_rows) in splits.items():
        train_mean = average_features(train_rows)
        author_margins = []
        author_win_rates = []
        for heldout in heldout_rows:
            same = style_similarity(train_mean, heldout["style_features"])
            impostor_similarities = [
                similarity
                for other_author, candidate in heldout_pool
                if other_author != author
                and (similarity := style_similarity(train_mean, candidate["style_features"]))
                is not None
            ]
            if same is None or not impostor_similarities:
                continue
            pool_mean = sum(impostor_similarities) / len(impostor_similarities)
            margin = round(same - pool_mean, 3)
            win_rate = round(
                sum(1 for value in impostor_similarities if same > value)
                / len(impostor_similarities),
                3,
            )
            same_author_similarities.append(same)
            source_pool_similarities.extend(impostor_similarities)
            margins.append(margin)
            win_rates.append(win_rate)
            author_margins.append(margin)
            author_win_rates.append(win_rate)
        author_reports[author] = {
            "train_fragments": len(train_rows),
            "heldout_fragments": len(heldout_rows),
            "mean_margin_vs_source_pool": mean(author_margins),
            "mean_pairwise_win_rate": mean(author_win_rates),
        }
    return {
        "candidate_author_count": len(splits),
        "mean_same_author_style_similarity": mean(same_author_similarities),
        "mean_source_pool_impostor_similarity": mean(source_pool_similarities),
        "mean_margin_vs_source_pool": mean(margins),
        "mean_pairwise_win_rate": mean(win_rates),
        "margin_stats": stats(margins),
        "pairwise_win_rate_stats": stats(win_rates),
        "author_reports": author_reports,
    }


def dominant_rate(values: list[str]) -> float | None:
    if not values:
        return None
    return round(max(Counter(values).values()) / len(values), 3)


def diagnose_coding_style_source(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
    source_kind: str = "personal_developer_fragments",
    pre_ai_cutoff: str = "2023-06-01",
    min_train: int = 20,
    min_heldout: int = 10,
    min_lines: int = 5,
    max_lines: int = 240,
) -> dict[str, Any]:
    clean, rejected = clean_fragments(
        rows,
        pre_ai_cutoff=pre_ai_cutoff,
        min_lines=min_lines,
        max_lines=max_lines,
    )
    repos = [str(row.get("repo") or "") for row in clean if row.get("repo")]
    languages = [str(row.get("language") or "") for row in clean if row.get("language")]
    caveats = []
    if source_kind == "project_style_guide":
        caveats.append("project_style_not_personal_author_style")
    if dominant_rate(repos) == 1.0 and len(set(repos)) == 1:
        caveats.append("single_project_style_may_dominate")
    if dominant_rate(languages) == 1.0 and len(set(languages)) == 1:
        caveats.append("single_language_style_may_dominate")

    splits = temporal_author_splits(clean, min_train=min_train, min_heldout=min_heldout)
    source_pool = source_pool_metrics(splits)
    if source_pool["candidate_author_count"] == 0:
        caveats.append("no_author_with_enough_temporal_fragments")
    elif source_pool["candidate_author_count"] < 3:
        caveats.append("too_few_authors_for_source_pool_baseline")
    margin = numeric(source_pool.get("mean_margin_vs_source_pool"))
    win_rate = numeric(source_pool.get("mean_pairwise_win_rate"))
    if margin is not None and margin <= 0:
        caveats.append("nonpositive_margin_vs_source_pool")
    elif margin is not None and margin < 0.03:
        caveats.append("weak_margin_vs_source_pool")
    if win_rate is not None and win_rate < 0.6:
        caveats.append("low_pairwise_win_rate_vs_source_pool")

    if source_kind == "project_style_guide":
        decision = "diagnostic_only"
        recommended_role = "project_style_control"
    elif (
        source_pool["candidate_author_count"] >= 3
        and margin is not None
        and margin >= 0.03
        and win_rate is not None
        and win_rate >= 0.6
    ):
        decision = "go"
        recommended_role = "personal_developer_style_candidate"
    elif clean:
        decision = "diagnostic_only"
        recommended_role = (
            "project_style_control"
            if "single_project_style_may_dominate" in caveats
            else "needs_more_data"
        )
    else:
        decision = "drop"
        recommended_role = "drop"

    return {
        "schema_version": SCHEMA_VERSION,
        "source_name": source_name,
        "source_kind": source_kind,
        "decision": decision,
        "recommended_role": recommended_role,
        "pre_ai_cutoff": pre_ai_cutoff,
        "input_rows": len(rows),
        "clean_fragments": len(clean),
        "rejected_fragment_reasons": dict(rejected),
        "author_count": len({str(row.get("author_id")) for row in clean}),
        "candidate_author_count": source_pool["candidate_author_count"],
        "repo_counts": dict(Counter(repos).most_common(10)),
        "language_counts": dict(Counter(languages).most_common(10)),
        "source_pool_baseline": source_pool,
        "caveats": sorted(set(caveats)),
        "minimums": {
            "min_train": min_train,
            "min_heldout": min_heldout,
            "min_lines": min_lines,
            "max_lines": max_lines,
            "min_candidate_authors_for_go": 3,
            "min_margin_vs_source_pool_for_go": 0.03,
            "min_pairwise_win_rate_for_go": 0.6,
        },
    }
