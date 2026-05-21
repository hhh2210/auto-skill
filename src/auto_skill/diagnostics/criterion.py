"""Per-criterion diagnostic helpers for WritingBench-style evaluator rows.

WritingBench tasks use 5 task-specific checklist criteria. The task-level
``overall_score`` is the mean of those 5 criterion scores, but the raw
per-criterion data is kept in ``row["scores"]`` as a
``dict[criterion_name, list[{"score", "reason"}]]``. This module exposes
helpers that turn that raw structure into per-task / per-criterion deltas
between two modes (e.g. ``ours_no_validation`` vs
``few_shot_examples_only``) so we can ask "in the cells where the candidate
mode loses, *which criterion* does it lose on" instead of only seeing the
collapsed task average.

The module deliberately avoids any LLM calls. It is a pure post-processor
over already-evaluated rows.

Important caveats baked into the API:
  - Tasks where the two modes do not share the *same* criterion set are
    not silently intersected. They are skipped and surfaced in
    ``PairingResult.skipped`` so the caller can audit why those tasks were
    dropped.
  - The keyword bucket ontology (``classify_criterion``) is a heuristic
    label, not a taxonomy. It uses substring keyword matching, returns one
    *primary* label for back-compat, and exposes ``classify_criterion_buckets``
    for the *full* multi-label match list. Diagnosis output always includes
    the matched keyword(s) so the reader can audit any specific row.
  - Bucket-level aggregates split worst-pick / best-pick counts into
    ``_all`` (every paired task) and ``_on_losing_tasks`` /
    ``_on_winning_tasks`` (only tasks where the candidate mode actually
    lost / won by overall delta). Research conclusions about systemic
    failure modes should cite the on-losing-tasks counts; the all-tasks
    counts are a sanity check.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

CriterionName = str
TaskKey = tuple[str, str]


@dataclass(frozen=True)
class CriterionScore:
    criterion: CriterionName
    score: float
    reason: str = ""


@dataclass(frozen=True)
class CriterionDelta:
    criterion: CriterionName
    baseline_score: float
    mode_score: float
    delta: float


@dataclass(frozen=True)
class TaskCriterionDeltas:
    pack_id: str
    task_id: str
    baseline_overall: float
    mode_overall: float
    overall_delta: float
    criteria: tuple[CriterionDelta, ...]

    @property
    def worst_criterion(self) -> CriterionDelta | None:
        if not self.criteria:
            return None
        return min(self.criteria, key=lambda item: item.delta)

    @property
    def best_criterion(self) -> CriterionDelta | None:
        if not self.criteria:
            return None
        return max(self.criteria, key=lambda item: item.delta)


@dataclass(frozen=True)
class SkippedTask:
    """A paired task that was dropped because the two modes scored
    different criterion sets. ``baseline_only`` and ``mode_only`` carry
    the asymmetric difference so the caller can investigate why.
    """

    pack_id: str
    task_id: str
    baseline_only: tuple[str, ...]
    mode_only: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateCell:
    """Duplicate successful rows for one ``(mode, pack_id, task_id)`` cell."""

    mode: str
    pack_id: str
    task_id: str
    count: int
    evaluator_kinds: tuple[str, ...]
    judge_models: tuple[str, ...]


@dataclass(frozen=True)
class PairingResult:
    deltas: tuple[TaskCriterionDeltas, ...]
    skipped: tuple[SkippedTask, ...] = field(default_factory=tuple)
    duplicate_cells: tuple[DuplicateCell, ...] = field(default_factory=tuple)


# Keyword buckets for the systemic-failure roll-up. Multilingual (zh+en)
# because WritingBench mixes both. Matching is case-insensitive substring on
# the normalized name. Order controls the *primary* (single-label) bucket
# returned by ``classify_criterion``; ``classify_criterion_buckets`` returns
# every match so callers can see the full label set.
#
# Ordering principle: more concrete dimensions before broader evaluative
# umbrellas. Specifically ``depth_specificity_practical`` precedes
# ``accuracy_professionalism`` so that names like
# "Content_Depth_and_Accuracy" or "Financial Accuracy and Specificity" are
# attributed to the more discriminative depth axis. Pure accuracy names
# ("Factual Accuracy", "Academic Rigor") do not contain depth keywords and
# stay in the accuracy bucket.
_KEYWORD_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "length_word_count",
        (
            "字数",
            "篇幅",
            "长度",
            "word count",
            "word_count",
            "length",
            "wordcount",
            "字符",
            "character count",
        ),
    ),
    (
        "format_structure",
        (
            "格式",
            "结构",
            "层次",
            "组织",
            "排版",
            "章节",
            "section",
            "structure",
            "format",
            "organization",
            "layout",
            "outline",
            "hierarchy",
        ),
    ),
    (
        "required_sections_completeness",
        (
            "必填",
            "必备",
            "完整",
            "齐全",
            "覆盖",
            "完备",
            "缺失",
            "全面",
            "completeness",
            "complete",
            "coverage",
            "required",
            "comprehensive",
            "essential",
            "all sections",
        ),
    ),
    (
        "evidence_citation",
        (
            "证据",
            "引用",
            "数据",
            "参考文献",
            "来源",
            "依据",
            "事实",
            "citation",
            "evidence",
            "source",
            "reference",
            "supporting",
        ),
    ),
    (
        "style_tone_voice",
        (
            "风格",
            "语气",
            "口吻",
            "笔调",
            "tone",
            "style",
            "voice",
            "register",
            "persona",
            "narrator",
        ),
    ),
    (
        "language_terminology",
        (
            "语言",
            "用词",
            "措辞",
            "术语",
            "表达",
            "language",
            "terminology",
            "wording",
            "phrasing",
            "diction",
        ),
    ),
    (
        "novelty_creativity",
        (
            "创新",
            "原创",
            "新颖",
            "独到",
            "创意",
            "novelty",
            "creativity",
            "originality",
            "innovative",
            "innovation",
        ),
    ),
    (
        "audience_persuasion",
        (
            "受众",
            "读者",
            "说服",
            "感染",
            "吸引",
            "audience",
            "persuasion",
            "engagement",
            "appeal",
            "reader",
        ),
    ),
    (
        "task_alignment",
        (
            "符合",
            "贴合",
            "对齐",
            "目标",
            "需求",
            "alignment",
            "relevance",
            "task",
            "objective",
        ),
    ),
    (
        "depth_specificity_practical",
        (
            "深度",
            "深入",
            "细节",
            "实用",
            "操作",
            "案例",
            "实证",
            "应用",
            "分析",
            "论述",
            "整合",
            "针对性",
            "具体",
            "depth",
            "deep",
            "detail",
            "practical",
            "utility",
            "case",
            "empirical",
            "application",
            "analytical",
            "analysis",
            "discussion",
            "integration",
            "specificity",
            "specific",
            "concrete",
            "implementation",
        ),
    ),
    (
        "accuracy_professionalism",
        (
            "准确",
            "专业",
            "严谨",
            "正确",
            "学术规范",
            "accuracy",
            "professional",
            "rigor",
            "correctness",
            "precise",
        ),
    ),
)


def _match_buckets(name: str) -> list[tuple[str, str]]:
    """Return all ``(bucket, matched_keyword)`` pairs that fire on ``name``.

    Multiple buckets may match. The order of returned pairs follows
    ``_KEYWORD_BUCKETS`` declaration order; for each bucket the first
    matching keyword is recorded.
    """

    normalized = (name or "").strip().lower()
    if not normalized:
        return []
    matches: list[tuple[str, str]] = []
    for bucket_name, keywords in _KEYWORD_BUCKETS:
        for kw in keywords:
            if kw in normalized:
                matches.append((bucket_name, kw))
                break
    return matches


def classify_criterion(name: str) -> str:
    """Map a free-form criterion name to a *primary* coarse bucket.

    Returns ``"other"`` when no bucket keyword matches. The buckets are a
    heuristic diagnostic aid only; they intentionally collapse many
    distinct task-specific criteria into a small ontology so we can spot
    whether the candidate mode is dropping a specific axis (e.g. required
    sections).
    """

    matches = _match_buckets(name)
    return matches[0][0] if matches else "other"


def classify_criterion_buckets(name: str) -> list[str]:
    """Return *all* matching buckets for ``name`` (multi-label).

    The first bucket in the returned list matches what
    ``classify_criterion`` returns. An empty list is returned when nothing
    matches; the consuming code can substitute ``["other"]`` if it wants
    multi-label counts to remain non-empty.
    """

    return [bucket for bucket, _ in _match_buckets(name)]


def matched_keywords_for(name: str) -> list[tuple[str, str]]:
    """Return the ``(bucket, matched_keyword)`` pairs for ``name``.

    Public alias of ``_match_buckets`` so the CLI / report can show readers
    why a given criterion landed in a particular bucket.
    """

    return _match_buckets(name)


def extract_criterion_scores(row: dict[str, Any]) -> tuple[CriterionScore, ...]:
    """Extract per-criterion scores from one eval row.

    Returns an empty tuple when the row has no ``scores`` payload, when no
    criterion has a numeric score, or when ``status != "success"``. The first
    parsed entry per criterion is used (matches the WritingBench evaluator's
    convention of one judge run per criterion).
    """

    if str(row.get("status") or "") != "success":
        return ()
    raw_scores = row.get("scores")
    if not isinstance(raw_scores, dict):
        return ()
    items: list[CriterionScore] = []
    for name, entries in raw_scores.items():
        if not isinstance(entries, list) or not entries:
            continue
        first = entries[0]
        if not isinstance(first, dict):
            continue
        score = first.get("score")
        if not isinstance(score, (int, float)):
            continue
        reason = first.get("reason") if isinstance(first.get("reason"), str) else ""
        items.append(CriterionScore(str(name), float(score), reason))
    return tuple(items)


def _row_task_key(row: dict[str, Any]) -> TaskKey:
    return (str(row.get("pack_id") or ""), str(row.get("task_id") or ""))


def per_task_criterion_deltas(
    rows: Iterable[dict[str, Any]],
    *,
    mode: str,
    baseline_mode: str,
) -> PairingResult:
    """Compute per-criterion deltas for ``mode`` vs ``baseline_mode``.

    Tasks are paired only when both modes report ``status=success`` *and*
    the two rows scored exactly the same set of criterion names. Tasks
    where the criterion sets differ are recorded in ``PairingResult.skipped``
    rather than silently intersected, because intersecting would produce a
    per-criterion delta over a different denominator than the row-level
    overall mean and bias the diagnosis.

    ``PairingResult.deltas`` is sorted by ``overall_delta`` ascending so
    the worst regressions show up first.
    """

    rows_by_mode_task: dict[tuple[str, TaskKey], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_mode = str(row.get("mode") or "")
        if row_mode not in (mode, baseline_mode):
            continue
        if str(row.get("status") or "") != "success":
            continue
        rows_by_mode_task[(row_mode, _row_task_key(row))].append(row)

    duplicate_cells = []
    by_mode_task: dict[tuple[str, TaskKey], dict[str, Any]] = {}
    for (row_mode, task_key), cell_rows in rows_by_mode_task.items():
        if len(cell_rows) == 1:
            by_mode_task[(row_mode, task_key)] = cell_rows[0]
            continue
        duplicate_cells.append(
            DuplicateCell(
                mode=row_mode,
                pack_id=task_key[0],
                task_id=task_key[1],
                count=len(cell_rows),
                evaluator_kinds=tuple(
                    sorted({str(row.get("evaluator_kind") or "") for row in cell_rows})
                ),
                judge_models=tuple(
                    sorted({str(row.get("judge_model") or "") for row in cell_rows})
                ),
            )
        )

    task_keys = sorted(
        {key for (m, key) in by_mode_task if m == mode}
        & {key for (m, key) in by_mode_task if m == baseline_mode}
    )

    deltas_list: list[TaskCriterionDeltas] = []
    skipped_list: list[SkippedTask] = []
    for task_key in task_keys:
        baseline_row = by_mode_task[(baseline_mode, task_key)]
        mode_row = by_mode_task[(mode, task_key)]
        baseline_scores = {c.criterion: c for c in extract_criterion_scores(baseline_row)}
        mode_scores = {c.criterion: c for c in extract_criterion_scores(mode_row)}
        baseline_set = set(baseline_scores)
        mode_set = set(mode_scores)
        if not baseline_set or not mode_set:
            continue
        if baseline_set != mode_set:
            skipped_list.append(
                SkippedTask(
                    pack_id=task_key[0],
                    task_id=task_key[1],
                    baseline_only=tuple(sorted(baseline_set - mode_set)),
                    mode_only=tuple(sorted(mode_set - baseline_set)),
                )
            )
            continue
        names = sorted(baseline_set)
        deltas = tuple(
            CriterionDelta(
                criterion=name,
                baseline_score=baseline_scores[name].score,
                mode_score=mode_scores[name].score,
                delta=mode_scores[name].score - baseline_scores[name].score,
            )
            for name in names
        )
        baseline_overall = mean(c.score for c in baseline_scores.values())
        mode_overall = mean(c.score for c in mode_scores.values())
        deltas_list.append(
            TaskCriterionDeltas(
                pack_id=task_key[0],
                task_id=task_key[1],
                baseline_overall=baseline_overall,
                mode_overall=mode_overall,
                overall_delta=mode_overall - baseline_overall,
                criteria=deltas,
            )
        )
    deltas_list.sort(key=lambda item: item.overall_delta)
    return PairingResult(
        deltas=tuple(deltas_list),
        skipped=tuple(skipped_list),
        duplicate_cells=tuple(
            sorted(
                duplicate_cells,
                key=lambda item: (item.mode, item.pack_id, item.task_id),
            )
        ),
    )


def aggregate_bucket_diagnosis(
    task_deltas: Sequence[TaskCriterionDeltas],
    *,
    loss_threshold: float = -0.5,
    win_threshold: float = 0.5,
) -> dict[str, Any]:
    """Roll per-task deltas up into systemic-failure diagnostics.

    Buckets come from ``classify_criterion`` (single-label primary). For
    each bucket we report:
      - ``appearances``: number of criterion-instances assigned to this
        bucket as their primary label (sum across buckets equals total
        criterion instances).
      - ``mean_delta``: average delta across those instances.
      - ``loss_count`` / ``win_count`` / ``tie_count``: per-instance bucket
        counts using ``loss_threshold`` / ``win_threshold``.
      - ``worst_pick_count_all``: number of tasks (across the full paired
        set) whose single worst criterion was primary-labelled this bucket.
      - ``worst_pick_count_on_losing_tasks``: same restricted to tasks
        where ``overall_delta < 0`` — this is the column research claims
        about systemic failure modes should cite.
      - ``worst_pick_count_multilabel``: same as ``worst_pick_count_all``
        but counts every bucket the worst criterion matched (a single
        worst criterion can contribute to multiple bucket counters).
      - ``worst_pick_count_on_losing_tasks_multilabel``: multi-label
        variant restricted to losing tasks.
      - ``best_pick_count_all`` / ``best_pick_count_on_winning_tasks``:
        symmetric ``best_criterion`` counters.

    Buckets are heuristic substring-keyword groupings; ``mean_delta`` and
    instance counts use the *primary* label (first match in
    ``_KEYWORD_BUCKETS``), while the ``_multilabel`` columns reflect every
    matched bucket.
    """

    bucket_to_deltas: dict[str, list[float]] = defaultdict(list)
    bucket_loss_counts: Counter[str] = Counter()
    bucket_win_counts: Counter[str] = Counter()
    bucket_tie_counts: Counter[str] = Counter()
    worst_pick_all_single: Counter[str] = Counter()
    worst_pick_losing_single: Counter[str] = Counter()
    worst_pick_all_multi: Counter[str] = Counter()
    worst_pick_losing_multi: Counter[str] = Counter()
    best_pick_all_single: Counter[str] = Counter()
    best_pick_winning_single: Counter[str] = Counter()
    best_pick_all_multi: Counter[str] = Counter()
    best_pick_winning_multi: Counter[str] = Counter()

    for task in task_deltas:
        for cd in task.criteria:
            primary = classify_criterion(cd.criterion)
            bucket_to_deltas[primary].append(cd.delta)
            if cd.delta <= loss_threshold:
                bucket_loss_counts[primary] += 1
            elif cd.delta >= win_threshold:
                bucket_win_counts[primary] += 1
            else:
                bucket_tie_counts[primary] += 1
        worst = task.worst_criterion
        if worst is not None:
            primary = classify_criterion(worst.criterion)
            worst_pick_all_single[primary] += 1
            if task.overall_delta < 0:
                worst_pick_losing_single[primary] += 1
            for bucket in classify_criterion_buckets(worst.criterion) or ["other"]:
                worst_pick_all_multi[bucket] += 1
                if task.overall_delta < 0:
                    worst_pick_losing_multi[bucket] += 1
        best = task.best_criterion
        if best is not None:
            primary = classify_criterion(best.criterion)
            best_pick_all_single[primary] += 1
            if task.overall_delta > 0:
                best_pick_winning_single[primary] += 1
            for bucket in classify_criterion_buckets(best.criterion) or ["other"]:
                best_pick_all_multi[bucket] += 1
                if task.overall_delta > 0:
                    best_pick_winning_multi[bucket] += 1

    # Make sure every bucket that appears in any counter is represented in
    # the output even if it has zero per-instance ``appearances``.
    all_buckets = (
        set(bucket_to_deltas)
        | set(worst_pick_all_single)
        | set(worst_pick_all_multi)
        | set(best_pick_all_single)
        | set(best_pick_all_multi)
    )
    buckets_summary: list[dict[str, Any]] = []
    for bucket in all_buckets:
        deltas = bucket_to_deltas.get(bucket, [])
        buckets_summary.append(
            {
                "bucket": bucket,
                "appearances": len(deltas),
                "mean_delta": mean(deltas) if deltas else None,
                "loss_count": bucket_loss_counts[bucket],
                "win_count": bucket_win_counts[bucket],
                "tie_count": bucket_tie_counts[bucket],
                "worst_pick_count_all": worst_pick_all_single[bucket],
                "worst_pick_count_on_losing_tasks": worst_pick_losing_single[bucket],
                "worst_pick_count_multilabel": worst_pick_all_multi[bucket],
                "worst_pick_count_on_losing_tasks_multilabel": worst_pick_losing_multi[bucket],
                "best_pick_count_all": best_pick_all_single[bucket],
                "best_pick_count_on_winning_tasks": best_pick_winning_single[bucket],
                "best_pick_count_multilabel": best_pick_all_multi[bucket],
                "best_pick_count_on_winning_tasks_multilabel": best_pick_winning_multi[bucket],
            }
        )
    buckets_summary.sort(
        key=lambda item: (
            -item["worst_pick_count_on_losing_tasks"],
            item["mean_delta"] if item["mean_delta"] is not None else 0.0,
            -item["appearances"],
        )
    )
    return {
        "loss_threshold": loss_threshold,
        "win_threshold": win_threshold,
        "primary_labels_are_first_match_heuristic": True,
        "buckets": buckets_summary,
    }


def per_criterion_aggregate(
    task_deltas: Sequence[TaskCriterionDeltas],
) -> list[dict[str, Any]]:
    """Aggregate per-criterion-name (no bucketing).

    Most criterion names are unique to a single task in WritingBench, so
    this is mostly useful when many packs share a benchmark family
    (e.g. all ``Paper Outline`` tasks reuse the same 5 criterion names).
    """

    by_name: dict[str, list[CriterionDelta]] = defaultdict(list)
    for task in task_deltas:
        for cd in task.criteria:
            by_name[cd.criterion].append(cd)
    out: list[dict[str, Any]] = []
    for name, deltas in by_name.items():
        out.append(
            {
                "criterion": name,
                "primary_bucket": classify_criterion(name),
                "matched_buckets": classify_criterion_buckets(name),
                "matched_keywords": matched_keywords_for(name),
                "appearances": len(deltas),
                "mean_delta": mean(d.delta for d in deltas),
                "min_delta": min(d.delta for d in deltas),
                "max_delta": max(d.delta for d in deltas),
                "loss_count": sum(1 for d in deltas if d.delta < 0),
                "win_count": sum(1 for d in deltas if d.delta > 0),
                "tie_count": sum(1 for d in deltas if d.delta == 0),
            }
        )
    out.sort(key=lambda item: (item["mean_delta"], -item["appearances"]))
    return out


def diagnose(
    rows: Iterable[dict[str, Any]],
    *,
    mode: str,
    baseline_mode: str,
    loss_threshold: float = -0.5,
    win_threshold: float = 0.5,
    top_n_regressions: int = 10,
    top_n_improvements: int = 5,
) -> dict[str, Any]:
    """End-to-end per-criterion diagnosis for ``mode`` vs ``baseline_mode``.

    The shape is JSON-serializable and is what the CLI dumps. Top regressions
    / improvements are kept short for human inspection; bucket aggregates
    cover the full sample.
    """

    pairing = per_task_criterion_deltas(rows, mode=mode, baseline_mode=baseline_mode)
    task_deltas = list(pairing.deltas)
    bucket_summary = aggregate_bucket_diagnosis(
        task_deltas, loss_threshold=loss_threshold, win_threshold=win_threshold
    )
    criterion_summary = per_criterion_aggregate(task_deltas)

    overall_deltas = [task.overall_delta for task in task_deltas]
    paired_overall = {
        "common_cells": len(task_deltas),
        "mean_overall_delta": mean(overall_deltas) if overall_deltas else None,
        "loss_count": sum(1 for d in overall_deltas if d < 0),
        "win_count": sum(1 for d in overall_deltas if d > 0),
        "tie_count": sum(1 for d in overall_deltas if d == 0),
        "min_overall_delta": min(overall_deltas) if overall_deltas else None,
        "max_overall_delta": max(overall_deltas) if overall_deltas else None,
    }

    skipped_payload = {
        "skipped_mismatched_criteria_count": len(pairing.skipped),
        "skipped_mismatched_criteria": [
            {
                "pack_id": s.pack_id,
                "task_id": s.task_id,
                "baseline_only": list(s.baseline_only),
                "mode_only": list(s.mode_only),
            }
            for s in pairing.skipped
        ],
        "duplicate_success_cells_count": len(pairing.duplicate_cells),
        "duplicate_success_cells": [
            {
                "mode": d.mode,
                "pack_id": d.pack_id,
                "task_id": d.task_id,
                "count": d.count,
                "evaluator_kinds": list(d.evaluator_kinds),
                "judge_models": list(d.judge_models),
            }
            for d in pairing.duplicate_cells
        ],
    }

    def _criterion_row(cd: CriterionDelta) -> dict[str, Any]:
        return {
            "criterion": cd.criterion,
            "primary_bucket": classify_criterion(cd.criterion),
            "matched_buckets": classify_criterion_buckets(cd.criterion),
            "matched_keywords": matched_keywords_for(cd.criterion),
            "baseline_score": cd.baseline_score,
            "mode_score": cd.mode_score,
            "delta": cd.delta,
        }

    def _pick_row(cd: CriterionDelta | None) -> dict[str, Any] | None:
        if cd is None:
            return None
        return {
            "criterion": cd.criterion,
            "primary_bucket": classify_criterion(cd.criterion),
            "matched_buckets": classify_criterion_buckets(cd.criterion),
            "matched_keywords": matched_keywords_for(cd.criterion),
            "delta": cd.delta,
        }

    def _task_to_row(task: TaskCriterionDeltas) -> dict[str, Any]:
        return {
            "pack_id": task.pack_id,
            "task_id": task.task_id,
            "baseline_overall": task.baseline_overall,
            "mode_overall": task.mode_overall,
            "overall_delta": task.overall_delta,
            "is_losing_task": task.overall_delta < 0,
            "is_winning_task": task.overall_delta > 0,
            "criteria": [_criterion_row(cd) for cd in task.criteria],
            "worst_criterion": _pick_row(task.worst_criterion),
            "best_criterion": _pick_row(task.best_criterion),
        }

    top_regressions = [_task_to_row(t) for t in task_deltas[:top_n_regressions]]
    top_improvements = [
        _task_to_row(t) for t in list(reversed(task_deltas))[:top_n_improvements]
    ]

    return {
        "mode": mode,
        "baseline_mode": baseline_mode,
        "paired_overall": paired_overall,
        "skipped": skipped_payload,
        "bucket_diagnosis": bucket_summary,
        "criterion_aggregate": criterion_summary,
        "top_regressions": top_regressions,
        "top_improvements": top_improvements,
        "bucket_disclaimer": (
            "Buckets are heuristic substring-keyword labels, not a taxonomy. "
            "primary_bucket is first-match for back-compat; matched_buckets is "
            "the full multi-label set; matched_keywords records the exact "
            "substring that fired each match. Cite worst_pick_count_on_losing_tasks "
            "(not worst_pick_count_all) for systemic-failure claims."
        ),
    }
