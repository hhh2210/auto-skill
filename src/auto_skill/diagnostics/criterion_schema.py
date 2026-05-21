"""Shared dataclasses for per-criterion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """A paired task dropped because the two modes scored different criteria."""

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
