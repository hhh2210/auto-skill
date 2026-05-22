"""Shared baseline protocol and result containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from auto_skill.schemas import UserExample


@dataclass(frozen=True)
class BaselineSpec:
    """Metadata for one CLI/eval baseline mode."""

    name: str
    requires_skill: bool
    requires_examples: bool
    benchmark_scope: tuple[str, ...] = ("writingbench", "presentbench")
    description: str = ""
    skill_lookup: str | None = None
    needs_layout_plan: bool = False
    needs_evidence_plan: bool = False
    induction_outputs: tuple[str, ...] = ()


@runtime_checkable
class Baseline(Protocol):
    """Common dispatch surface for heldout-generation baseline prompts."""

    spec: BaselineSpec

    def build_heldout_prompt(
        self,
        *,
        task: dict[str, Any],
        examples: list[UserExample],
        skill_md: str | None = None,
        max_material_chars: int = 4000,
    ) -> str:
        """Build the user-visible heldout prompt for this baseline mode."""


@dataclass(frozen=True)
class PromptRunResult:
    """Result from one model call used by the MVP scripts."""

    text: str
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    duration_seconds: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillInductionResult:
    """A generated skill package plus the evidence used to build it."""

    pack_id: str
    mode: str
    skill_md: str
    feature_reports: list[dict[str, Any]] = field(default_factory=list)
    cross_example_report: dict[str, Any] | None = None
    leave_one_out: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = "skill-induction/v1"
        row["status"] = "success"
        row["solver_model"] = _first_call_model(self.model_calls)
        return row


def _first_call_model(model_calls: list[dict[str, Any]]) -> str | None:
    """Return the first non-empty ``model`` string from a model_calls list."""

    for call in model_calls:
        if not isinstance(call, dict):
            continue
        model = call.get("model")
        if isinstance(model, str) and model.strip():
            return model
    return None
