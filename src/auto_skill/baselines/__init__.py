"""Baseline registry for heldout prompt dispatch and skill-row metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auto_skill.baselines._protocol import Baseline, BaselineSpec
from auto_skill.baselines._shared import build_heldout_generation_prompt
from auto_skill.schemas import UserExample

ONE_SHOT_SKILL_MODE = "one_shot_skill_from_examples"
FEATURE_SKILL_MODE = "auto_skill_feature_driven_no_validation"
FULL_SKILL_MODE = "auto_skill_ours_full"
FEATURE_SIGNATURE_MODE = f"{FEATURE_SKILL_MODE}::feature_signatures"
OPERATIONAL_ANCHOR_MODE = f"{FEATURE_SKILL_MODE}::operational_anchors"


@dataclass(frozen=True)
class PromptBaseline:
    """Registry adapter for modes that share the canonical heldout prompt builder."""

    spec: BaselineSpec

    def build_heldout_prompt(
        self,
        *,
        task: dict[str, Any],
        examples: list[UserExample],
        skill_md: str | None = None,
        max_material_chars: int = 4000,
    ) -> str:
        return build_heldout_generation_prompt(
            task=task,
            mode=self.spec.name,
            examples=examples,
            skill_md=skill_md,
            max_material_chars=max_material_chars,
        )

    def skill_from_index(self, skills: dict[tuple[str, str], str], pack_id: str) -> str | None:
        if not self.spec.skill_lookup:
            return None
        return skills.get((pack_id, self.spec.skill_lookup))


def _baseline(
    name: str,
    *,
    requires_skill: bool = False,
    requires_examples: bool = False,
    skill_lookup: str | None = None,
    needs_layout_plan: bool = False,
    needs_evidence_plan: bool = False,
    induction_outputs: tuple[str, ...] = (),
    benchmark_scope: tuple[str, ...] = ("writingbench", "presentbench"),
    description: str = "",
) -> PromptBaseline:
    return PromptBaseline(
        BaselineSpec(
            name=name,
            requires_skill=requires_skill,
            requires_examples=requires_examples,
            benchmark_scope=benchmark_scope,
            description=description,
            skill_lookup=skill_lookup,
            needs_layout_plan=needs_layout_plan,
            needs_evidence_plan=needs_evidence_plan,
            induction_outputs=induction_outputs,
        )
    )


_BASELINES: tuple[PromptBaseline, ...] = (
    _baseline("prompt_only", description="Heldout task without examples or skill."),
    _baseline(
        "few_shot_examples_only",
        requires_examples=True,
        description="Heldout task with train examples only.",
    ),
    _baseline(
        "few_shot_examples_bundle_preserve",
        requires_examples=True,
        description="Author-style heldout task preserving example bundle shape.",
    ),
    _baseline(
        ONE_SHOT_SKILL_MODE,
        requires_skill=True,
        skill_lookup=ONE_SHOT_SKILL_MODE,
        induction_outputs=(ONE_SHOT_SKILL_MODE,),
        description="One-shot SKILL.md induced from examples.",
    ),
    _baseline(
        "examples_plus_one_shot_skill",
        requires_skill=True,
        requires_examples=True,
        skill_lookup=ONE_SHOT_SKILL_MODE,
        description="Examples plus the one-shot SKILL.md.",
    ),
    _baseline(
        FEATURE_SKILL_MODE,
        induction_outputs=(FEATURE_SKILL_MODE,),
        description="Feature-driven skill row produced before validation merge.",
    ),
    _baseline(
        "ours_no_validation",
        requires_skill=True,
        skill_lookup=FEATURE_SKILL_MODE,
        description="Feature-driven skill without leave-one-out validation.",
    ),
    _baseline(
        "examples_plus_feature_skill",
        requires_skill=True,
        requires_examples=True,
        skill_lookup=FEATURE_SKILL_MODE,
        description="Examples plus feature-driven skill.",
    ),
    _baseline(
        "examples_plus_operational_skill",
        requires_examples=True,
        description="Legacy examples-plus-operational mode without a registered skill row.",
    ),
    _baseline(
        "slide_constrained_examples_plus_feature_skill",
        requires_skill=True,
        requires_examples=True,
        skill_lookup=FEATURE_SKILL_MODE,
        description="PresentBench examples plus feature skill with slide constraints.",
    ),
    _baseline(
        "layout_plan_examples_plus_feature_skill",
        requires_skill=True,
        requires_examples=True,
        skill_lookup=FEATURE_SKILL_MODE,
        needs_layout_plan=True,
        benchmark_scope=("presentbench",),
        description="PresentBench layout-plan variant before final generation.",
    ),
    _baseline(
        "feature_signatures_only",
        requires_skill=True,
        skill_lookup=FEATURE_SIGNATURE_MODE,
        description="Feature-signature context without examples.",
    ),
    _baseline(
        "examples_plus_feature_signatures",
        requires_skill=True,
        requires_examples=True,
        skill_lookup=FEATURE_SIGNATURE_MODE,
        description="Examples plus feature-signature context.",
    ),
    _baseline(
        "task_first_feature_signatures",
        requires_skill=True,
        skill_lookup=FEATURE_SIGNATURE_MODE,
        description="Task-first generation with feature signatures.",
    ),
    _baseline(
        "task_first_operational_anchors",
        requires_skill=True,
        skill_lookup=OPERATIONAL_ANCHOR_MODE,
        description="Task-first generation with operational anchors.",
    ),
    _baseline(
        "task_first_evidence_anchored_operational_anchors",
        requires_skill=True,
        skill_lookup=OPERATIONAL_ANCHOR_MODE,
        description="Task-first generation with evidence-anchored operational anchors.",
    ),
    _baseline(
        "task_first_two_level_operational_anchors",
        requires_skill=True,
        skill_lookup=OPERATIONAL_ANCHOR_MODE,
        description="Task-first generation with two-level operational anchors.",
    ),
    _baseline(
        "task_first_planned_operational_anchors",
        requires_skill=True,
        skill_lookup=OPERATIONAL_ANCHOR_MODE,
        needs_evidence_plan=True,
        benchmark_scope=("writingbench",),
        description="Two-stage planned generation with operational anchors.",
    ),
    _baseline(
        FULL_SKILL_MODE,
        induction_outputs=(FEATURE_SKILL_MODE, FULL_SKILL_MODE),
        description="Validation-aware full skill row.",
    ),
    _baseline(
        "auto_skill",
        requires_skill=True,
        skill_lookup=FULL_SKILL_MODE,
        description="Heldout eval using the validation-aware full skill.",
    ),
)

REGISTRY: dict[str, Baseline] = {baseline.spec.name: baseline for baseline in _BASELINES}

INDUCTION_REQUEST_ALIASES: dict[str, tuple[str, ...]] = {
    ONE_SHOT_SKILL_MODE: (ONE_SHOT_SKILL_MODE,),
    "auto_skill_feature_driven": (FEATURE_SKILL_MODE, FULL_SKILL_MODE),
}


def baseline_for_mode(mode: str) -> Baseline:
    """Return the registered baseline, falling back to legacy prompt behavior."""

    baseline = REGISTRY.get(mode)
    if baseline is not None:
        return baseline
    return _baseline(mode, description="Unregistered legacy mode fallback.")


def mode_requires_skill(mode: str) -> bool:
    return baseline_for_mode(mode).spec.requires_skill


def skill_for_mode(
    mode: str,
    skills: dict[tuple[str, str], str],
    pack_id: str,
) -> str | None:
    baseline = baseline_for_mode(mode)
    if isinstance(baseline, PromptBaseline):
        return baseline.skill_from_index(skills, pack_id)
    skill_lookup = baseline.spec.skill_lookup
    if skill_lookup is None:
        return None
    return skills.get((pack_id, skill_lookup))


def requested_induction_outputs(
    requested_modes: set[str],
    *,
    include_full: bool,
) -> set[str]:
    """Expand CLI induction mode aliases into concrete skill-row modes."""

    output_modes: set[str] = set()
    for mode in requested_modes:
        for output_mode in INDUCTION_REQUEST_ALIASES.get(mode, (mode,)):
            if output_mode == FULL_SKILL_MODE and not include_full:
                continue
            if output_mode in REGISTRY and REGISTRY[output_mode].spec.induction_outputs:
                output_modes.add(output_mode)
    return output_modes


__all__ = [
    "Baseline",
    "BaselineSpec",
    "FEATURE_SIGNATURE_MODE",
    "FEATURE_SKILL_MODE",
    "FULL_SKILL_MODE",
    "INDUCTION_REQUEST_ALIASES",
    "ONE_SHOT_SKILL_MODE",
    "OPERATIONAL_ANCHOR_MODE",
    "REGISTRY",
    "baseline_for_mode",
    "mode_requires_skill",
    "requested_induction_outputs",
    "skill_for_mode",
]
