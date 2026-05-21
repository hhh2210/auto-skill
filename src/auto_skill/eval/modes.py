"""Shared evaluation and reporting mode constants."""

FULL_SKILL_MODES = (
    "one_shot_skill_from_examples",
    "auto_skill_feature_driven_no_validation",
    "auto_skill_ours_full",
)

MVP_SKILL_MODES = (
    "one_shot_skill_from_examples",
    "auto_skill_feature_driven_no_validation",
)

FULL_EVAL_MODES = (
    "prompt_only",
    "few_shot_examples_only",
    "one_shot_skill_from_examples",
    "ours_no_validation",
    "auto_skill",
)

MVP_EVAL_MODES = (
    "prompt_only",
    "few_shot_examples_only",
    "one_shot_skill_from_examples",
    "ours_no_validation",
)

PRESENTBENCH_OFFICIAL_MODES = ("prompt_only", "auto_skill")

SKILL_REQUIRED_MODES = {
    "one_shot_skill_from_examples",
    "ours_no_validation",
    "auto_skill",
    "examples_plus_one_shot_skill",
    "examples_plus_feature_skill",
    "feature_signatures_only",
    "examples_plus_feature_signatures",
    "task_first_feature_signatures",
    "task_first_operational_anchors",
    "task_first_evidence_anchored_operational_anchors",
    "task_first_two_level_operational_anchors",
    "task_first_planned_operational_anchors",
    "slide_constrained_examples_plus_feature_skill",
    "layout_plan_examples_plus_feature_skill",
}

DEFAULT_BASELINE_MODE = "prompt_only"
DEFAULT_ARTIFACT_COMPARE_MODES = (
    "one_shot_skill_from_examples",
    "auto_skill_feature_driven_no_validation",
)
