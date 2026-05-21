"""Compatibility facade - moved to auto_skill.baselines.feature_driven."""

from auto_skill.baselines import feature_driven as _impl

_EXPORTS = (
    "Literal",
    "RuleStatus",
    "RuleSupport",
    "UserExample",
    "annotations",
    "build_cross_example_analysis_prompt",
    "build_feature_extraction_prompt",
    "build_skill_compilation_prompt",
    "classify_rule_support",
    "dataclass",
    "merge_rule_supports",
)

globals().update({name: getattr(_impl, name) for name in _EXPORTS})

__all__ = [name for name in _EXPORTS if not name.startswith("_")]
