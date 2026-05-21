"""Compatibility facade for MVP helpers moved into baselines/ and prompts/."""

from auto_skill.baselines import _protocol as _protocol_impl
from auto_skill.baselines import _shared as _shared_impl
from auto_skill.baselines import feature_driven_with_validation as _validation_impl
from auto_skill.baselines import layout_plan as _layout_plan_impl
from auto_skill.baselines import one_shot_skill_from_examples as _one_shot_impl
from auto_skill.baselines.feature_driven import (
    build_task_first_feature_signature_prompt as build_task_first_feature_signature_prompt,
)
from auto_skill.llm.parse import parse_json_object as parse_json_object
from auto_skill.prompts import _format as _format_impl
from auto_skill.prompts import evidence as _evidence_impl
from auto_skill.prompts import judge as _judge_impl
from auto_skill.prompts import operational_anchor as _operational_anchor_impl

for _impl in (
    _protocol_impl,
    _shared_impl,
    _one_shot_impl,
    _validation_impl,
    _evidence_impl,
    _operational_anchor_impl,
    _format_impl,
    _layout_plan_impl,
    _judge_impl,
):
    globals().update(
        {
            name: getattr(_impl, name)
            for name in dir(_impl)
            if not name.startswith("__")
        }
    )

for _name in (
    "_impl",
    "_protocol_impl",
    "_shared_impl",
    "_one_shot_impl",
    "_validation_impl",
    "_evidence_impl",
    "_operational_anchor_impl",
    "_format_impl",
    "_layout_plan_impl",
    "_judge_impl",
):
    globals().pop(_name, None)

__all__ = [name for name in globals() if not name.startswith("_")]
