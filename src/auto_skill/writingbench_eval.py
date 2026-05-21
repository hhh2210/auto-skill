"""Compatibility facade - moved to auto_skill.eval.writingbench."""

from auto_skill.eval import writingbench as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("__")
    }
)

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
