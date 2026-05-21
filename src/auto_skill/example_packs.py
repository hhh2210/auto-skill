"""Compatibility facade - moved to auto_skill.cleaning.packs."""

from auto_skill.cleaning import packs as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("__")
    }
)

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
