"""Compatibility facade - moved to auto_skill.cleaning.author_style.audit."""

from auto_skill.cleaning.author_style.audit import *  # noqa: F401,F403
from auto_skill.cleaning.author_style.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
