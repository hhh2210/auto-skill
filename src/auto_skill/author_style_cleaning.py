"""Compatibility facade for author-style cleaning helpers."""

from auto_skill.cleaning.author_style.common import *  # noqa: F403
from auto_skill.cleaning.author_style.gpt import *  # noqa: F403
from auto_skill.cleaning.author_style.packs import *  # noqa: F403
from auto_skill.cleaning.author_style.pipeline import *  # noqa: F403
from auto_skill.cleaning.author_style.pipeline import main
from auto_skill.cleaning.author_style.sources import *  # noqa: F403
from auto_skill.llm.parse import parse_json_object  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
