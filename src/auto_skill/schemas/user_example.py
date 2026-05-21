"""User-facing example and skill package schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UserExample:
    """A user-visible example that can be used for skill induction."""

    example_id: str
    task_input: str
    output: str | None = None
    user_notes: str | None = None
    materials: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must be non-empty")
        if not self.task_input.strip():
            raise ValueError("task_input must be non-empty")


@dataclass(frozen=True)
class SkillPackage:
    """A generated skill package before it is written to disk."""

    name: str
    skill_md: str
    metadata: dict[str, Any] = field(default_factory=dict)
    templates: dict[str, str] = field(default_factory=dict)
    tests: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.skill_md.strip():
            raise ValueError("skill_md must be non-empty")
