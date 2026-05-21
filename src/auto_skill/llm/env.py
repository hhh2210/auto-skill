"""Environment parsing helpers for OpenAI-compatible chat clients."""

from __future__ import annotations

import os


class ConfigError(ValueError):
    """Raised when local LLM environment configuration is invalid."""


def first_set_env(*names: str) -> str | None:
    """Return the first environment value among ``names`` that is set and non-empty."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def first_set_env_name(prefixes: tuple[str, ...], suffix: str) -> str | None:
    """Return the env var name (with prefix) that is the first non-empty match."""

    for prefix in prefixes:
        name = f"{prefix}_{suffix}"
        value = os.getenv(name)
        if value:
            return name
    return None


def parse_chain_float_env(
    prefixes: tuple[str, ...], suffix: str, *, default: float
) -> float:
    name = first_set_env_name(prefixes, suffix)
    if name is None:
        return default
    return parse_float_env(name, default=default)


def parse_chain_int_env(
    prefixes: tuple[str, ...], suffix: str, *, default: int
) -> int:
    name = first_set_env_name(prefixes, suffix)
    if name is None:
        return default
    return parse_int_env(name, default=default)


def parse_chain_positive_int_env(
    prefixes: tuple[str, ...], suffix: str, *, default: int
) -> int:
    name = first_set_env_name(prefixes, suffix)
    if name is None:
        return default
    return parse_positive_int_env(name, default=default)


def parse_chain_optional_float_env(
    prefixes: tuple[str, ...], suffix: str
) -> float | None:
    name = first_set_env_name(prefixes, suffix)
    if name is None:
        return None
    return parse_optional_float_env(name)


def parse_chain_optional_positive_int_env(
    prefixes: tuple[str, ...], suffix: str
) -> int | None:
    name = first_set_env_name(prefixes, suffix)
    if name is None:
        return None
    return parse_optional_positive_int_env(name)


def parse_chain_optional_bool_env(
    prefixes: tuple[str, ...], suffix: str
) -> bool | None:
    name = first_set_env_name(prefixes, suffix)
    if name is None:
        return None
    return parse_optional_bool_env(name)


def parse_chain_bool_env(
    prefixes: tuple[str, ...], suffix: str, *, default: bool
) -> bool:
    name = first_set_env_name(prefixes, suffix)
    if name is None:
        return default
    return parse_bool_env(name, default=default)


def parse_optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} must be non-negative, got {raw!r}")
    return value


def parse_float_env(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {raw!r}")
    return value


def parse_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    return value


def parse_positive_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {raw!r}")
    return value


def parse_optional_positive_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {raw!r}")
    return value


def parse_optional_bool_env(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


def parse_bool_env(name: str, *, default: bool) -> bool:
    value = parse_optional_bool_env(name)
    return default if value is None else value
