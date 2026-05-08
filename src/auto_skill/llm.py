"""OpenAI-compatible chat completion client used for data construction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

_UNSET = object()


class ConfigError(ValueError):
    """Raised when local LLM environment configuration is invalid."""


@dataclass(frozen=True)
class ChatCompletionConfig:
    """Configuration for Bailian or any OpenAI-compatible chat completions endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 600.0
    max_retries: int = 0
    temperature: float | None = None
    num_threads: int = 1
    enable_thinking: bool | None = None
    thinking_budget: int | None = None
    stream: bool = False
    stream_log: bool = False
    stream_include_usage: bool = True

    @classmethod
    def from_env(
        cls,
        env_file: Path | None = None,
        *,
        prefix: str | None = None,
    ) -> ChatCompletionConfig:
        """Load chat-completion config from environment.

        ``prefix=None`` (default) preserves the legacy ``BAILIAN_*`` lookup with
        ``OPENAI_*`` as fallback. When ``prefix`` is given (e.g. ``"JUDGE"``),
        ``{prefix}_*`` is consulted first; ``BASE_URL``/``API_KEY`` and the
        optional tuning params fall back to ``BAILIAN_*``/``OPENAI_*``, but
        ``MODEL`` has no fallback so a missing ``{prefix}_MODEL`` is an error.
        This lets callers split judge-side from solver-side cleanly.
        """

        if env_file is not None:
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        if prefix is None:
            url_chain = ("BAILIAN_BASE_URL", "OPENAI_BASE_URL")
            key_chain = ("BAILIAN_API_KEY", "OPENAI_API_KEY")
            model_chain = ("BAILIAN_MODEL", "OPENAI_MODEL")
            param_prefixes: tuple[str, ...] = ("BAILIAN",)
            url_label = "BAILIAN_BASE_URL or OPENAI_BASE_URL"
            key_label = "BAILIAN_API_KEY or OPENAI_API_KEY"
            model_label = "BAILIAN_MODEL or OPENAI_MODEL"
        else:
            url_chain = (f"{prefix}_BASE_URL", "BAILIAN_BASE_URL", "OPENAI_BASE_URL")
            key_chain = (f"{prefix}_API_KEY", "BAILIAN_API_KEY", "OPENAI_API_KEY")
            model_chain = (f"{prefix}_MODEL",)
            param_prefixes = (prefix, "BAILIAN")
            url_label = f"{prefix}_BASE_URL"
            key_label = f"{prefix}_API_KEY"
            model_label = f"{prefix}_MODEL (required when prefix={prefix!r}; no fallback)"

        base_url = first_set_env(*url_chain)
        api_key = first_set_env(*key_chain)
        model = first_set_env(*model_chain)

        missing = [
            name
            for name, value in (
                (url_label, base_url),
                (key_label, api_key),
                (model_label, model),
            )
            if not value
        ]
        if missing:
            raise ConfigError("missing required environment variables: " + ", ".join(missing))

        timeout_seconds = parse_chain_float_env(
            param_prefixes, "TIMEOUT_SECONDS", default=600.0
        )
        max_retries = parse_chain_int_env(param_prefixes, "MAX_RETRIES", default=0)
        num_threads = parse_chain_positive_int_env(
            param_prefixes, "NUM_THREADS", default=1
        )

        return cls(
            base_url=str(base_url),
            api_key=str(api_key),
            model=str(model),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            temperature=parse_chain_optional_float_env(param_prefixes, "TEMPERATURE"),
            num_threads=num_threads,
            enable_thinking=parse_chain_optional_bool_env(
                param_prefixes, "ENABLE_THINKING"
            ),
            thinking_budget=parse_chain_optional_positive_int_env(
                param_prefixes, "THINKING_BUDGET"
            ),
            stream=parse_chain_bool_env(param_prefixes, "STREAM", default=False),
            stream_log=parse_chain_bool_env(param_prefixes, "STREAM_LOG", default=False),
            stream_include_usage=parse_chain_bool_env(
                param_prefixes, "STREAM_INCLUDE_USAGE", default=True
            ),
        )


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
    if value < 0:
        raise ConfigError(f"{name} must be non-negative, got {raw!r}")
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


@dataclass(frozen=True)
class ChatCompletionResult:
    text: str
    model: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None
    request_id: str | None = None


class ChatCompletionClient:
    """Thin wrapper around the OpenAI SDK for compatible chat completion APIs."""

    def __init__(self, config: ChatCompletionConfig):
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None | object = _UNSET,
        thinking_budget: int | None | object = _UNSET,
    ) -> ChatCompletionResult:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resolved_enable_thinking = (
            self.config.enable_thinking if enable_thinking is _UNSET else enable_thinking
        )
        resolved_thinking_budget = (
            self.config.thinking_budget if thinking_budget is _UNSET else thinking_budget
        )
        extra_body: dict[str, Any] = {}
        if resolved_enable_thinking is not None:
            extra_body["enable_thinking"] = resolved_enable_thinking
        if resolved_thinking_budget is not None and resolved_enable_thinking is not False:
            extra_body["thinking_budget"] = resolved_thinking_budget
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.config.stream:
            return self._complete_streaming(kwargs)
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage is not None else None
        return ChatCompletionResult(
            text=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            request_id=getattr(response, "id", None),
        )

    def _complete_streaming(self, kwargs: dict[str, Any]) -> ChatCompletionResult:
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        if self.config.stream_include_usage:
            stream_kwargs["stream_options"] = {"include_usage": True}
        chunks = self._client.chat.completions.create(**stream_kwargs)
        content_parts: list[str] = []
        model: str | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        request_id: str | None = None
        for chunk in chunks:
            model = getattr(chunk, "model", None) or model
            request_id = getattr(chunk, "id", None) or request_id
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage.model_dump() if hasattr(chunk_usage, "model_dump") else None
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            reasoning = getattr(delta, "reasoning_content", None)
            text = getattr(delta, "content", None)
            if self.config.stream_log and reasoning:
                print(reasoning, end="", file=sys.stderr, flush=True)
            if text:
                content_parts.append(text)
                if self.config.stream_log:
                    print(text, end="", file=sys.stderr, flush=True)
        if self.config.stream_log:
            print(file=sys.stderr, flush=True)
        return ChatCompletionResult(
            text="".join(content_parts),
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            request_id=request_id,
        )
