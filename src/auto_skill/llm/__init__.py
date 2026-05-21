"""OpenAI-compatible chat completion client used for data construction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from auto_skill.llm.client import (
    ChatCompletionClient,
    ChatCompletionConfig,
    ChatCompletionResult,
)
from auto_skill.llm.env import (
    ConfigError,
    first_set_env,
    first_set_env_name,
    parse_bool_env,
    parse_chain_bool_env,
    parse_chain_float_env,
    parse_chain_int_env,
    parse_chain_optional_bool_env,
    parse_chain_optional_float_env,
    parse_chain_optional_positive_int_env,
    parse_chain_positive_int_env,
    parse_float_env,
    parse_int_env,
    parse_optional_bool_env,
    parse_optional_float_env,
    parse_optional_positive_int_env,
    parse_positive_int_env,
)

__all__ = [
    "Any",
    "ChatCompletionClient",
    "ChatCompletionConfig",
    "ChatCompletionResult",
    "ConfigError",
    "OpenAI",
    "Path",
    "dataclass",
    "first_set_env",
    "first_set_env_name",
    "load_dotenv",
    "os",
    "parse_bool_env",
    "parse_chain_bool_env",
    "parse_chain_float_env",
    "parse_chain_int_env",
    "parse_chain_optional_bool_env",
    "parse_chain_optional_float_env",
    "parse_chain_optional_positive_int_env",
    "parse_chain_positive_int_env",
    "parse_float_env",
    "parse_int_env",
    "parse_optional_bool_env",
    "parse_optional_float_env",
    "parse_optional_positive_int_env",
    "parse_positive_int_env",
    "sys",
]

for _module_name in ("client", "env"):
    globals().pop(_module_name, None)
del _module_name
