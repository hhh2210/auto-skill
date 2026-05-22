from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_skill.llm import (
    ChatCompletionClient,
    ChatCompletionConfig,
    ConfigError,
    parse_int_env,
    parse_optional_bool_env,
    parse_optional_positive_int_env,
    parse_positive_int_env,
)


class LLMConfigTests(unittest.TestCase):
    def test_parse_positive_int_env_uses_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parse_positive_int_env("BAILIAN_NUM_THREADS", default=1), 1)

    def test_parse_int_env_rejects_negative_to_preserve_legacy_contract(self) -> None:
        with patch.dict(os.environ, {"BAILIAN_MAX_RETRIES": "-1"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "must be non-negative"):
                parse_int_env("BAILIAN_MAX_RETRIES", default=0)

    def test_parse_positive_int_env_rejects_zero(self) -> None:
        with patch.dict(os.environ, {"BAILIAN_NUM_THREADS": "0"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "must be positive"):
                parse_positive_int_env("BAILIAN_NUM_THREADS", default=1)

    def test_parse_optional_positive_int_env_rejects_zero(self) -> None:
        with patch.dict(os.environ, {"BAILIAN_THINKING_BUDGET": "0"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "must be positive"):
                parse_optional_positive_int_env("BAILIAN_THINKING_BUDGET")

    def test_config_reads_bailian_num_threads(self) -> None:
        env = {
            "BAILIAN_BASE_URL": "https://example.com/v1",
            "BAILIAN_API_KEY": "test-key",
            "BAILIAN_MODEL": "qwen-plus",
            "BAILIAN_NUM_THREADS": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ChatCompletionConfig.from_env(Path("/tmp/auto-skill-no-such.env"))

        self.assertEqual(config.num_threads, 3)

    def test_parse_optional_bool_env(self) -> None:
        with patch.dict(os.environ, {"BAILIAN_ENABLE_THINKING": "false"}, clear=True):
            self.assertFalse(parse_optional_bool_env("BAILIAN_ENABLE_THINKING"))
        with patch.dict(os.environ, {"BAILIAN_ENABLE_THINKING": "true"}, clear=True):
            self.assertTrue(parse_optional_bool_env("BAILIAN_ENABLE_THINKING"))
        with patch.dict(os.environ, {"BAILIAN_ENABLE_THINKING": "maybe"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "must be a boolean"):
                parse_optional_bool_env("BAILIAN_ENABLE_THINKING")

    def test_config_reads_enable_thinking(self) -> None:
        env = {
            "BAILIAN_BASE_URL": "https://example.com/v1",
            "BAILIAN_API_KEY": "test-key",
            "BAILIAN_MODEL": "qwen-plus",
            "BAILIAN_ENABLE_THINKING": "false",
            "BAILIAN_THINKING_BUDGET": "2048",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ChatCompletionConfig.from_env(Path("/tmp/auto-skill-no-such.env"))

        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.thinking_budget, 2048)

    def test_config_reads_stream_flags(self) -> None:
        env = {
            "BAILIAN_BASE_URL": "https://example.com/v1",
            "BAILIAN_API_KEY": "test-key",
            "BAILIAN_MODEL": "qwen-plus",
            "BAILIAN_STREAM": "true",
            "BAILIAN_STREAM_LOG": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ChatCompletionConfig.from_env(Path("/tmp/auto-skill-no-such.env"))

        self.assertTrue(config.stream)
        self.assertFalse(config.stream_log)

    def test_prefix_requires_explicit_model(self) -> None:
        env = {
            "BAILIAN_BASE_URL": "https://example.com/v1",
            "BAILIAN_API_KEY": "test-key",
            "BAILIAN_MODEL": "qwen-plus",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ConfigError, "JUDGE_MODEL"):
                ChatCompletionConfig.from_env(
                    Path("/tmp/auto-skill-no-such.env"),
                    prefix="JUDGE",
                )

    def test_prefix_falls_back_to_bailian_for_url_and_key(self) -> None:
        env = {
            "BAILIAN_BASE_URL": "https://example.com/v1",
            "BAILIAN_API_KEY": "test-key",
            "BAILIAN_MODEL": "qwen-plus",
            "JUDGE_MODEL": "qwen-max",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ChatCompletionConfig.from_env(
                Path("/tmp/auto-skill-no-such.env"),
                prefix="JUDGE",
            )
        self.assertEqual(config.model, "qwen-max")
        self.assertEqual(config.base_url, "https://example.com/v1")
        self.assertEqual(config.api_key, "test-key")

    def test_prefix_overrides_url_and_optional_params(self) -> None:
        env = {
            "BAILIAN_BASE_URL": "https://bailian.example/v1",
            "BAILIAN_API_KEY": "bailian-key",
            "BAILIAN_MODEL": "qwen-plus",
            "BAILIAN_TEMPERATURE": "0.7",
            "JUDGE_BASE_URL": "https://judge.example/v1",
            "JUDGE_API_KEY": "judge-key",
            "JUDGE_MODEL": "claude-haiku-4-5",
            "JUDGE_TEMPERATURE": "0.0",
            "JUDGE_NUM_THREADS": "2",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ChatCompletionConfig.from_env(
                Path("/tmp/auto-skill-no-such.env"),
                prefix="JUDGE",
            )
        self.assertEqual(config.base_url, "https://judge.example/v1")
        self.assertEqual(config.api_key, "judge-key")
        self.assertEqual(config.model, "claude-haiku-4-5")
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.num_threads, 2)

    def test_legacy_from_env_unchanged(self) -> None:
        env = {
            "BAILIAN_BASE_URL": "https://example.com/v1",
            "BAILIAN_API_KEY": "test-key",
            "BAILIAN_MODEL": "qwen-plus",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ChatCompletionConfig.from_env(Path("/tmp/auto-skill-no-such.env"))
        self.assertEqual(config.base_url, "https://example.com/v1")
        self.assertEqual(config.model, "qwen-plus")

    def test_streaming_completion_accumulates_content_chunks(self) -> None:
        class Usage:
            def model_dump(self) -> dict[str, int]:
                return {"completion_tokens": 2, "prompt_tokens": 1, "total_tokens": 3}

        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] | None = None

            def create(self, **kwargs: object) -> list[SimpleNamespace]:
                self.kwargs = kwargs
                return [
                    SimpleNamespace(
                        model="qwen-plus",
                        choices=[
                            SimpleNamespace(
                                finish_reason=None,
                                delta=SimpleNamespace(
                                    reasoning_content="thinking",
                                    content=None,
                                ),
                            )
                        ],
                    ),
                    SimpleNamespace(
                        model="qwen-plus",
                        choices=[
                            SimpleNamespace(
                                finish_reason=None,
                                delta=SimpleNamespace(reasoning_content=None, content="hel"),
                            )
                        ],
                    ),
                    SimpleNamespace(
                        model="qwen-plus",
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                delta=SimpleNamespace(reasoning_content=None, content="lo"),
                            )
                        ],
                        usage=Usage(),
                    ),
                ]

        fake_completions = FakeCompletions()
        config = ChatCompletionConfig(
            base_url="https://example.com/v1",
            api_key="test-key",
            model="qwen-plus",
            enable_thinking=True,
            stream=True,
        )
        client = ChatCompletionClient(config)
        client._client = SimpleNamespace(  # noqa: SLF001 - replace SDK client with local fake.
            chat=SimpleNamespace(completions=fake_completions)
        )

        result = client.complete([{"role": "user", "content": "hi"}], max_tokens=10)

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.model, "qwen-plus")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNone(result.request_id)
        self.assertEqual(
            result.usage,
            {"completion_tokens": 2, "prompt_tokens": 1, "total_tokens": 3},
        )
        self.assertIsNotNone(fake_completions.kwargs)
        assert fake_completions.kwargs is not None
        self.assertTrue(fake_completions.kwargs["stream"])
        self.assertEqual(fake_completions.kwargs["extra_body"], {"enable_thinking": True})
        self.assertEqual(fake_completions.kwargs["stream_options"], {"include_usage": True})


if __name__ == "__main__":
    unittest.main()
