#!/usr/bin/env python3
"""PresentBench judge that calls an OpenAI-compatible chat completions endpoint.

CLI mirrors upstream ``data/PresentBench_code/judge.py`` so the
``run_presentbench_official_judge.py`` wrapper can route to either backend with
the same arg layout. Upstream judge handlers (checklist loading, scoring,
truncation, resume) are reused — only the model call goes through a
third-party OpenAI-compatible proxy configured via
``GOOGLE_THIRD_API_URL`` / ``GOOGLE_THIRD_API_KEY`` (the proxy is expected to
forward to a Gemini-class judge model). Slides PDF is sent as a
chat-completions ``file`` content part; ``.md``/``.txt`` materials are inlined
as text.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Upstream PresentBench code is expected on PYTHONPATH (the wrapper sets this
# via subprocess_env_with_code_root). Imports are done lazily inside main() so
# argument parsing and env validation do not require the upstream tree.


SUPPORTED_FILE_EXTS = {".pdf", ".md", ".txt"}

ENV_API_URL = "GOOGLE_THIRD_API_URL"
ENV_API_KEY = "GOOGLE_THIRD_API_KEY"
ENV_TIMEOUT = "GOOGLE_THIRD_API_TIMEOUT_SECONDS"
ENV_MAX_RETRIES = "GOOGLE_THIRD_API_MAX_RETRIES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PresentBench judge using OpenAI-compatible chat completions. "
            "Mirrors upstream judge.py CLI."
        )
    )
    parser.add_argument(
        "--api_type",
        choices=["openai"],
        required=True,
        help="Only 'openai' is supported by this script.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--thinking_level",
        help="Reserved for parity with upstream; ignored by chat-completions backend.",
    )
    parser.add_argument("--slides", type=str, help="Path to slides file.")
    parser.add_argument(
        "--material",
        type=str,
        nargs="+",
        help="Path(s) to material file(s). Multiple files allowed.",
    )
    parser.add_argument("--judge_prompt", type=str, default="")
    parser.add_argument("--common_judge_prompt", type=str, default="")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--zero_score", action="store_true")
    parser.add_argument("--min_timestamp", type=str)
    return parser.parse_args()


def build_openai_client(timeout: float, max_retries: int):
    from openai import OpenAI

    base_url = os.getenv(ENV_API_URL)
    api_key = os.getenv(ENV_API_KEY)
    if not base_url:
        raise SystemExit(f"error: {ENV_API_URL} must be set for --api_type openai")
    if not api_key:
        raise SystemExit(f"error: {ENV_API_KEY} must be set for --api_type openai")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


def encode_file_part(path: str) -> dict:
    ext = Path(path).suffix.lower()
    if ext in {".md", ".txt"}:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        return {
            "type": "text",
            "text": f"---\n[Material file: {os.path.basename(path)}]\n{text}\n---",
        }
    if ext == ".pdf":
        mime = "application/pdf"
    else:
        mime = "application/octet-stream"
    with open(path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("ascii")
    return {
        "type": "file",
        "file": {
            "filename": os.path.basename(path),
            "file_data": f"data:{mime};base64,{b64}",
        },
    }


class OpenAIChatJudgeAPI:
    """Adapter that satisfies upstream ``JudgeAPI``'s OpenAI-like file-path API."""

    def __init__(self, client, default_model: str):
        self._client = client
        self._default_model = default_model

    def upload_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_FILE_EXTS:
            logging.warning(
                "Unsupported file extension %s for chat-completions judge; "
                "will be sent as application/octet-stream.",
                ext,
            )
        return file_path

    def generate_content(self, model, prompt, file_paths=None, **_kwargs):
        file_paths = file_paths or []
        content_parts: list[dict] = []
        for path in file_paths:
            try:
                content_parts.append(encode_file_part(path))
            except FileNotFoundError as exc:
                logging.error("openai chat judge: %s", exc)
                return None, None
        content_parts.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_parts}]
        try:
            response = self._client.chat.completions.create(
                model=model or self._default_model,
                messages=messages,
            )
        except Exception as exc:
            logging.error("openai chat completion failed: %s", exc)
            return None, None
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            logging.error("openai chat completion returned no choices: %s", exc)
            return None, None
        try:
            payload = response.model_dump()
        except AttributeError:
            payload = None
        return text, payload


def install_factory_patch(default_model: str) -> None:
    """Monkey-patch upstream ``judge.create_judge_api`` to handle ``openai``."""

    import judge  # type: ignore[import-not-found]
    from utils.api.base import OpenAIAPI  # type: ignore[import-not-found]
    from utils.api.judge_api import JudgeAPI  # type: ignore[import-not-found]

    timeout = float(os.getenv(ENV_TIMEOUT) or 600.0)
    max_retries = int(os.getenv(ENV_MAX_RETRIES) or 0)
    sdk_client = build_openai_client(timeout=timeout, max_retries=max_retries)

    class _ChatJudgeClient(OpenAIAPI):
        # Subclass OpenAIAPI so upstream JudgeAPI's isinstance check accepts us.
        def __init__(self, adapter: OpenAIChatJudgeAPI):
            self._adapter = adapter

        def upload_file(self, file_path: str) -> str:
            return self._adapter.upload_file(file_path)

        def generate_content(self, model, prompt, file_paths=None, **kwargs):
            return self._adapter.generate_content(
                model=model,
                prompt=prompt,
                file_paths=file_paths,
                **kwargs,
            )

    adapter = OpenAIChatJudgeAPI(sdk_client, default_model=default_model)
    chat_client = _ChatJudgeClient(adapter)

    original_factory = judge.create_judge_api

    def patched_factory(api_type: str):
        if api_type == "openai":
            return JudgeAPI(chat_client)
        return original_factory(api_type)

    judge.create_judge_api = patched_factory


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    try:
        from dotenv import load_dotenv  # noqa: WPS433
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(override=False)

    if args.api_type != "openai":
        print(
            f"error: unsupported --api_type {args.api_type!r}; this script only handles 'openai'",
            file=sys.stderr,
        )
        return 2

    install_factory_patch(default_model=args.model)
    import judge  # type: ignore[import-not-found]

    judge.main(args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
