#!/usr/bin/env python3
"""Create multimodal material digests for PresentBench tasks."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl, write_jsonl  # noqa: E402
from auto_skill.llm import ChatCompletionClient, ChatCompletionConfig, ConfigError  # noqa: E402
from auto_skill.multimodal_materials import (  # noqa: E402
    DEFAULT_DIGEST_PROMPT_VERSION,
    build_material_digest_prompt,
    image_content_part,
    parse_page_spec,
    render_pdf_pages,
    repo_relative_path,
    resolve_repo_path,
    task_identifier,
    text_content_part,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--config-prefix",
        help=(
            "Optional env prefix for the multimodal model, e.g. QWEN_VL. "
            "When omitted, BAILIAN_* / OPENAI_* are used."
        ),
    )
    parser.add_argument("--pack-id")
    parser.add_argument("--task-id")
    parser.add_argument("--source", default="PresentBench")
    parser.add_argument("--role", choices=["train", "heldout", "both"], default="heldout")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pages", default="1-2", help="1-indexed PDF pages to render.")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-image-bytes", type=int, default=9_500_000)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int, default=4096)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true", dest="enable_thinking")
    thinking.add_argument("--no-enable-thinking", action="store_false", dest="enable_thinking")
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_tasks(
    packs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[str, str, dict[str, Any]]]:
    selected: list[tuple[str, str, dict[str, Any]]] = []
    for pack in packs:
        if args.source and pack.get("source") != args.source:
            continue
        pack_id = str(pack.get("pack_id"))
        if args.pack_id and pack_id != args.pack_id:
            continue
        role_to_key = {
            "train": ("train", "train_examples"),
            "heldout": ("heldout", "heldout_tasks"),
        }
        roles = ("train", "heldout") if args.role == "both" else (args.role,)
        for role in roles:
            label, key = role_to_key[role]
            for task in pack.get(key, []):
                task_id = task_identifier(task)
                if args.task_id and task_id != args.task_id:
                    continue
                selected.append((pack_id, label, task))
                if args.limit is not None and len(selected) >= args.limit:
                    return selected
    return selected


def material_paths(task: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for material in task.get("materials") or []:
        if not isinstance(material, dict) or not material.get("path"):
            continue
        paths.append(resolve_repo_path(str(material["path"])))
    return paths


def build_messages_for_task(
    task: dict[str, Any],
    *,
    pages: tuple[int, ...],
    dpi: int,
    max_image_bytes: int,
    temp_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    task_id = task_identifier(task)
    paths = material_paths(task)
    prompt = build_material_digest_prompt(
        task_id=task_id,
        task_input=str(task.get("task_input") or ""),
        material_paths=[repo_relative_path(path) for path in paths],
    )
    parts: list[dict[str, Any]] = [text_content_part(prompt)]
    attached: list[str] = []
    warnings: list[str] = []
    for path in paths:
        if not path.exists():
            warnings.append(f"missing material: {repo_relative_path(path)}")
            continue
        if path.suffix.lower() == ".pdf":
            try:
                rendered = render_pdf_pages(path, pages=pages, out_dir=temp_dir, dpi=dpi)
            except Exception as exc:  # noqa: BLE001 - row-level failure is more useful.
                warnings.append(f"render failed for {repo_relative_path(path)}: {exc}")
                continue
            for image_path in rendered:
                size = image_path.stat().st_size
                if size > max_image_bytes:
                    warnings.append(
                        f"skip oversized rendered page {image_path.name}: {size} bytes"
                    )
                    continue
                parts.append(image_content_part(image_path))
                attached.append(repo_relative_path(path) + f"#page-image={image_path.name}")
        elif path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            parts.append(
                text_content_part(f"\n\n[Material text: {repo_relative_path(path)}]\n{text}")
            )
            attached.append(repo_relative_path(path))
        else:
            warnings.append(f"unsupported material type: {repo_relative_path(path)}")
    return [{"role": "user", "content": parts}], attached, warnings


def main() -> int:
    args = parse_args()
    try:
        pages = parse_page_spec(args.pages)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    packs = load_jsonl(args.packs)
    selected = iter_tasks(packs, args)
    print(f"Selected {len(selected)} PresentBench material digest task(s).")
    if args.dry_run:
        for pack_id, role, task in selected:
            material_count = len(material_paths(task))
            print(f"- {pack_id} {role} {task_identifier(task)} materials={material_count}")
        return 0

    try:
        config = ChatCompletionConfig.from_env(args.env_file, prefix=args.config_prefix)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    client = ChatCompletionClient(config)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="presentbench-material-digest-") as tmp:
        temp_dir = Path(tmp)
        for pack_id, role, task in selected:
            task_id = task_identifier(task)
            messages, attached, warnings = build_messages_for_task(
                task,
                pages=pages,
                dpi=args.dpi,
                max_image_bytes=args.max_image_bytes,
                temp_dir=temp_dir,
            )
            try:
                result = client.complete(
                    messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                )
                status = "success" if result.finish_reason == "stop" else "incomplete"
                row = {
                    "schema_version": "presentbench-material-digest/v1",
                    "prompt_template_version": DEFAULT_DIGEST_PROMPT_VERSION,
                    "pack_id": pack_id,
                    "role": role,
                    "task_id": task_id,
                    "source": task.get("source"),
                    "model": result.model or config.model,
                    "status": status,
                    "finish_reason": result.finish_reason,
                    "usage": result.usage,
                    "request_id": result.request_id,
                    "pages": list(pages),
                    "attached_materials": attached,
                    "warnings": warnings,
                    "digest_text": result.text,
                }
            except Exception as exc:  # noqa: BLE001 - keep batch append-only.
                row = {
                    "schema_version": "presentbench-material-digest/v1",
                    "prompt_template_version": DEFAULT_DIGEST_PROMPT_VERSION,
                    "pack_id": pack_id,
                    "role": role,
                    "task_id": task_id,
                    "source": task.get("source"),
                    "model": config.model,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pages": list(pages),
                    "attached_materials": attached,
                    "warnings": warnings,
                }
            rows.append(row)
            write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} material digest row(s) to {args.out}")
    return 0 if all(row.get("status") == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
