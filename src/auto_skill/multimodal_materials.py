"""Helpers for multimodal PresentBench material digestion."""

from __future__ import annotations

import base64
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIGEST_PROMPT_VERSION = "presentbench-material-digest/qwen-vl/v1"


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def parse_page_spec(value: str) -> tuple[int, ...]:
    """Parse a 1-indexed page selector such as ``"1,3-5"``."""

    pages: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"invalid page range {part!r}")
            pages.update(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError(f"invalid page number {part!r}")
            pages.add(page)
    if not pages:
        raise ValueError("page selector must include at least one page")
    return tuple(sorted(pages))


def encode_image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_content_part(path: Path) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": encode_image_data_url(path)}}


def text_content_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def render_pdf_pages(
    pdf_path: Path,
    *,
    pages: tuple[int, ...],
    out_dir: Path,
    dpi: int = 144,
) -> list[Path]:
    """Render selected PDF pages to PNG files using ``pdftoppm``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for page in pages:
        prefix = out_dir / f"{pdf_path.stem}.page-{page}"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(dpi),
                "-png",
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        candidates = sorted(out_dir.glob(f"{prefix.name}-*.png"))
        if not candidates:
            raise FileNotFoundError(f"pdftoppm produced no PNG for {pdf_path} page {page}")
        rendered.append(candidates[-1])
    return rendered


def build_material_digest_prompt(
    *,
    task_id: str,
    task_input: str,
    material_paths: list[str],
) -> str:
    material_list = "\n".join(f"- {path}" for path in material_paths) or "- none"
    return f"""You are preparing a structured visual/material digest for a slide-generation agent.

The downstream agent must generate a PresentBench slide deck. Inspect the provided
material page images and any inline text carefully. Extract only source-grounded
information that can help build slides.

Task ID: {task_id}

Task instructions:
{task_input}

Material files:
{material_list}

Return a compact JSON object with these keys:
- visual_elements: important figures, diagrams, tables, charts, equations,
  screenshots, or page layouts.
- source_fidelity_constraints: concrete details that generated slides must preserve.
- slide_generation_hints: content/layout suggestions grounded in the material.
- missing_or_uncertain: details that cannot be read confidently from the provided pages.

Do not invent content. If a page is unreadable or a figure is unclear, say so.
"""


def task_identifier(task: dict[str, Any]) -> str:
    return str(task.get("task_id") or task.get("example_id") or task.get("source_task_id"))
