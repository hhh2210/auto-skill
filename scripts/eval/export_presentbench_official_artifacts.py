#!/usr/bin/env python3
"""Export PresentBench eval text rows as simple official-result PDF artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from textwrap import wrap
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from auto_skill.example_packs import load_jsonl  # noqa: E402
from auto_skill.presentbench_eval import presentbench_result_dir  # noqa: E402

SLIDE_HEADING_RE = re.compile(r"^\s*(?:#{1,3}\s*)?Slide\s+\d+\s*[:.-]?\s*(.*)$", re.I)


def pack_task_index(packs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for pack in packs:
        if pack.get("source") != "PresentBench":
            continue
        pack_id = str(pack.get("pack_id") or "")
        for task in pack.get("heldout_tasks") or []:
            task_id = str(task.get("task_id") or "")
            if pack_id and task_id:
                index[(pack_id, task_id)] = task
    return index


def mode_result_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--mode-result-root must be MODE=PATH, got: {value}")
        mode, raw_path = value.split("=", 1)
        mode = mode.strip()
        if not mode:
            raise ValueError(f"--mode-result-root mode is empty: {value}")
        roots[mode] = Path(raw_path).expanduser()
    return roots


def clean_text(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text)
    text = text.replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_slide_pages(text: str, *, max_pages: int) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = SLIDE_HEADING_RE.match(line)
        if heading and current:
            pages.append(current)
            current = [clean_text(line)]
        else:
            current.append(clean_text(line))
    if current:
        pages.append(current)
    if not pages:
        pages = [["Empty PresentBench generation"]]
    return pages[:max_pages]


def pdf_escape(text: str) -> str:
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 13 Tf", "50 750 Td", "16 TL"]
    first = True
    for line in lines:
        for wrapped in wrap(line, width=88) or [""]:
            if not first:
                commands.append("T*")
            commands.append(f"({pdf_escape(wrapped)}) Tj")
            first = False
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("latin-1", errors="replace")


def write_simple_pdf(path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # Filled after page object IDs are known.
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_ids = []
    for lines in pages:
        stream = page_stream(lines)
        content_id = len(objects) + 2
        page_id = len(objects) + 1
        page_ids.append(page_id)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"%PDF-1.4\n")
        offsets = [0]
        for object_id, body in enumerate(objects, start=1):
            offsets.append(handle.tell())
            handle.write(f"{object_id} 0 obj\n".encode("ascii"))
            handle.write(body)
            handle.write(b"\nendobj\n")
        xref_offset = handle.tell()
        handle.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        handle.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            handle.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        handle.write(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )


def export_rows(
    *,
    packs: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    result_roots: dict[str, Path],
    max_pages: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    task_index = pack_task_index(packs)
    exported = []
    for row in eval_rows:
        if row.get("status") != "success":
            continue
        mode = str(row.get("mode") or "")
        result_root = result_roots.get(mode)
        if result_root is None:
            continue
        task = task_index.get((str(row.get("pack_id") or ""), str(row.get("task_id") or "")))
        if task is None:
            continue
        generation = row.get("generation") or {}
        text = generation.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        result_dir = presentbench_result_dir(result_root, str(task.get("source_task_id") or ""))
        pdf_path = result_dir / "slides.pdf"
        if pdf_path.exists() and not overwrite:
            status = "exists"
        else:
            write_simple_pdf(pdf_path, split_slide_pages(text, max_pages=max_pages))
            status = "written"
        metadata_path = result_dir / "auto_skill_export_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "pack_id": row.get("pack_id"),
                    "task_id": row.get("task_id"),
                    "source_task_id": task.get("source_task_id"),
                    "mode": mode,
                    "eval_status": row.get("status"),
                    "eval_overall_score": row.get("overall_score"),
                    "generation_model": generation.get("model"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        exported.append(
            {
                "status": status,
                "mode": mode,
                "pack_id": row.get("pack_id"),
                "task_id": row.get("task_id"),
                "source_task_id": task.get("source_task_id"),
                "slide_artifact": str(pdf_path),
            }
        )
    return exported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packs",
        type=Path,
        default=Path("artifacts/packs/example_packs.v1.jsonl"),
    )
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument(
        "--mode-result-root",
        action="append",
        required=True,
        help="Mode-to-result-root mapping, e.g. prompt_only=../PresentBench/results/prompt_only.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/presentbench_exported_artifacts.jsonl"),
    )
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_pages <= 0:
        print("error: --max-pages must be positive", file=sys.stderr)
        return 2
    try:
        result_roots = mode_result_roots(args.mode_result_root)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    packs = load_jsonl(args.packs)
    eval_rows = []
    for path in args.eval:
        eval_rows.extend(load_jsonl(path))
    exported = export_rows(
        packs=packs,
        eval_rows=eval_rows,
        result_roots=result_roots,
        max_pages=args.max_pages,
        overwrite=args.overwrite,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported),
        encoding="utf-8",
    )
    print(f"Wrote {len(exported)} slide artifact rows to {args.out}")
    print(json.dumps({"exported": len(exported)}, ensure_ascii=False, indent=2))
    return 0 if exported else 3


if __name__ == "__main__":
    raise SystemExit(main())
