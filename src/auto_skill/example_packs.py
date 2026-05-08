"""Build user-visible example packs from benchmark few-shot splits."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

PRIVATE_FIELDS = ("supervision", "judge", "statistics")
SCHEMA_VERSION = "example-pack/v1"
BUILDER_VERSION = "example-pack-builder/0.2"
PROMPT_TEMPLATE_VERSION = "desired-output/user-visible-only/v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_MATERIAL_CHARS = 16_000

logging.getLogger("pypdf").setLevel(logging.ERROR)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "item"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def repo_relative_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        pass
    try:
        candidate.resolve().relative_to(REPO_ROOT.parent.resolve())
        return os.path.relpath(candidate.resolve(), REPO_ROOT.resolve())
    except ValueError:
        return str(candidate)


def resolve_material_path(path: str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def task_token_estimate(task: dict[str, Any]) -> int:
    return max(1, len(task.get("task_input", "")) // 4)


def compact_material(material: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": repo_relative_path(material.get("path")),
        "bytes": material.get("bytes"),
        "is_lfs_pointer": material.get("is_lfs_pointer"),
    }


def read_pdf_excerpt(path: Path, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        return f"[PDF text unavailable: pypdf is not installed: {exc}]"

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - keep data build resilient.
        return f"[PDF text unavailable: {type(exc).__name__}: {str(exc)[:160]}]"

    chunks: list[str] = []
    current = 0
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            text = f"[Page {page_index} extraction failed: {type(exc).__name__}]"
        if not text.strip():
            continue
        chunk = f"\n[PDF page {page_index}]\n{text.strip()}"
        chunks.append(chunk)
        current += len(chunk)
        if current >= max_chars:
            break
    return "\n".join(chunks)[:max_chars] or "[PDF text extraction returned no text.]"


def read_material_excerpt(material: dict[str, Any], max_chars: int) -> dict[str, Any]:
    material_path = resolve_material_path(material.get("path"))
    if material_path is None:
        return {"path": None, "status": "missing_path", "text": ""}
    compact_path = repo_relative_path(material_path)
    if not material_path.exists():
        return {"path": compact_path, "status": "missing_file", "text": ""}

    suffix = material_path.suffix.lower()
    if suffix in {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}:
        text = material_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return {"path": compact_path, "status": "extracted", "text": text}
    if suffix == ".pdf":
        return {
            "path": compact_path,
            "status": "extracted",
            "text": read_pdf_excerpt(material_path, max_chars),
        }
    return {"path": compact_path, "status": "unsupported_type", "text": ""}


def material_context(materials: list[dict[str, Any]], max_chars: int = MAX_MATERIAL_CHARS) -> str:
    if not materials:
        return "No separate material files."

    sections = []
    remaining = max_chars
    for material in materials:
        if remaining <= 0:
            break
        excerpt = read_material_excerpt(material, remaining)
        text = excerpt["text"].strip()
        if text:
            sections.append(
                f"## Material: {excerpt['path']} ({excerpt['status']})\n{text[:remaining]}"
            )
            remaining -= len(text)
        else:
            sections.append(f"## Material: {excerpt['path']} ({excerpt['status']})")
    return "\n\n".join(sections)


def clean_task(
    task: dict[str, Any],
    *,
    task_id: str,
    role: str,
    generation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    desired_output: dict[str, Any] | None = None
    if role == "train":
        desired_output = {
            "status": "needs_generation",
            "generation_job_id": f"{task_id}::generate_desired_output",
        }
        if generation_metadata:
            desired_output.update(generation_metadata)

    return {
        "schema_version": SCHEMA_VERSION,
        "example_id" if role == "train" else "task_id": task_id,
        "source": task["source"],
        "source_task_id": str(task["source_id"]),
        "domain": task.get("domain", {}),
        "task_input": task["task_input"],
        "materials": [compact_material(item) for item in task.get("materials", [])],
        "expected_artifacts": task.get("expected_artifacts", []),
        "desired_output": desired_output,
    }


def private_task(task: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_ref": task_id,
        "source": task["source"],
        "source_task_id": str(task["source_id"]),
        "supervision": task.get("supervision"),
        "judge": task.get("judge"),
        "statistics": task.get("statistics"),
    }


def generation_prompt(
    task: dict[str, Any],
    *,
    example_id: str,
    max_material_chars: int = MAX_MATERIAL_CHARS,
) -> str:
    artifact_types = ", ".join(task.get("expected_artifacts", [])) or "task output"
    materials = task.get("materials", [])
    materials_text = material_context(
        [compact_material(item) for item in materials],
        max_chars=max_material_chars,
    )

    return f"""Create a high-quality final output for the task below.

This output will later be paired with the task input as a user-visible example.
Use only the task input and material excerpts below.
Return only the final output. Do not include meta-commentary about dataset construction.

Example ID: {example_id}
Expected artifact type: {artifact_types}

Task input:
{task["task_input"]}

Material excerpts:
{materials_text}

Write the final desired output only.
"""


def generation_job(
    task: dict[str, Any],
    *,
    pack_id: str,
    example_id: str,
    max_material_chars: int = MAX_MATERIAL_CHARS,
) -> dict[str, Any]:
    prompt = generation_prompt(
        task,
        example_id=example_id,
        max_material_chars=max_material_chars,
    )
    return {
        "schema_version": "example-generation-job/v1",
        "builder_version": BUILDER_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_sha256": prompt_sha256(prompt),
        "material_budget_chars": max_material_chars,
        "job_id": f"{example_id}::generate_desired_output",
        "pack_id": pack_id,
        "example_id": example_id,
        "source": task["source"],
        "source_task_id": str(task["source_id"]),
        "artifact_types": task.get("expected_artifacts", []),
        "prompt": prompt,
        "private_eval_ref": "artifacts/private/example_private_eval.jsonl",
    }


def build_pack(
    split: dict[str, Any],
    *,
    max_material_chars: int = MAX_MATERIAL_CHARS,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pack_id = slugify(split["split_id"])

    train_examples = []
    heldout_tasks = []
    train_private = []
    heldout_private = []
    jobs = []

    for index, task in enumerate(split["train_examples"]):
        example_id = f"{pack_id}::train::{index}"
        job = generation_job(
            task,
            pack_id=pack_id,
            example_id=example_id,
            max_material_chars=max_material_chars,
        )
        train_examples.append(
            clean_task(
                task,
                task_id=example_id,
                role="train",
                generation_metadata={
                    "prompt_sha256": job["prompt_sha256"],
                    "prompt_template_version": job["prompt_template_version"],
                    "builder_version": job["builder_version"],
                },
            )
        )
        train_private.append(private_task(task, task_id=example_id))
        jobs.append(job)

    for index, task in enumerate(split["heldout_tasks"]):
        task_id = f"{pack_id}::heldout::{index}"
        heldout_tasks.append(clean_task(task, task_id=task_id, role="heldout"))
        heldout_private.append(private_task(task, task_id=task_id))

    pack = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "pack_id": pack_id,
        "split_id": split["split_id"],
        "source": split["source"],
        "learning_problem": split.get("learning_problem", "few_shot_skill_induction"),
        "domain": split.get("domain", {}),
        "input_boundary": {
            "auto_skill_module_can_use": [
                "train_examples.task_input",
                "train_examples.materials",
                "train_examples.desired_output.text",
                "optional user notes if added later",
            ],
            "must_not_use_for_induction": [
                "private rubrics",
                "private judge prompts",
                "heldout tasks",
                "heldout scores",
            ],
        },
        "train_examples": train_examples,
        "heldout_tasks": heldout_tasks,
    }
    private = {
        "schema_version": SCHEMA_VERSION,
        "pack_id": pack_id,
        "split_id": split["split_id"],
        "source": split["source"],
        "train_private": train_private,
        "heldout_private": heldout_private,
    }
    return pack, private, jobs


def summarize_packs(packs: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> str:
    source_counts = Counter(pack["source"] for pack in packs)
    lines = [
        "# Example Pack Summary",
        "",
        "These packs are the current data-first surface for the project.",
        "",
        "## Totals",
        "",
        f"- Packs: {len(packs)}",
        f"- Train examples needing desired-output generation: {len(jobs)}",
        f"- Heldout tasks: {sum(len(pack['heldout_tasks']) for pack in packs)}",
        f"- Sources: {dict(source_counts)}",
        "",
        "## Packs",
        "",
        "| pack_id | source | domain | train | heldout | input tokens est. | materials |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for pack in packs:
        domain = pack.get("domain", {})
        domain_label = " / ".join(str(value) for value in domain.values() if value)
        tasks = pack["train_examples"] + pack["heldout_tasks"]
        token_estimate = sum(task_token_estimate(task) for task in tasks)
        material_count = sum(len(task.get("materials", [])) for task in tasks)
        lines.append(
            (
                "| {pack_id} | {source} | {domain} | {train} | {heldout} | "
                "{tokens} | {materials} |"
            ).format(
                pack_id=pack["pack_id"],
                source=pack["source"],
                domain=domain_label,
                train=len(pack["train_examples"]),
                heldout=len(pack["heldout_tasks"]),
                tokens=token_estimate,
                materials=material_count,
            )
        )

    lines.extend(
        [
            "",
            "## Next Data Step",
            "",
            (
                "Run the generation jobs to fill `train_examples[*].desired_output`, "
                "then freeze a first `example_packs.v1.jsonl` for auto-skill "
                "framework iteration."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_example_pack_artifacts(
    splits: list[dict[str, Any]],
    *,
    max_material_chars: int = MAX_MATERIAL_CHARS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    packs: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for split in splits:
        pack, private, split_jobs = build_pack(split, max_material_chars=max_material_chars)
        packs.append(pack)
        private_rows.append(private)
        jobs.extend(split_jobs)
    return packs, private_rows, jobs
