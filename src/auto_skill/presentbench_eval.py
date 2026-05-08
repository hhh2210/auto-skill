"""PresentBench official-evaluation readiness helpers."""

from __future__ import annotations

import importlib.util
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PRESENTBENCH_REQUIRED_MODULES = (
    "dotenv",
    "google.genai",
    "PIL",
    "requests",
    "tqdm",
    "yaml",
)


@dataclass(frozen=True)
class PresentBenchOfficialEvalReadiness:
    task_id: str
    source_task_id: str
    status: str
    case_dir: str
    expected_result_dir: str
    official_code_root: str | None
    missing: list[str]
    material_files: list[str]
    slide_artifact: str | None
    failure_flag: str | None
    score_artifact: str | None
    score_selection_error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def presentbench_case_dir(data_root: Path, source_task_id: str) -> Path:
    return data_root / source_task_id


def presentbench_result_dir(result_root: Path, source_task_id: str) -> Path:
    return result_root / source_task_id / "generation_task" / "results"


def find_slide_artifact(result_dir: Path) -> Path | None:
    for name in ("slides.pdf", "slides.pptx"):
        candidate = result_dir / name
        if candidate.exists():
            return candidate
    return None


def find_material_files(case_dir: Path) -> list[Path]:
    """Match PresentBench's material discovery order."""

    material_files: list[Path] = []
    for suffix in (".md", ".pdf"):
        candidate = case_dir / f"material{suffix}"
        if candidate.exists():
            material_files.append(candidate)
            break

    index = 1
    while True:
        found = None
        for suffix in (".md", ".pdf"):
            candidate = case_dir / f"material_{index}{suffix}"
            if candidate.exists():
                found = candidate
                break
        if found is None:
            break
        material_files.append(found)
        index += 1
    return material_files


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


SCORE_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


def score_artifact_timestamp(path: Path) -> datetime | None:
    match = SCORE_TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def score_artifact_judge_prefix(path: Path) -> str | None:
    match = SCORE_TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    prefix = path.name[: match.start()]
    return prefix.removesuffix("_")


def find_score_artifact(result_dir: Path, *, judge_model: str | None = None) -> Path | None:
    score_files = sorted(result_dir.glob("*_score.yaml"))
    if judge_model:
        score_files = [path for path in score_files if path.name.startswith(f"{judge_model}_")]
    if not score_files:
        return None

    timestamped = [
        (timestamp, score_artifact_judge_prefix(path), path)
        for path in score_files
        if (timestamp := score_artifact_timestamp(path)) is not None
    ]
    if timestamped:
        prefixes = {prefix for _, prefix, _ in timestamped if prefix}
        if judge_model is None and len(prefixes) > 1:
            names = ", ".join(path.name for _, _, path in timestamped)
            raise ValueError(
                "ambiguous PresentBench score files from multiple judge models; "
                f"pass --judge-model. files: {names}"
            )
        timestamped.sort(key=lambda item: (item[0], item[2].name), reverse=True)
        return timestamped[0][2]

    if len(score_files) == 1:
        return score_files[0]
    names = ", ".join(path.name for path in score_files)
    raise ValueError(f"ambiguous PresentBench score files without timestamps: {names}")


def load_presentbench_score(score_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(score_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"PresentBench score file is not a mapping: {score_path}")
    total = data.get("total")
    if not isinstance(total, dict):
        raise ValueError(f"PresentBench score file missing total: {score_path}")
    score = total.get("weighted_arithmetic_mean_percent")
    if isinstance(score, bool):
        raise ValueError(
            "PresentBench score file missing total.weighted_arithmetic_mean_percent: "
            f"{score_path}"
        )
    if not isinstance(score, (int, float)):
        raise ValueError(
            "PresentBench score file missing total.weighted_arithmetic_mean_percent: "
            f"{score_path}"
        )
    return {
        "score_percent": float(score),
        "yes_count": total.get("yes_count"),
        "valid_count": total.get("valid_count"),
        "not_applicable_count": total.get("not_applicable_count"),
        "score_file": str(score_path),
    }


def check_presentbench_official_eval_readiness(
    *,
    task: dict[str, Any],
    data_root: Path,
    result_root: Path,
    code_root: Path | None = None,
    judge_model: str | None = None,
) -> PresentBenchOfficialEvalReadiness:
    """Check whether a PresentBench heldout task can be scored by official scripts."""

    task_id = str(task.get("task_id") or task.get("example_id") or "")
    source_task_id = str(task.get("source_task_id") or "")
    case_dir = presentbench_case_dir(data_root, source_task_id)
    result_dir = presentbench_result_dir(result_root, source_task_id)
    missing = []

    domain = source_task_id.split("/", 1)[0]
    required = {
        "case_dir": case_dir,
        "instructions": case_dir / "generation_task" / "instructions.md",
        "judge_prompt": case_dir / "generation_task" / "judge_prompt.json",
        "common_judge_prompt": data_root / domain / "common_judge_prompt.json",
        "judge_weights": data_root / domain / "judge_weights.yaml",
        "result_dir": result_dir,
    }
    for label, path in required.items():
        if not path.exists():
            missing.append(f"{label}: {path}")

    if code_root is not None:
        for label, path in {
            "presentbench_code_root": code_root,
            "presentbench_judge_all": code_root / "judge_all.py",
            "presentbench_judge": code_root / "judge.py",
            "presentbench_scoring": code_root / "scoring.py",
            "presentbench_utils_paths": code_root / "utils" / "paths.py",
            "presentbench_utils_material": code_root / "utils" / "material_utils.py",
            "presentbench_utils_judge": code_root / "utils" / "judge_utils.py",
            "presentbench_utils_score": code_root / "utils" / "score_utils.py",
            "presentbench_utils_api_base": code_root / "utils" / "api" / "base.py",
            "presentbench_utils_api_judge": code_root / "utils" / "api" / "judge_api.py",
            "presentbench_utils_pptx_to_pdf": code_root / "utils" / "pptx_to_pdf.py",
            "presentbench_utils_count_pages": code_root / "utils" / "count_pages.py",
            "presentbench_utils_truncate_pages": code_root / "utils" / "truncate_pages.py",
            "presentbench_utils_encode_file": code_root / "utils" / "encode_file.py",
            "presentbench_utils_generate_checklist": (
                code_root / "utils" / "generate_checklist.py"
            ),
        }.items():
            if not path.exists():
                missing.append(f"{label}: {path}")
        for module_name in PRESENTBENCH_REQUIRED_MODULES:
            if not module_available(module_name):
                missing.append(f"presentbench_python_module: {module_name}")

    material_files = find_material_files(case_dir)
    if not material_files:
        missing.append(f"material.md/material.pdf or material_N.*: {case_dir}")

    slide_artifact = find_slide_artifact(result_dir)
    failure_flag = result_dir / "slides_generation_failed.txt"
    if slide_artifact is None and not failure_flag.exists():
        missing.append(f"slides.pdf or slides.pptx: {result_dir}")
    if slide_artifact is not None and slide_artifact.suffix.lower() == ".pptx":
        if not module_available("pptx"):
            missing.append("presentbench_python_module: pptx")
        if shutil.which("libreoffice") is None:
            missing.append("presentbench_binary: libreoffice")
    score_selection_error = None
    try:
        score_artifact = find_score_artifact(result_dir, judge_model=judge_model)
    except ValueError as exc:
        score_artifact = None
        score_selection_error = str(exc)

    if missing:
        status = "missing_official_eval_artifacts"
    elif score_selection_error:
        status = "ambiguous_score_artifacts"
    elif score_artifact is not None:
        status = "scored"
    elif slide_artifact is not None:
        status = "ready_for_official_judge"
    elif failure_flag.exists():
        status = "ready_for_zero_score"
    else:
        status = "missing_official_eval_artifacts"

    return PresentBenchOfficialEvalReadiness(
        task_id=task_id,
        source_task_id=source_task_id,
        status=status,
        case_dir=str(case_dir),
        expected_result_dir=str(result_dir),
        official_code_root=str(code_root) if code_root else None,
        missing=missing,
        material_files=[str(path) for path in material_files],
        slide_artifact=str(slide_artifact) if slide_artifact else None,
        failure_flag=str(failure_flag) if failure_flag.exists() else None,
        score_artifact=str(score_artifact) if score_artifact else None,
        score_selection_error=score_selection_error,
    )
