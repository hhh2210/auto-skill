from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.eval.run_presentbench_official_judge import (
    CHAT_JUDGE_SCRIPT,
    build_judge_all_command,
    build_judge_command,
    command_manifest,
    outer_subprocess_workers,
    parse_mode_result_roots,
    preflight_warnings,
    resolve_judge_script,
    selected_judge_commands,
    subprocess_env_with_code_root,
    validate_preflight,
)

OPENAI_JUDGE_ENV_KEYS = (
    "GOOGLE_THIRD_API_URL",
    "GOOGLE_THIRD_API_KEY",
)


def write_minimal_presentbench_code_root(code_root: Path) -> None:
    """Create the official-code files required by readiness checks."""

    for relative in (
        "judge_all.py",
        "judge.py",
        "scoring.py",
        "utils/paths.py",
        "utils/material_utils.py",
        "utils/judge_utils.py",
        "utils/score_utils.py",
        "utils/api/base.py",
        "utils/api/judge_api.py",
        "utils/pptx_to_pdf.py",
        "utils/count_pages.py",
        "utils/truncate_pages.py",
        "utils/encode_file.py",
        "utils/generate_checklist.py",
    ):
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")


class PresentBenchOfficialJudgeRunnerTests(unittest.TestCase):
    def test_parse_mode_result_roots_requires_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "MODE=PATH"):
            parse_mode_result_roots(["/tmp/results"])
        with self.assertRaisesRegex(ValueError, "non-empty MODE=PATH"):
            parse_mode_result_roots(["auto_skill=  "])

    def test_parse_mode_result_roots_maps_repeated_values(self) -> None:
        roots = parse_mode_result_roots(
            ["prompt_only= /tmp/prompt ", "auto_skill=/tmp/skill"]
        )

        self.assertEqual(
            roots,
            {
                "prompt_only": Path("/tmp/prompt"),
                "auto_skill": Path("/tmp/skill"),
            },
        )

    def test_build_judge_all_command_includes_optional_controls(self) -> None:
        command = build_judge_all_command(
            python_executable="/python",
            judge_all=Path("data/PresentBench_code/judge_all.py"),
            agent_name="auto_skill",
            data_root=Path("data/PresentBench_repo"),
            result_root=Path("../PresentBench/results/auto_skill"),
            api_type="gemini",
            model="gemini-3-flash-preview",
            max_workers=4,
            thinking_level="low",
            min_timestamp="2026-05-09T00:00:00",
        )

        self.assertEqual(command[0], "/python")
        self.assertIn("--agent_name", command)
        self.assertIn("auto_skill", command)
        self.assertIn("--max_workers", command)
        self.assertIn("4", command)
        self.assertIn("--thinking_level", command)
        self.assertIn("--min_timestamp", command)

    def test_validate_preflight_requires_gemini_key_unless_dry_run_override(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            result_root = root / "results"
            code_root.mkdir()
            data_root.mkdir()
            result_root.mkdir()
            (code_root / "judge_all.py").write_text("# judge\n", encoding="utf-8")
            (code_root / "judge.py").write_text("# judge\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                errors = validate_preflight(
                    code_root=code_root,
                    data_root=data_root,
                    mode_roots={"prompt_only": result_root},
                    api_type="gemini",
                    allow_missing_env=False,
                )
                self.assertIn("GENAI_API_KEY", "\n".join(errors))

                errors = validate_preflight(
                    code_root=code_root,
                    data_root=data_root,
                    mode_roots={"prompt_only": result_root},
                    api_type="gemini",
                    allow_missing_env=True,
                )
                self.assertEqual(errors, [])

    def test_preflight_warnings_records_missing_gemini_key_for_dry_run_manifest(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                preflight_warnings(api_type="gemini", allow_missing_env=True),
                ["GENAI_API_KEY is required by upstream PresentBench gemini judge"],
            )
            self.assertEqual(
                preflight_warnings(api_type="gemini", allow_missing_env=False),
                [],
            )

    def test_build_judge_command_targets_one_selected_case(self) -> None:
        command = build_judge_command(
            python_executable="/python",
            judge=Path("data/PresentBench_code/judge.py"),
            readiness={
                "status": "ready_for_official_judge",
                "source_task_id": "education/course/case1",
                "case_dir": "/data/education/course/case1",
                "expected_result_dir": "/results/education/case1/generation_task/results",
                "slide_artifact": "/results/education/case1/generation_task/results/slides.pdf",
                "material_files": ["/data/education/course/case1/material.md"],
            },
            api_type="gemini",
            model="gemini-3-flash-preview",
            retry=5,
            thinking_level=None,
            min_timestamp=None,
        )

        self.assertEqual(command[0], "/python")
        self.assertIn("data/PresentBench_code/judge.py", command)
        self.assertIn("--slides", command)
        self.assertIn("/results/education/case1/generation_task/results/slides.pdf", command)
        self.assertIn("--judge_prompt", command)
        self.assertIn(
            "/data/education/course/case1/generation_task/judge_prompt.json",
            command,
        )
        self.assertIn("--common_judge_prompt", command)
        self.assertIn("/data/education/common_judge_prompt.json", command)
        self.assertIn("--material", command)
        self.assertIn("/data/education/course/case1/material.md", command)

    def test_selected_judge_commands_only_renders_unscored_pack_cells(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            case_dir = data_root / "education" / "case1"
            result_root = root / "results" / "prompt"
            result_dir = result_root / "education" / "case1" / "generation_task" / "results"
            case_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            write_minimal_presentbench_code_root(code_root)
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 1\n", encoding="utf-8"
            )
            (case_dir / "generation_task").mkdir()
            (case_dir / "generation_task" / "instructions.md").write_text(
                "instructions", encoding="utf-8"
            )
            (case_dir / "generation_task" / "judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (case_dir / "material.md").write_text("material", encoding="utf-8")
            (result_dir / "slides.pdf").write_text("%PDF-1.4\n", encoding="utf-8")
            packs_path = root / "packs.jsonl"
            packs_path.write_text(
                (
                    '{"pack_id":"pack-present","source":"PresentBench",'
                    '"train_examples":[{"example_id":"e","task_input":"make slides",'
                    '"desired_output":{"status":"generated","text":"slides"},'
                    '"materials":[]}],'
                    '"heldout_tasks":[{"task_id":"heldout-1",'
                    '"source_task_id":"education/case1"}]}\n'
                ),
                encoding="utf-8",
            )

            commands, errors = selected_judge_commands(
                packs_path=packs_path,
                code_root=code_root,
                data_root=data_root,
                mode_roots={"prompt_only": result_root},
                python_executable="/python",
                api_type="gemini",
                model="gemini-3-flash-preview",
                retry=5,
                thinking_level=None,
                min_timestamp=None,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(commands), 1)
        self.assertIn("--slides", commands[0])
        self.assertIn(str(result_dir / "slides.pdf"), commands[0])

    def test_selected_judge_commands_all_scored_is_noop_success(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            case_dir = data_root / "education" / "case1"
            result_root = root / "results" / "prompt"
            result_dir = result_root / "education" / "case1" / "generation_task" / "results"
            case_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            write_minimal_presentbench_code_root(code_root)
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 1\n", encoding="utf-8"
            )
            (case_dir / "generation_task").mkdir()
            (case_dir / "generation_task" / "instructions.md").write_text(
                "instructions", encoding="utf-8"
            )
            (case_dir / "generation_task" / "judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (case_dir / "material.md").write_text("material", encoding="utf-8")
            (result_dir / "slides.pdf").write_text("%PDF-1.4\n", encoding="utf-8")
            (result_dir / "gemini-3-flash-preview_2026-05-09_00-00-00_score.yaml").write_text(
                "total:\n  weighted_arithmetic_mean_percent: 100\n",
                encoding="utf-8",
            )
            packs_path = root / "packs.jsonl"
            packs_path.write_text(
                (
                    '{"pack_id":"pack-present","source":"PresentBench",'
                    '"train_examples":[{"example_id":"e","task_input":"make slides",'
                    '"desired_output":{"status":"generated","text":"slides"},'
                    '"materials":[]}],'
                    '"heldout_tasks":[{"task_id":"heldout-1",'
                    '"source_task_id":"education/case1"}]}\n'
                ),
                encoding="utf-8",
            )

            commands, errors = selected_judge_commands(
                packs_path=packs_path,
                code_root=code_root,
                data_root=data_root,
                mode_roots={"prompt_only": result_root},
                python_executable="/python",
                api_type="gemini",
                model="gemini-3-flash-preview",
                retry=5,
                thinking_level=None,
                min_timestamp=None,
            )

        self.assertEqual(commands, [])
        self.assertEqual(errors, [])

    def test_subprocess_env_prepends_presentbench_code_root_to_pythonpath(self) -> None:
        with TemporaryDirectory() as tmp:
            code_root = Path(tmp) / "PresentBench_code"
            code_root.mkdir()
            with patch.dict(os.environ, {"PYTHONPATH": "/existing"}, clear=True):
                env = subprocess_env_with_code_root(code_root)

        self.assertEqual(
            env["PYTHONPATH"],
            f"{code_root.resolve()}{os.pathsep}/existing",
        )

    def test_outer_subprocess_workers_keeps_all_presentbench_outer_serial(self) -> None:
        self.assertEqual(outer_subprocess_workers(all_presentbench=True, max_workers=8), 1)
        self.assertEqual(outer_subprocess_workers(all_presentbench=False, max_workers=8), 8)

    def test_command_manifest_records_argv_and_shell_command(self) -> None:
        manifest = command_manifest(
            [["/python", "judge.py", "--model", "gemini 3"]],
            [],
            ["GENAI_API_KEY is required"],
        )

        self.assertEqual(
            manifest["schema_version"],
            "presentbench-official-judge-commands/v1",
        )
        self.assertEqual(
            manifest["commands"][0]["argv"],
            ["/python", "judge.py", "--model", "gemini 3"],
        )
        self.assertIn("'gemini 3'", manifest["commands"][0]["shell"])
        self.assertEqual(manifest["warnings"], ["GENAI_API_KEY is required"])

    def test_cli_dry_run_renders_all_presentbench_modes_without_credentials(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            prompt_root = root / "prompt"
            skill_root = root / "skill"
            commands_out = root / "commands.json"
            code_root.mkdir()
            data_root.mkdir()
            prompt_root.mkdir()
            skill_root.mkdir()
            (code_root / "judge_all.py").write_text("# judge\n", encoding="utf-8")
            (code_root / "judge.py").write_text("# judge\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/eval/run_presentbench_official_judge.py",
                    "--env-file",
                    str(root / "empty.env"),
                    "--code-root",
                    str(code_root),
                    "--data-root",
                    str(data_root),
                    "--mode-result-root",
                    f"prompt_only={prompt_root}",
                    "--mode-result-root",
                    f"auto_skill={skill_root}",
                    "--dry-run",
                    "--allow-missing-env",
                    "--all-presentbench",
                    "--commands-out",
                    str(commands_out),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                env={k: v for k, v in os.environ.items() if k != "GENAI_API_KEY"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Selected 2 PresentBench official judge command", completed.stdout)
            self.assertNotIn("--agent_name prompt_only", completed.stdout)
            manifest = json.loads(commands_out.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["commands"]), 2)
            self.assertIn("--agent_name", manifest["commands"][0]["argv"])
            self.assertIn("GENAI_API_KEY", "\n".join(manifest["warnings"]))

    def test_cli_expect_commands_fails_on_count_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            prompt_root = root / "prompt"
            code_root.mkdir()
            data_root.mkdir()
            prompt_root.mkdir()
            (code_root / "judge_all.py").write_text("# judge\n", encoding="utf-8")
            (code_root / "judge.py").write_text("# judge\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/eval/run_presentbench_official_judge.py",
                    "--code-root",
                    str(code_root),
                    "--data-root",
                    str(data_root),
                    "--mode-result-root",
                    f"prompt_only={prompt_root}",
                    "--dry-run",
                    "--allow-missing-env",
                    "--all-presentbench",
                    "--expect-commands",
                    "2",
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("expected 2 selected judge commands, got 1", completed.stderr)

    def test_allow_missing_env_is_dry_run_only(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/eval/run_presentbench_official_judge.py",
                "--allow-missing-env",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("only valid with --dry-run", completed.stderr)

    def test_resolve_judge_script_routes_openai_to_in_repo_chat_judge(self) -> None:
        with TemporaryDirectory() as tmp:
            code_root = Path(tmp) / "code"
            code_root.mkdir()
            self.assertEqual(
                resolve_judge_script(code_root=code_root, api_type="gemini"),
                code_root / "judge.py",
            )
            self.assertEqual(
                resolve_judge_script(code_root=code_root, api_type="gemini_inline"),
                code_root / "judge.py",
            )
            self.assertEqual(
                resolve_judge_script(code_root=code_root, api_type="openai"),
                CHAT_JUDGE_SCRIPT,
            )

    def test_validate_preflight_requires_openai_env_unless_dry_run_override(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            result_root = root / "results"
            code_root.mkdir()
            data_root.mkdir()
            result_root.mkdir()
            (code_root / "judge.py").write_text("# judge\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                errors = validate_preflight(
                    code_root=code_root,
                    data_root=data_root,
                    mode_roots={"prompt_only": result_root},
                    api_type="openai",
                    allow_missing_env=False,
                )
                self.assertTrue(
                    any("openai" in err for err in errors),
                    f"expected openai env hint in {errors!r}",
                )

                errors = validate_preflight(
                    code_root=code_root,
                    data_root=data_root,
                    mode_roots={"prompt_only": result_root},
                    api_type="openai",
                    allow_missing_env=True,
                )
                self.assertEqual(errors, [])

            with patch.dict(
                os.environ,
                {"GOOGLE_THIRD_API_URL": "https://x", "GOOGLE_THIRD_API_KEY": "k"},
                clear=True,
            ):
                errors = validate_preflight(
                    code_root=code_root,
                    data_root=data_root,
                    mode_roots={"prompt_only": result_root},
                    api_type="openai",
                    allow_missing_env=False,
                )
                self.assertEqual(errors, [])

    def test_preflight_warnings_records_missing_openai_env_for_dry_run_manifest(
        self,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            warnings = preflight_warnings(api_type="openai", allow_missing_env=True)
            self.assertEqual(len(warnings), 1)
            self.assertIn("openai", warnings[0])
            self.assertEqual(
                preflight_warnings(api_type="openai", allow_missing_env=False),
                [],
            )

        with patch.dict(
            os.environ,
            {"GOOGLE_THIRD_API_URL": "https://x", "GOOGLE_THIRD_API_KEY": "k"},
            clear=True,
        ):
            self.assertEqual(
                preflight_warnings(api_type="openai", allow_missing_env=True),
                [],
            )

    def test_selected_judge_commands_uses_chat_judge_for_openai_api_type(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            case_dir = data_root / "education" / "case1"
            result_root = root / "results" / "prompt"
            result_dir = result_root / "education" / "case1" / "generation_task" / "results"
            case_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            write_minimal_presentbench_code_root(code_root)
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 1\n", encoding="utf-8"
            )
            (case_dir / "generation_task").mkdir()
            (case_dir / "generation_task" / "instructions.md").write_text(
                "instructions", encoding="utf-8"
            )
            (case_dir / "generation_task" / "judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (case_dir / "material.md").write_text("material", encoding="utf-8")
            (result_dir / "slides.pdf").write_text("%PDF-1.4\n", encoding="utf-8")
            packs_path = root / "packs.jsonl"
            packs_path.write_text(
                (
                    '{"pack_id":"pack-present","source":"PresentBench",'
                    '"train_examples":[{"example_id":"e","task_input":"make slides",'
                    '"desired_output":{"status":"generated","text":"slides"},'
                    '"materials":[]}],'
                    '"heldout_tasks":[{"task_id":"heldout-1",'
                    '"source_task_id":"education/case1"}]}\n'
                ),
                encoding="utf-8",
            )

            commands, errors = selected_judge_commands(
                packs_path=packs_path,
                code_root=code_root,
                data_root=data_root,
                mode_roots={"prompt_only": result_root},
                python_executable="/python",
                api_type="openai",
                model="gemini-3-flash-preview",
                retry=5,
                thinking_level=None,
                min_timestamp=None,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][0], "/python")
        self.assertIn(str(CHAT_JUDGE_SCRIPT), commands[0])
        self.assertIn("--api_type", commands[0])
        api_type_idx = commands[0].index("--api_type")
        self.assertEqual(commands[0][api_type_idx + 1], "openai")

    def test_cli_dry_run_renders_openai_per_pack_command(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            case_dir = data_root / "education" / "case1"
            prompt_root = root / "prompt"
            result_dir = prompt_root / "education" / "case1" / "generation_task" / "results"
            commands_out = root / "commands.json"
            packs_path = root / "packs.jsonl"
            case_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            write_minimal_presentbench_code_root(code_root)
            (data_root / "education" / "common_judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (data_root / "education" / "judge_weights.yaml").write_text(
                "total: 1\n", encoding="utf-8"
            )
            (case_dir / "generation_task").mkdir()
            (case_dir / "generation_task" / "instructions.md").write_text(
                "instructions", encoding="utf-8"
            )
            (case_dir / "generation_task" / "judge_prompt.json").write_text(
                "{}", encoding="utf-8"
            )
            (case_dir / "material.md").write_text("material", encoding="utf-8")
            (result_dir / "slides.pdf").write_text("%PDF-1.4\n", encoding="utf-8")
            packs_path.write_text(
                (
                    '{"pack_id":"pack-present","source":"PresentBench",'
                    '"train_examples":[{"example_id":"e","task_input":"make slides",'
                    '"desired_output":{"status":"generated","text":"slides"},'
                    '"materials":[]}],'
                    '"heldout_tasks":[{"task_id":"heldout-1",'
                    '"source_task_id":"education/case1"}]}\n'
                ),
                encoding="utf-8",
            )

            scrubbed_env = {
                k: v
                for k, v in os.environ.items()
                if k not in OPENAI_JUDGE_ENV_KEYS and k != "GENAI_API_KEY"
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/eval/run_presentbench_official_judge.py",
                    "--env-file",
                    str(root / "empty.env"),
                    "--packs",
                    str(packs_path),
                    "--code-root",
                    str(code_root),
                    "--data-root",
                    str(data_root),
                    "--mode-result-root",
                    f"prompt_only={prompt_root}",
                    "--api-type",
                    "openai",
                    "--model",
                    "gemini-3-flash-preview",
                    "--limit-heldout",
                    "2",
                    "--max-workers",
                    "4",
                    "--dry-run",
                    "--allow-missing-env",
                    "--commands-out",
                    str(commands_out),
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                env=scrubbed_env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(commands_out.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["commands"]), 1)
            argv = manifest["commands"][0]["argv"]
            self.assertIn(str(CHAT_JUDGE_SCRIPT), argv)
            self.assertIn("openai", argv)
            self.assertIn(
                "GOOGLE_THIRD_API_URL+GOOGLE_THIRD_API_KEY",
                "\n".join(manifest["warnings"]),
            )

    def test_all_presentbench_rejects_openai_api_type(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            prompt_root = root / "prompt"
            code_root.mkdir()
            data_root.mkdir()
            prompt_root.mkdir()
            (code_root / "judge_all.py").write_text("# judge\n", encoding="utf-8")
            (code_root / "judge.py").write_text("# judge\n", encoding="utf-8")

            scrubbed_env = {
                k: v
                for k, v in os.environ.items()
                if k not in OPENAI_JUDGE_ENV_KEYS and k != "GENAI_API_KEY"
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/eval/run_presentbench_official_judge.py",
                    "--code-root",
                    str(code_root),
                    "--data-root",
                    str(data_root),
                    "--mode-result-root",
                    f"prompt_only={prompt_root}",
                    "--api-type",
                    "openai",
                    "--all-presentbench",
                    "--dry-run",
                    "--allow-missing-env",
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                env=scrubbed_env,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "--all-presentbench is not supported with --api-type 'openai'",
                completed.stderr,
            )


if __name__ == "__main__":
    unittest.main()
