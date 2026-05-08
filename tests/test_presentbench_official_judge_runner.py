from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.eval.run_presentbench_official_judge import (
    build_judge_all_command,
    parse_mode_result_roots,
    subprocess_env_with_code_root,
    validate_preflight,
)


class PresentBenchOfficialJudgeRunnerTests(unittest.TestCase):
    def test_parse_mode_result_roots_requires_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "MODE=PATH"):
            parse_mode_result_roots(["/tmp/results"])

    def test_parse_mode_result_roots_maps_repeated_values(self) -> None:
        roots = parse_mode_result_roots(
            ["prompt_only=/tmp/prompt", "auto_skill=/tmp/skill"]
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

    def test_cli_dry_run_renders_default_modes_without_credentials(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_root = root / "code"
            data_root = root / "data"
            prompt_root = root / "prompt"
            skill_root = root / "skill"
            code_root.mkdir()
            data_root.mkdir()
            prompt_root.mkdir()
            skill_root.mkdir()
            (code_root / "judge_all.py").write_text("# judge\n", encoding="utf-8")

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
                    "--mode-result-root",
                    f"auto_skill={skill_root}",
                    "--dry-run",
                    "--allow-missing-env",
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--agent_name prompt_only", completed.stdout)
        self.assertIn("--agent_name auto_skill", completed.stdout)

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


if __name__ == "__main__":
    unittest.main()
