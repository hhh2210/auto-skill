from __future__ import annotations

import unittest

from auto_skill.coding_style_diagnostic import diagnose_coding_style_source


def fragment(author: str, index: int, code: str, *, repo: str = "sample/repo") -> dict[str, object]:
    return {
        "source_id": f"{author}-{index}",
        "author_id": author,
        "repo": repo,
        "language": "python",
        "path": f"src/{author}_{index}.py",
        "date": f"2021-01-{index + 1:02d}",
        "code": code,
        "license_ok": True,
    }


class CodingStyleDiagnosticTests(unittest.TestCase):
    def test_go_for_separable_pre_ai_personal_developer_fragments(self) -> None:
        styles = {
            "author_a": "def f(x):\n    y = x + 1\n    return y\n",
            "author_b": (
                "def calculateValue(inputValue):\n"
                "    interimValue = inputValue + 1\n"
                "    return interimValue\n"
            ),
            "author_c": (
                "def f(x):\n"
                "\t# compact tabbed style\n"
                "\tif x:\n"
                "\t\treturn x+1\n"
                "\treturn 0\n"
            ),
        }
        rows = [
            fragment(author, index, code)
            for author, code in styles.items()
            for index in range(3)
        ]

        report = diagnose_coding_style_source(
            rows,
            source_name="fixture",
            min_train=2,
            min_heldout=1,
            min_lines=2,
        )

        self.assertEqual(report["decision"], "go")
        self.assertEqual(
            report["recommended_role"],
            "personal_developer_style_candidate",
        )
        self.assertEqual(report["candidate_author_count"], 3)
        self.assertGreaterEqual(
            report["source_pool_baseline"]["mean_margin_vs_source_pool"],
            0.03,
        )

    def test_project_style_guide_is_control_not_personal_source(self) -> None:
        report = diagnose_coding_style_source(
            [
                {
                    "source_id": "linux-style",
                    "code": "int f(void)\n{\n\treturn 0;\n}\n",
                    "date": "2021-01-01",
                }
            ],
            source_name="linux-style-doc",
            source_kind="project_style_guide",
            min_train=2,
            min_heldout=1,
            min_lines=2,
        )

        self.assertEqual(report["decision"], "diagnostic_only")
        self.assertEqual(report["recommended_role"], "project_style_control")
        self.assertIn("project_style_not_personal_author_style", report["caveats"])

    def test_drop_when_no_clean_pre_ai_author_fragments(self) -> None:
        report = diagnose_coding_style_source(
            [
                {
                    "source_id": "generated",
                    "author_id": "author_a",
                    "repo": "sample/repo",
                    "language": "python",
                    "path": "generated/file.py",
                    "date": "2024-01-01",
                    "code": "print('late')\n",
                    "is_generated": True,
                }
            ],
            source_name="bad",
            min_train=2,
            min_heldout=1,
            min_lines=1,
        )

        self.assertEqual(report["decision"], "drop")
        self.assertIn("post_ai_cutoff", report["rejected_fragment_reasons"])
        self.assertIn("generated_code", report["rejected_fragment_reasons"])


if __name__ == "__main__":
    unittest.main()
