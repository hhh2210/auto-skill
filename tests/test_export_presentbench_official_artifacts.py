from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.eval.export_presentbench_official_artifacts import (
    export_rows,
    mode_result_roots,
    split_slide_pages,
)


class ExportPresentBenchOfficialArtifactsTests(unittest.TestCase):
    def test_mode_result_roots_requires_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "MODE=PATH"):
            mode_result_roots(["bad"])
        with self.assertRaisesRegex(ValueError, "non-empty MODE=PATH"):
            mode_result_roots(["auto_skill=  "])

        self.assertEqual(
            mode_result_roots([" auto_skill = results/auto "]),
            {"auto_skill": Path("results/auto")},
        )

    def test_split_slide_pages_uses_slide_headings(self) -> None:
        pages = split_slide_pages("# Slide 1: A\nbody\n\nSlide 2: B\nbody", max_pages=10)

        self.assertEqual(len(pages), 2)
        self.assertIn("Slide 1", pages[0][0])

    def test_export_rows_writes_pdf_to_presentbench_result_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            packs = [
                {
                    "pack_id": "pack-present",
                    "source": "PresentBench",
                    "heldout_tasks": [
                        {
                            "task_id": "pack-present::heldout::0",
                            "source_task_id": "education/demo",
                        }
                    ],
                }
            ]
            rows = [
                {
                    "pack_id": "pack-present",
                    "task_id": "pack-present::heldout::0",
                    "mode": "auto_skill",
                    "status": "success",
                    "overall_score": 9,
                    "generation": {"text": "# Slide 1: Demo\nHello", "model": "qwen"},
                }
            ]

            exported = export_rows(
                packs=packs,
                eval_rows=rows,
                result_roots={"auto_skill": tmp / "results" / "auto_skill"},
                max_pages=10,
                overwrite=False,
            )

            pdf = (
                tmp
                / "results"
                / "auto_skill"
                / "education/demo/generation_task/results/slides.pdf"
            )
            self.assertEqual(len(exported), 1)
            self.assertTrue(pdf.exists())
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-1.4"))
            self.assertTrue((pdf.parent / "auto_skill_export_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
