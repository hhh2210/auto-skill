from __future__ import annotations

import base64
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from auto_skill.multimodal_materials import (
    build_material_digest_prompt,
    encode_image_data_url,
    image_content_part,
    parse_page_spec,
)


class MultimodalMaterialsTests(unittest.TestCase):
    def test_parse_page_spec_accepts_ranges_and_commas(self) -> None:
        self.assertEqual(parse_page_spec("1,3-5,2"), (1, 2, 3, 4, 5))

    def test_parse_page_spec_rejects_empty_or_invalid_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_page_spec(" , ")
        with self.assertRaisesRegex(ValueError, "invalid page range"):
            parse_page_spec("3-2")
        with self.assertRaisesRegex(ValueError, "invalid page number"):
            parse_page_spec("0")

    def test_image_content_part_uses_openai_compatible_image_url(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image = Path(tmp_dir) / "page.png"
            image.write_bytes(b"png-bytes")

            part = image_content_part(image)

        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(
            part["image_url"]["url"].split(",", 1)[1],
            base64.b64encode(b"png-bytes").decode("ascii"),
        )

    def test_encode_image_data_url_defaults_unknown_type_to_png(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image = Path(tmp_dir) / "page.unknown"
            image.write_bytes(b"bytes")

            url = encode_image_data_url(image)

        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_build_material_digest_prompt_names_visual_digest_schema(self) -> None:
        prompt = build_material_digest_prompt(
            task_id="task-1",
            task_input="Create slides.",
            material_paths=["data/PresentBench_repo/case/material.pdf"],
        )

        self.assertIn("structured visual/material digest", prompt)
        self.assertIn("visual_elements", prompt)
        self.assertIn("source_fidelity_constraints", prompt)
        self.assertIn("data/PresentBench_repo/case/material.pdf", prompt)


if __name__ == "__main__":
    unittest.main()

