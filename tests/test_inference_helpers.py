from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from app.inference import (
    GenerationRequest,
    InvalidInputError,
    LocalInferenceService,
    make_public_url,
    save_generation_outputs,
)
from app.settings import Settings


class NoRuntimeInferenceService(LocalInferenceService):
    def _ensure_runtime(self):  # type: ignore[no-untyped-def]
        raise AssertionError("runtime should not be initialized for invalid input")


class InferenceHelperTests(unittest.TestCase):
    def test_save_generation_outputs_writes_pngs_and_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            images = [
                Image.new("RGB", (64, 64), color="red"),
                Image.new("RGB", (64, 64), color="blue"),
            ]

            job_id, saved_images, zip_relative_path = save_generation_outputs(
                images=images,
                output_root=output_root,
                job_id="job-123",
            )

            self.assertEqual(job_id, "job-123")
            self.assertEqual(len(saved_images), 2)
            self.assertTrue((output_root / "job-123" / "image-1.png").exists())
            self.assertTrue((output_root / zip_relative_path).exists())

            with zipfile.ZipFile(output_root / zip_relative_path) as archive:
                self.assertEqual(sorted(archive.namelist()), ["image-1.png", "image-2.png"])

    def test_make_public_url_normalizes_separators(self) -> None:
        self.assertEqual(
            make_public_url("job-123\\image-1.png"),
            "/outputs/job-123/image-1.png",
        )

    def test_invalid_input_image_fails_before_runtime_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = replace(
                Settings.from_env(),
                output_dir=Path(temp_dir) / "outputs",
                cache_dir=Path(temp_dir) / ".cache",
            )
            service = NoRuntimeInferenceService(settings)

            with self.assertRaises(InvalidInputError):
                service.generate(
                    GenerationRequest(
                        prompt="portrait",
                        negative_prompt="low quality",
                        face_image_bytes=[b"not an image"],
                        face_positions=["left"],
                        width=512,
                        height=512,
                        steps=1,
                        guidance_scale=7.5,
                        num_images=1,
                        ip_adapter_scale=0.6,
                    )
                )


if __name__ == "__main__":
    unittest.main()
