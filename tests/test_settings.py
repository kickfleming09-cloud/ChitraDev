from __future__ import annotations

from pathlib import Path
import os
import unittest
from unittest.mock import patch

from app.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_from_env_reads_overrides(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        with patch.dict(
            os.environ,
            {
                "HOST": "0.0.0.0",
                "PORT": "9100",
                "OUTPUT_DIR": "tmp-output",
                "CACHE_DIR": "tmp-cache",
                "DEFAULT_WIDTH": "640",
                "DEFAULT_GUIDANCE_SCALE": "6.5",
                "DEFAULT_BATCH_TOTAL_IMAGES": "120",
                "MAX_BATCH_TOTAL_IMAGES": "600",
                "DEFAULT_BATCH_CHUNK_SIZE": "2",
                "MAX_BATCH_CHUNK_SIZE": "8",
                "LOCAL_FILES_ONLY": "true",
            },
            clear=False,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 9100)
        self.assertEqual(settings.output_dir, (project_root / "tmp-output").resolve())
        self.assertEqual(settings.cache_dir, (project_root / "tmp-cache").resolve())
        self.assertEqual(settings.default_width, 640)
        self.assertEqual(settings.default_guidance_scale, 6.5)
        self.assertEqual(settings.default_batch_total_images, 120)
        self.assertEqual(settings.max_batch_total_images, 600)
        self.assertEqual(settings.default_batch_chunk_size, 2)
        self.assertEqual(settings.max_batch_chunk_size, 8)
        self.assertTrue(settings.local_files_only)

    def test_from_env_rejects_invalid_numbers(self) -> None:
        with patch.dict(os.environ, {"PORT": "not-a-port"}, clear=False):
            with self.assertRaisesRegex(ValueError, "PORT must be an integer."):
                Settings.from_env()

    def test_from_env_rejects_invalid_default_dimensions(self) -> None:
        with patch.dict(os.environ, {"DEFAULT_WIDTH": "500"}, clear=False):
            with self.assertRaisesRegex(ValueError, "DEFAULT_WIDTH must be a multiple of 64."):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
