from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
import tempfile
import time
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from app.inference import (
    FaceNotFoundError,
    GeneratedImagesResult,
    GenerationResult,
    GenerationRequest,
    save_generation_outputs,
)
from app.main import create_app, get_inference_service
from app.settings import Settings


def _sample_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


class StubInferenceService:
    def __init__(self, output_dir: Path, error: Exception | None = None) -> None:
        self.output_dir = output_dir
        self.error = error
        self.last_request: GenerationRequest | None = None
        self.requests: list[GenerationRequest] = []

    def generate_images(self, request: GenerationRequest) -> GeneratedImagesResult:
        if self.error is not None:
            raise self.error

        self.last_request = request
        self.requests.append(request)
        images = [
            Image.new("RGB", (request.width, request.height), color="white")
            for _ in range(request.num_images)
        ]
        return GeneratedImagesResult(
            seed=request.seed or 1234,
            images=images,
            warning="Detected 2 faces and used the largest one.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        generated = self.generate_images(request)
        job_id, saved_images, zip_relative_path = save_generation_outputs(
            images=generated.images,
            output_root=self.output_dir,
            job_id=f"stub-{uuid4().hex}",
        )
        return GenerationResult(
            job_id=job_id,
            seed=generated.seed,
            images=saved_images,
            zip_relative_path=zip_relative_path,
            warning=generated.warning,
        )


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base_settings = Settings.from_env()
        self.settings = replace(
            base_settings,
            output_dir=Path(self.temp_dir.name) / "outputs",
            cache_dir=Path(self.temp_dir.name) / ".cache",
        )
        self.settings.ensure_directories()
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generate_returns_download_urls(self) -> None:
        stub_service = StubInferenceService(self.settings.output_dir)
        self.app.dependency_overrides[get_inference_service] = lambda: stub_service

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": "cinematic portrait in a neon city",
                "person1_prompt": "wearing a black suit",
                "width": "512",
                "height": "512",
                "steps": "20",
            },
            files={"face_image": ("face.png", _sample_png_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["images"]), 1)
        self.assertTrue(payload["images"][0]["download_url"].startswith("/outputs/"))
        self.assertTrue(payload["zip_url"].startswith("/outputs/"))
        self.assertEqual(payload["warning"], "Detected 2 faces and used the largest one.")
        self.assertIn("Main subject: wearing a black suit.", payload["resolved_prompt"])

        image_relative_path = payload["images"][0]["download_url"].removeprefix("/outputs/")
        zip_relative_path = payload["zip_url"].removeprefix("/outputs/")
        self.assertTrue((self.settings.output_dir / image_relative_path).exists())
        self.assertTrue((self.settings.output_dir / zip_relative_path).exists())
        self.assertIsNotNone(stub_service.last_request)
        self.assertEqual(len(stub_service.last_request.face_image_bytes), 1)

    def test_generate_accepts_prompt_only(self) -> None:
        stub_service = StubInferenceService(self.settings.output_dir)
        self.app.dependency_overrides[get_inference_service] = lambda: stub_service

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": "a futuristic mountain observatory at sunrise",
                "width": "512",
                "height": "512",
                "steps": "20",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resolved_prompt"], "a futuristic mountain observatory at sunrise.")
        self.assertIsNotNone(stub_service.last_request)
        self.assertEqual(stub_service.last_request.face_image_bytes, [])
        self.assertEqual(stub_service.last_request.face_positions, [])
        self.assertNotIn("merged faces", stub_service.last_request.negative_prompt)

    def test_generate_accepts_structured_logo_prompt(self) -> None:
        stub_service = StubInferenceService(self.settings.output_dir)
        self.app.dependency_overrides[get_inference_service] = lambda: stub_service

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": (
                    "Create a premium, high-speed logo for an F1-inspired racing company named [Company Name]. "
                    "The logo should feel modern, aggressive, aerodynamic, and elite.\n"
                    "Style: futuristic motorsport, luxury racing brand, clean vector logo, minimal but powerful.\n"
                    "Colors: black, red, silver, white, and carbon-fiber accents.\n"
                    "Typography: bold, italicized, angular, racing-inspired font.\n"
                    "Icon idea: abstract F1 car nose, speed trail, racing flag detail, or aerodynamic wing shape.\n"
                    "Avoid copying the official Formula 1 logo. Make it original, premium, and brand-ready.\n"
                    "Output: clean vector-style logo, transparent background, high contrast, professional branding, no mockup, no extra text."
                ),
                "width": "512",
                "height": "512",
                "steps": "20",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("futuristic motorsport", payload["resolved_prompt"])
        self.assertIn("clean vector-style logo", payload["resolved_prompt"])
        self.assertIn("placeholder text", payload["warning"])
        self.assertIsNotNone(stub_service.last_request)
        self.assertIn("official Formula 1 logo", stub_service.last_request.negative_prompt)
        self.assertIn("mockup", stub_service.last_request.negative_prompt)

    def test_generate_ignores_empty_second_file_field(self) -> None:
        stub_service = StubInferenceService(self.settings.output_dir)
        self.app.dependency_overrides[get_inference_service] = lambda: stub_service

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": "studio portrait",
                "person1_prompt": "wearing a denim jacket",
            },
            files={
                "face_image": ("face.png", _sample_png_bytes(), "image/png"),
                "face_image_2": ("", b"", "application/octet-stream"),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(stub_service.last_request)
        self.assertEqual(len(stub_service.last_request.face_image_bytes), 1)

    def test_index_renders_form(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("ChitraDev", response.text)
        self.assertIn("Generate images", response.text)
        self.assertIn("Open batch creator", response.text)
        self.assertIn("Person 2 face image", response.text)

    def test_batch_page_renders_separate_creator(self) -> None:
        response = self.client.get("/batch")

        self.assertEqual(response.status_code, 200)
        self.assertIn("ChitraDev Batch Creator", response.text)
        self.assertIn("Total outputs", response.text)

    def test_batch_start_completes_and_writes_zip(self) -> None:
        stub_service = StubInferenceService(self.settings.output_dir)
        self.app.dependency_overrides[get_inference_service] = lambda: stub_service

        response = self.client.post(
            "/api/batch",
            data={
                "prompt": "a clean vector logo for Apex Velocity",
                "total_images": "3",
                "chunk_size": "2",
                "width": "512",
                "height": "512",
                "steps": "20",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        status_payload = self._wait_for_batch(payload["job_id"])

        self.assertEqual(status_payload["status"], "completed")
        self.assertEqual(status_payload["completed_images"], 3)
        self.assertEqual(len(status_payload["preview_images"]), 3)
        self.assertTrue(status_payload["zip_url"].startswith("/outputs/"))
        self.assertEqual([request.num_images for request in stub_service.requests], [2, 1])

        zip_relative_path = status_payload["zip_url"].removeprefix("/outputs/")
        self.assertTrue((self.settings.output_dir / zip_relative_path).exists())
        image_paths = [
            self.settings.output_dir / item["download_url"].removeprefix("/outputs/")
            for item in status_payload["preview_images"]
        ]
        self.assertTrue(all(path.exists() for path in image_paths))

    def test_batch_rejects_total_over_batch_limit(self) -> None:
        response = self.client.post(
            "/api/batch",
            data={
                "prompt": "batch prompt",
                "total_images": str(self.settings.max_batch_total_images + 1),
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("total_images must be between", response.json()["detail"])

    def test_health_returns_status_without_loading_model(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_loaded": False})

    def test_generate_rejects_empty_face_upload(self) -> None:
        self.app.dependency_overrides[get_inference_service] = lambda: StubInferenceService(
            self.settings.output_dir
        )

        response = self.client.post(
            "/api/generate",
            data={"prompt": "portrait"},
            files={"face_image": ("face.png", b"", "image/png")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Person 1 face image was empty.")

    def test_generate_rejects_person_controls_without_face(self) -> None:
        self.app.dependency_overrides[get_inference_service] = lambda: StubInferenceService(
            self.settings.output_dir
        )

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": "portrait",
                "person1_prompt": "wearing a denim jacket",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            "Upload a face image to use FaceID person controls, or put the full description in the main prompt.",
        )

    def test_generate_rejects_second_face_without_first_face(self) -> None:
        self.app.dependency_overrides[get_inference_service] = lambda: StubInferenceService(
            self.settings.output_dir
        )

        response = self.client.post(
            "/api/generate",
            data={"prompt": "portrait"},
            files={"face_image_2": ("face2.png", _sample_png_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Upload Person 1 face image before adding Person 2.")

    def test_generate_rejects_oversized_face_upload(self) -> None:
        settings = replace(self.settings, max_face_image_mb=1)
        app = create_app(settings)
        app.dependency_overrides[get_inference_service] = lambda: StubInferenceService(
            settings.output_dir
        )
        client = TestClient(app)

        response = client.post(
            "/api/generate",
            data={"prompt": "portrait"},
            files={"face_image": ("face.png", b"x" * ((1024 * 1024) + 1), "image/png")},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "Person 1 face image must be 1 MB or smaller.")

    def test_generate_rejects_non_multiple_of_64_width(self) -> None:
        self.app.dependency_overrides[get_inference_service] = lambda: StubInferenceService(
            self.settings.output_dir
        )

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": "portrait",
                "width": "510",
                "height": "512",
            },
            files={"face_image": ("face.png", _sample_png_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "width must be a multiple of 64.")

    def test_generate_maps_face_errors(self) -> None:
        self.app.dependency_overrides[get_inference_service] = lambda: StubInferenceService(
            self.settings.output_dir,
            error=FaceNotFoundError("No face was detected in the uploaded image."),
        )

        response = self.client.post(
            "/api/generate",
            data={"prompt": "portrait"},
            files={"face_image": ("face.png", _sample_png_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "No face was detected in the uploaded image.")

    def test_generate_accepts_two_faces_and_builds_dual_prompt(self) -> None:
        stub_service = StubInferenceService(self.settings.output_dir)
        self.app.dependency_overrides[get_inference_service] = lambda: stub_service

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": "rainy cyberpunk street",
                "person1_prompt": "holding an umbrella and looking at Person 2",
                "person2_prompt": "smiling back while carrying a coffee",
                "interaction_prompt": "Person 1 is offering shelter to Person 2 under the umbrella",
                "person1_position": "left",
                "person2_position": "right",
            },
            files={
                "face_image": ("face1.png", _sample_png_bytes(), "image/png"),
                "face_image_2": ("face2.png", _sample_png_bytes(), "image/png"),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(stub_service.last_request)
        self.assertEqual(len(stub_service.last_request.face_image_bytes), 2)
        self.assertEqual(stub_service.last_request.face_positions, ["left", "right"])
        self.assertIn("Two distinct people in the same scene.", stub_service.last_request.prompt)
        self.assertIn("Interaction between Person 1 and Person 2", stub_service.last_request.prompt)
        self.assertIn("merged faces", stub_service.last_request.negative_prompt)

    def test_generate_requires_second_face_for_person2_controls(self) -> None:
        self.app.dependency_overrides[get_inference_service] = lambda: StubInferenceService(
            self.settings.output_dir
        )

        response = self.client.post(
            "/api/generate",
            data={
                "prompt": "portrait",
                "person2_prompt": "waving",
            },
            files={"face_image": ("face.png", _sample_png_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            "Upload a second face image to use Person 2 or interaction controls.",
        )

    def _wait_for_batch(self, job_id: str) -> dict[str, object]:
        for _ in range(50):
            response = self.client.get(f"/api/batch/{job_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"completed", "failed", "canceled"}:
                return payload
            time.sleep(0.05)
        self.fail(f"Batch job {job_id} did not finish.")


if __name__ == "__main__":
    unittest.main()
