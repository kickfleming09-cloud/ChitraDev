from __future__ import annotations

from dataclasses import dataclass
import io
import importlib
import os
from pathlib import Path
import random
import threading
from typing import Sequence
from uuid import uuid4
import zipfile

from PIL import Image, UnidentifiedImageError

from .settings import Settings


FACE_EMBEDDING_DIM = 512


class GenerationError(Exception):
    """Base exception for generation failures."""


class InvalidInputError(GenerationError):
    """Raised when the provided input cannot be processed."""


class FaceNotFoundError(GenerationError):
    """Raised when no usable face was found in the reference image."""


class ModelInitializationError(GenerationError):
    """Raised when the diffusion runtime could not be created."""


class OutOfMemoryGenerationError(GenerationError):
    """Raised when the model cannot fit the requested generation settings."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str
    face_image_bytes: list[bytes]
    face_positions: list[str]
    width: int
    height: int
    steps: int
    guidance_scale: float
    num_images: int
    ip_adapter_scale: float
    seed: int | None = None


@dataclass(frozen=True)
class SavedImage:
    file_name: str
    relative_path: str


@dataclass(frozen=True)
class GenerationResult:
    job_id: str
    seed: int
    images: list[SavedImage]
    zip_relative_path: str
    warning: str | None = None


@dataclass(frozen=True)
class GeneratedImagesResult:
    seed: int
    images: list[Image.Image]
    warning: str | None = None


@dataclass(frozen=True)
class RuntimeContext:
    pipeline: object
    face_analyzer: object
    torch: object
    cv2: object
    numpy: object
    device: str
    torch_dtype: object
    runtime_warning: str | None = None


def make_public_url(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    return f"/outputs/{normalized}"


def save_generation_outputs(
    images: Sequence[Image.Image],
    output_root: Path,
    job_id: str | None = None,
) -> tuple[str, list[SavedImage], str]:
    actual_job_id = job_id or uuid4().hex
    job_dir = output_root / actual_job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    saved_images: list[SavedImage] = []
    for index, image in enumerate(images, start=1):
        file_name = f"image-{index}.png"
        image_path = job_dir / file_name
        image.save(image_path, format="PNG")
        saved_images.append(
            SavedImage(
                file_name=file_name,
                relative_path=f"{actual_job_id}/{file_name}",
            )
        )

    zip_path = job_dir / "results.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for saved_image in saved_images:
            archive.write(job_dir / saved_image.file_name, arcname=saved_image.file_name)

    return actual_job_id, saved_images, f"{actual_job_id}/{zip_path.name}"


class LocalInferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._runtime: RuntimeContext | None = None
        self._generation_lock = threading.Lock()

    @property
    def runtime_loaded(self) -> bool:
        return self._runtime is not None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        generated = self.generate_images(request)

        job_id, saved_images, zip_relative_path = save_generation_outputs(
            images=generated.images,
            output_root=self.settings.output_dir,
        )

        return GenerationResult(
            job_id=job_id,
            seed=generated.seed,
            images=saved_images,
            zip_relative_path=zip_relative_path,
            warning=generated.warning,
        )

    def generate_images(self, request: GenerationRequest) -> GeneratedImagesResult:
        if len(request.face_image_bytes) != len(request.face_positions):
            raise InvalidInputError("Face image and position counts must match.")

        source_images = [self._decode_input_image(image_bytes) for image_bytes in request.face_image_bytes]
        do_classifier_free_guidance = request.guidance_scale > 1.0

        with self._generation_lock:
            runtime = self._ensure_runtime()
            if source_images:
                id_embeds, face_warning = self._prepare_face_embeddings(
                    runtime,
                    source_images,
                    do_classifier_free_guidance,
                )
                ip_adapter_masks = self._build_ip_adapter_masks(
                    runtime=runtime,
                    width=request.width,
                    height=request.height,
                    positions=request.face_positions,
                )
            else:
                id_embeds = self._build_neutral_face_embeddings(runtime, do_classifier_free_guidance)
                face_warning = None
                ip_adapter_masks = None

            seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
            generator = runtime.torch.Generator(device="cpu").manual_seed(seed)

            pipeline = runtime.pipeline
            if not source_images:
                pipeline.set_ip_adapter_scale(0.0)
            elif len(request.face_positions) >= 2:
                pipeline.set_ip_adapter_scale([[request.ip_adapter_scale] * len(request.face_positions)])
            else:
                pipeline.set_ip_adapter_scale(request.ip_adapter_scale)

            try:
                with runtime.torch.inference_mode():
                    images = pipeline(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        ip_adapter_image_embeds=[id_embeds],
                        num_images_per_prompt=request.num_images,
                        num_inference_steps=request.steps,
                        guidance_scale=request.guidance_scale,
                        width=request.width,
                        height=request.height,
                        generator=generator,
                        cross_attention_kwargs=(
                            {"ip_adapter_masks": ip_adapter_masks}
                            if ip_adapter_masks is not None
                            else None
                        ),
                    ).images
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise OutOfMemoryGenerationError(
                        "The requested generation ran out of GPU memory. "
                        "Try a smaller size, fewer images, or fewer steps."
                    ) from exc
                raise GenerationError(f"Image generation failed: {exc}") from exc
            finally:
                if runtime.device == "cuda":
                    runtime.torch.cuda.empty_cache()

            warnings = [warning for warning in [runtime.runtime_warning, face_warning] if warning]
            combined_warning = " ".join(warnings) if warnings else None

            return GeneratedImagesResult(
                seed=seed,
                images=images,
                warning=combined_warning,
            )

    def _ensure_runtime(self) -> RuntimeContext:
        if self._runtime is not None:
            return self._runtime

        self.settings.ensure_directories()
        os.environ.setdefault("HF_HOME", str(self.settings.cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(self.settings.cache_dir / "hub"))
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

        try:
            torch = importlib.import_module("torch")
            cv2 = importlib.import_module("cv2")
            numpy = importlib.import_module("numpy")
            onnxruntime = importlib.import_module("onnxruntime")
            diffusers = importlib.import_module("diffusers")
            face_analysis_module = importlib.import_module("insightface.app")
            if self.settings.hf_token:
                huggingface_hub = importlib.import_module("huggingface_hub")
                huggingface_hub.login(
                    token=self.settings.hf_token,
                    add_to_git_credential=False,
                )
        except ImportError as exc:
            raise ModelInitializationError(
                "Missing runtime dependencies. Create the Python 3.11 venv, install "
                "the required packages, and try again."
            ) from exc

        available_providers = set(onnxruntime.get_available_providers())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        runtime_warning = None
        if device != "cuda":
            runtime_warning = (
                "CUDA was not available, so generation is running on CPU and may be very slow."
            )

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        hub_kwargs = {
            "cache_dir": str(self.settings.cache_dir),
            "local_files_only": self.settings.local_files_only,
        }

        try:
            pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
                self.settings.base_model,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                **hub_kwargs,
            )
            pipeline.scheduler = diffusers.DDIMScheduler.from_config(pipeline.scheduler.config)

            if device == "cuda":
                pipeline.enable_vae_slicing()
            else:
                pipeline.to("cpu")

            try:
                pipeline.load_ip_adapter(
                    self.settings.ip_adapter_source,
                    subfolder=None,
                    weight_name=self.settings.ip_adapter_weight,
                    image_encoder_folder=None,
                    **hub_kwargs,
                )
            except TypeError:
                pipeline.load_ip_adapter(
                    self.settings.ip_adapter_source,
                    subfolder=None,
                    weight_name=self.settings.ip_adapter_weight,
                    image_encoder_folder=None,
                )

            if device == "cuda":
                pipeline.enable_model_cpu_offload()
            pipeline.set_progress_bar_config(disable=True)
        except Exception as exc:
            raise ModelInitializationError(
                "Model loading failed. Check your internet connection or point "
                "SD_BASE_MODEL and IP_ADAPTER_SOURCE to local model folders. "
                f"Root cause: {exc}"
            ) from exc

        providers: list[str] = []
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        try:
            face_analyzer = face_analysis_module.FaceAnalysis(
                name=self.settings.insightface_model_name,
                providers=providers,
                root=str(self.settings.cache_dir),
            )
            face_analyzer.prepare(
                ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                det_size=(
                    self.settings.insightface_det_size,
                    self.settings.insightface_det_size,
                ),
            )
        except Exception as exc:
            raise ModelInitializationError(
                "InsightFace failed to initialize. Verify that insightface and "
                "onnxruntime-gpu are installed correctly."
            ) from exc

        self._runtime = RuntimeContext(
            pipeline=pipeline,
            face_analyzer=face_analyzer,
            torch=torch,
            cv2=cv2,
            numpy=numpy,
            device=device,
            torch_dtype=torch_dtype,
            runtime_warning=runtime_warning,
        )
        return self._runtime

    def _decode_input_image(self, face_image_bytes: bytes) -> Image.Image:
        if not face_image_bytes:
            raise InvalidInputError("Please upload a face image before generating.")

        try:
            return Image.open(io.BytesIO(face_image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidInputError("The uploaded face file was not a valid image.") from exc

    def _prepare_face_embeddings(
        self,
        runtime: RuntimeContext,
        images: list[Image.Image],
        do_classifier_free_guidance: bool,
    ) -> tuple[object, str | None]:
        positive_embeds: list[object] = []
        warnings: list[str] = []

        for index, image in enumerate(images, start=1):
            cv_image = runtime.cv2.cvtColor(runtime.numpy.asarray(image), runtime.cv2.COLOR_RGB2BGR)
            faces = runtime.face_analyzer.get(cv_image)
            if not faces:
                raise FaceNotFoundError(f"No face was detected in face image {index}.")

            selected_face = max(
                faces,
                key=lambda face: max(face.bbox[2] - face.bbox[0], 1)
                * max(face.bbox[3] - face.bbox[1], 1),
            )
            if len(faces) > 1:
                warnings.append(f"Face image {index}: detected {len(faces)} faces and used the largest one.")

            positive_embeds.append(runtime.torch.from_numpy(selected_face.normed_embedding).unsqueeze(0))

        stacked_embeds = runtime.torch.stack(positive_embeds, dim=1)
        if do_classifier_free_guidance:
            negative_embeds = runtime.torch.zeros_like(stacked_embeds)
            stacked_embeds = runtime.torch.cat([negative_embeds, stacked_embeds], dim=0)

        id_embeds = stacked_embeds.to(
            dtype=runtime.torch_dtype,
            device=runtime.device,
        )
        return id_embeds, " ".join(warnings) if warnings else None

    def _build_neutral_face_embeddings(
        self,
        runtime: RuntimeContext,
        do_classifier_free_guidance: bool,
    ) -> object:
        batch_size = 2 if do_classifier_free_guidance else 1
        return runtime.torch.zeros(
            (batch_size, 1, FACE_EMBEDDING_DIM),
            dtype=runtime.torch_dtype,
            device=runtime.device,
        )

    def _build_ip_adapter_masks(
        self,
        runtime: RuntimeContext,
        width: int,
        height: int,
        positions: list[str],
    ) -> list[object] | None:
        if len(positions) < 2:
            return None

        overlap = max(width // 10, 32)
        mask_arrays: list[object] = []
        for position in positions:
            mask = runtime.numpy.zeros((height, width), dtype=runtime.numpy.float32)
            if position == "left":
                mask[:, : min(width, (width // 2) + overlap)] = 1.0
            elif position == "right":
                mask[:, max(0, (width // 2) - overlap) :] = 1.0
            else:
                mask[:, :] = 1.0
            mask_arrays.append(mask)

        mask_tensor = runtime.torch.from_numpy(runtime.numpy.stack(mask_arrays, axis=0)).unsqueeze(0)
        return [mask_tensor.to(device=runtime.device, dtype=runtime.torch_dtype)]
