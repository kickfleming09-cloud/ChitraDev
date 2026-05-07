from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .batch import BatchGenerationRequest, BatchJobManager
from .inference import (
    FaceNotFoundError,
    GenerationError,
    GenerationRequest,
    InvalidInputError,
    LocalInferenceService,
    ModelInitializationError,
    OutOfMemoryGenerationError,
    make_public_url,
)
from .prompting import (
    FacePrompt,
    compose_generation_prompt,
    compose_negative_prompt,
    compose_prompt_warning,
)
from .settings import Settings


MIN_DIMENSION = 256
MAX_SEED = 2**32 - 1


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_inference_service(request: Request) -> LocalInferenceService:
    return request.app.state.service_factory()


def get_batch_manager(request: Request) -> BatchJobManager:
    return request.app.state.batch_manager


def _has_uploaded_file(upload: UploadFile | None) -> bool:
    return bool(upload and upload.filename and upload.filename.strip())


async def _read_upload_bytes(upload: UploadFile, label: str, settings: Settings) -> bytes:
    contents = await upload.read()
    if not contents:
        raise HTTPException(status_code=400, detail=f"{label} was empty.")
    if len(contents) > settings.max_face_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} must be {settings.max_face_image_mb} MB or smaller.",
        )
    return contents


def _require_prompt(prompt: str) -> str:
    value = prompt.strip()
    if not value:
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")
    return value


def _resolve_dimension(value: int | None, field_name: str, default: int, maximum: int) -> int:
    actual = value if value is not None else default
    if actual < MIN_DIMENSION or actual > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be between {MIN_DIMENSION} and {maximum}.",
        )
    if actual % 64 != 0:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a multiple of 64.",
        )
    return actual


def _resolve_steps(value: int | None, maximum: int, default: int) -> int:
    actual = value if value is not None else default
    if actual < 1 or actual > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"steps must be between 1 and {maximum}.",
        )
    return actual


def _resolve_guidance(value: float | None, maximum: float, default: float) -> float:
    actual = value if value is not None else default
    if actual <= 0 or actual > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"guidance_scale must be greater than 0 and no more than {maximum}.",
        )
    return actual


def _resolve_num_images(value: int | None, maximum: int, default: int) -> int:
    actual = value if value is not None else default
    if actual < 1 or actual > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"num_images must be between 1 and {maximum}.",
        )
    return actual


def _resolve_batch_total(value: int | None, maximum: int, default: int) -> int:
    actual = value if value is not None else default
    if actual < 1 or actual > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"total_images must be between 1 and {maximum}.",
        )
    return actual


def _resolve_batch_chunk_size(value: int | None, maximum: int, default: int, total: int) -> int:
    actual = value if value is not None else default
    if actual < 1 or actual > maximum:
        raise HTTPException(
            status_code=422,
            detail=f"chunk_size must be between 1 and {maximum}.",
        )
    if actual > total:
        raise HTTPException(
            status_code=422,
            detail="chunk_size must be no more than total_images.",
        )
    return actual


def _resolve_scale(value: float | None, default: float) -> float:
    actual = value if value is not None else default
    if actual < 0 or actual > 1.5:
        raise HTTPException(
            status_code=422,
            detail="ip_adapter_scale must be between 0 and 1.5.",
        )
    return actual


def _resolve_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 0 or value > MAX_SEED:
        raise HTTPException(
            status_code=422,
            detail=f"seed must be between 0 and {MAX_SEED}.",
        )
    return value


def _resolve_position(value: str | None, default: str) -> str:
    actual = (value or default).strip().lower()
    if actual not in {"left", "right"}:
        raise HTTPException(status_code=422, detail="Positions must be either left or right.")
    return actual


def create_app(settings: Settings | None = None) -> FastAPI:
    actual_settings = settings or Settings.from_env()
    actual_settings.ensure_directories()

    templates = Jinja2Templates(directory=str(actual_settings.template_dir))
    app = FastAPI(title="ChitraDev")
    app.state.settings = actual_settings
    app.state.batch_manager = BatchJobManager(actual_settings)

    @lru_cache(maxsize=1)
    def _service_factory() -> LocalInferenceService:
        return LocalInferenceService(actual_settings)

    app.state.service_factory = _service_factory
    app.mount(
        "/outputs",
        StaticFiles(directory=str(actual_settings.output_dir)),
        name="outputs",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "defaults": {
                    "negative_prompt": settings.default_negative_prompt,
                    "width": settings.default_width,
                    "height": settings.default_height,
                    "steps": settings.default_steps,
                    "guidance_scale": settings.default_guidance_scale,
                    "ip_adapter_scale": settings.default_ip_adapter_scale,
                    "num_images": settings.default_num_images,
                    "person1_position": "left",
                    "person2_position": "right",
                },
                "limits": {
                    "max_width": settings.max_width,
                    "max_height": settings.max_height,
                    "max_steps": settings.max_steps,
                    "max_num_images": settings.max_num_images,
                    "max_guidance_scale": settings.max_guidance_scale,
                },
            },
        )

    @app.get("/batch", response_class=HTMLResponse)
    async def batch_window(
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="batch.html",
            context={
                "request": request,
                "defaults": {
                    "negative_prompt": settings.default_negative_prompt,
                    "width": settings.default_width,
                    "height": settings.default_height,
                    "steps": settings.default_steps,
                    "guidance_scale": settings.default_guidance_scale,
                    "total_images": settings.default_batch_total_images,
                    "chunk_size": settings.default_batch_chunk_size,
                },
                "limits": {
                    "max_width": settings.max_width,
                    "max_height": settings.max_height,
                    "max_steps": settings.max_steps,
                    "max_guidance_scale": settings.max_guidance_scale,
                    "max_batch_total_images": settings.max_batch_total_images,
                    "max_batch_chunk_size": settings.max_batch_chunk_size,
                },
            },
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        service = request.app.state.service_factory()
        return {
            "status": "ok",
            "model_loaded": service.runtime_loaded,
        }

    @app.post("/api/generate")
    async def generate(
        prompt: str = Form(...),
        face_image: UploadFile | None = File(None),
        face_image_2: UploadFile | None = File(None),
        negative_prompt: str | None = Form(None),
        person1_prompt: str | None = Form(None),
        person2_prompt: str | None = Form(None),
        interaction_prompt: str | None = Form(None),
        person1_position: str | None = Form(None),
        person2_position: str | None = Form(None),
        width: int | None = Form(None),
        height: int | None = Form(None),
        steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        seed: int | None = Form(None),
        num_images: int | None = Form(None),
        ip_adapter_scale: float | None = Form(None),
        settings: Settings = Depends(get_settings),
        service: LocalInferenceService = Depends(get_inference_service),
    ) -> dict[str, object]:
        has_first_face = _has_uploaded_file(face_image)
        has_second_face_upload = _has_uploaded_file(face_image_2)
        has_person1_controls = bool(person1_prompt and person1_prompt.strip())
        has_person2_controls = bool(person2_prompt and person2_prompt.strip())
        has_interaction_controls = bool(interaction_prompt and interaction_prompt.strip())

        if not has_first_face and has_second_face_upload:
            raise HTTPException(
                status_code=422,
                detail="Upload Person 1 face image before adding Person 2.",
            )
        if not has_first_face and (has_person1_controls or has_person2_controls or has_interaction_controls):
            raise HTTPException(
                status_code=422,
                detail="Upload a face image to use FaceID person controls, or put the full description in the main prompt.",
            )
        if has_first_face and face_image and face_image.content_type and not face_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="The uploaded face file must be an image.")
        if has_second_face_upload and face_image_2 and face_image_2.content_type and not face_image_2.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="The second face file must be an image.")

        face_image_bytes = (
            await _read_upload_bytes(face_image, "Person 1 face image", settings)
            if has_first_face and face_image
            else b""
        )
        second_face_bytes = (
            await _read_upload_bytes(face_image_2, "Person 2 face image", settings)
            if has_second_face_upload and face_image_2
            else b""
        )

        has_first_face_bytes = bool(face_image_bytes)
        has_second_face = bool(second_face_bytes)
        person1_position_value = _resolve_position(person1_position, "left")
        person2_position_value = _resolve_position(person2_position, "right")

        if has_second_face and person1_position_value == person2_position_value:
            raise HTTPException(
                status_code=422,
                detail="For two-face mode, Person 1 and Person 2 must use different sides.",
            )
        if has_first_face_bytes and not has_second_face and (has_person2_controls or has_interaction_controls):
            raise HTTPException(
                status_code=422,
                detail="Upload a second face image to use Person 2 or interaction controls.",
            )

        face_prompts: list[FacePrompt] = []
        face_images: list[bytes] = []
        face_positions: list[str] = []
        if has_first_face_bytes:
            face_prompts.append(
                FacePrompt(
                    label="Person 1",
                    role_prompt=person1_prompt or "",
                    position=person1_position_value,
                )
            )
            face_images.append(face_image_bytes)
            face_positions.append(person1_position_value)
        if has_second_face:
            face_prompts.append(
                FacePrompt(
                    label="Person 2",
                    role_prompt=person2_prompt or "",
                    position=person2_position_value,
                )
            )
            face_images.append(second_face_bytes)
            face_positions.append(person2_position_value)

        required_prompt = _require_prompt(prompt)
        resolved_prompt = compose_generation_prompt(
            scene_prompt=required_prompt,
            face_prompts=face_prompts,
            interaction_prompt=interaction_prompt,
        )
        resolved_negative_prompt = compose_negative_prompt(
            base_negative_prompt=(
                negative_prompt.strip()
                if negative_prompt and negative_prompt.strip()
                else settings.default_negative_prompt
            ),
            num_faces=len(face_images),
            scene_prompt=required_prompt,
        )
        prompt_warning = compose_prompt_warning(required_prompt)

        request_model = GenerationRequest(
            prompt=resolved_prompt,
            negative_prompt=resolved_negative_prompt,
            face_image_bytes=face_images,
            face_positions=face_positions,
            width=_resolve_dimension(width, "width", settings.default_width, settings.max_width),
            height=_resolve_dimension(height, "height", settings.default_height, settings.max_height),
            steps=_resolve_steps(steps, settings.max_steps, settings.default_steps),
            guidance_scale=_resolve_guidance(
                guidance_scale,
                settings.max_guidance_scale,
                settings.default_guidance_scale,
            ),
            num_images=_resolve_num_images(
                num_images,
                settings.max_num_images,
                settings.default_num_images,
            ),
            ip_adapter_scale=_resolve_scale(
                ip_adapter_scale,
                settings.default_ip_adapter_scale,
            ),
            seed=_resolve_seed(seed),
        )

        try:
            result = await run_in_threadpool(service.generate, request_model)
        except InvalidInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FaceNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ModelInitializationError, OutOfMemoryGenerationError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GenerationError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        images = [
            {
                "file_name": image.file_name,
                "preview_url": make_public_url(image.relative_path),
                "download_url": make_public_url(image.relative_path),
            }
            for image in result.images
        ]
        warnings = [warning for warning in [result.warning, prompt_warning] if warning]
        return {
            "job_id": result.job_id,
            "images": images,
            "zip_url": make_public_url(result.zip_relative_path),
            "warning": " ".join(warnings) if warnings else None,
            "seed": result.seed,
            "resolved_prompt": resolved_prompt,
        }

    @app.post("/api/batch")
    async def start_batch(
        prompt: str = Form(...),
        negative_prompt: str | None = Form(None),
        width: int | None = Form(None),
        height: int | None = Form(None),
        steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        seed: int | None = Form(None),
        total_images: int | None = Form(None),
        chunk_size: int | None = Form(None),
        settings: Settings = Depends(get_settings),
        service: LocalInferenceService = Depends(get_inference_service),
        batch_manager: BatchJobManager = Depends(get_batch_manager),
    ) -> dict[str, object]:
        required_prompt = _require_prompt(prompt)
        total_images_value = _resolve_batch_total(
            total_images,
            settings.max_batch_total_images,
            settings.default_batch_total_images,
        )
        chunk_size_value = _resolve_batch_chunk_size(
            chunk_size,
            settings.max_batch_chunk_size,
            settings.default_batch_chunk_size,
            total_images_value,
        )
        resolved_prompt = compose_generation_prompt(
            scene_prompt=required_prompt,
            face_prompts=[],
        )
        resolved_negative_prompt = compose_negative_prompt(
            base_negative_prompt=(
                negative_prompt.strip()
                if negative_prompt and negative_prompt.strip()
                else settings.default_negative_prompt
            ),
            num_faces=0,
            scene_prompt=required_prompt,
        )
        batch_request = BatchGenerationRequest(
            prompt=resolved_prompt,
            negative_prompt=resolved_negative_prompt,
            width=_resolve_dimension(width, "width", settings.default_width, settings.max_width),
            height=_resolve_dimension(height, "height", settings.default_height, settings.max_height),
            steps=_resolve_steps(steps, settings.max_steps, settings.default_steps),
            guidance_scale=_resolve_guidance(
                guidance_scale,
                settings.max_guidance_scale,
                settings.default_guidance_scale,
            ),
            total_images=total_images_value,
            chunk_size=chunk_size_value,
            seed=_resolve_seed(seed),
        )
        snapshot = batch_manager.start(
            batch_request,
            service,
            initial_warning=compose_prompt_warning(required_prompt),
        )
        if snapshot is None:
            raise HTTPException(status_code=500, detail="Batch job could not be created.")
        return snapshot

    @app.get("/api/batch/{job_id}")
    async def get_batch_status(
        job_id: str,
        batch_manager: BatchJobManager = Depends(get_batch_manager),
    ) -> dict[str, object]:
        snapshot = batch_manager.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Batch job was not found.")
        return snapshot

    @app.post("/api/batch/{job_id}/cancel")
    async def cancel_batch(
        job_id: str,
        batch_manager: BatchJobManager = Depends(get_batch_manager),
    ) -> dict[str, object]:
        snapshot = batch_manager.cancel(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Batch job was not found.")
        return snapshot

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    runtime_settings = Settings.from_env()
    uvicorn.run(
        "app.main:app",
        host=runtime_settings.host,
        port=runtime_settings.port,
        reload=False,
    )
