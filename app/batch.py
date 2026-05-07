from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import threading
from time import time
from uuid import uuid4
import zipfile

from .inference import GenerationRequest, LocalInferenceService, SavedImage, make_public_url
from .settings import Settings


BATCH_PREVIEW_LIMIT = 12


def _combine_messages(messages: list[str | None]) -> str | None:
    unique: list[str] = []
    seen: set[str] = set()
    for message in messages:
        cleaned = message.strip() if message else ""
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    return " ".join(unique) if unique else None


@dataclass(frozen=True)
class BatchGenerationRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    total_images: int
    chunk_size: int
    seed: int | None = None


@dataclass
class BatchJob:
    job_id: str
    request: BatchGenerationRequest
    status: str = "queued"
    completed_images: int = 0
    saved_images: list[SavedImage] = field(default_factory=list)
    zip_relative_path: str | None = None
    warning: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class BatchJobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chitradev-batch")

    def start(
        self,
        request: BatchGenerationRequest,
        service: LocalInferenceService,
        initial_warning: str | None = None,
    ) -> dict[str, object]:
        job_id = f"batch-{uuid4().hex}"
        job = BatchJob(job_id=job_id, request=request, warning=initial_warning)
        with self._lock:
            self._jobs[job_id] = job

        self._executor.submit(self._run_job, job_id, service)
        return self.snapshot(job_id)

    def cancel(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"queued", "running"}:
                job.cancel_requested = True
                job.updated_at = time()
        return self.snapshot(job_id)

    def snapshot(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._snapshot_locked(job)

    def _run_job(self, job_id: str, service: LocalInferenceService) -> None:
        self._mark(job_id, status="running")
        job_dir = self.settings.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        request = self._get_request(job_id)
        if request is None:
            return

        self._write_prompt_summary(job_dir, request)
        warnings: list[str] = []

        try:
            while True:
                with self._lock:
                    job = self._jobs[job_id]
                    if job.cancel_requested:
                        break
                    remaining = job.request.total_images - job.completed_images
                    completed = job.completed_images

                if remaining <= 0:
                    break

                chunk_count = min(request.chunk_size, remaining)
                chunk_seed = request.seed + completed if request.seed is not None else None
                generation_request = GenerationRequest(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    face_image_bytes=[],
                    face_positions=[],
                    width=request.width,
                    height=request.height,
                    steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    num_images=chunk_count,
                    ip_adapter_scale=0.0,
                    seed=chunk_seed,
                )
                result = service.generate_images(generation_request)
                if result.warning:
                    warnings.append(result.warning)

                saved_images: list[SavedImage] = []
                for offset, image in enumerate(result.images, start=1):
                    next_index = completed + offset
                    file_name = f"batch-image-{next_index:05d}.png"
                    image.save(job_dir / file_name, format="PNG")
                    saved_images.append(
                        SavedImage(
                            file_name=file_name,
                            relative_path=f"{job_id}/{file_name}",
                        )
                    )

                self._add_images(job_id, saved_images, " ".join(warnings) if warnings else None)

            zip_relative_path = self._write_zip(job_id, job_dir)
            with self._lock:
                job = self._jobs[job_id]
                job.zip_relative_path = zip_relative_path
                job.warning = _combine_messages([job.warning, *warnings])
                job.status = "canceled" if job.cancel_requested else "completed"
                job.updated_at = time()
        except Exception as exc:
            zip_relative_path = self._write_zip(job_id, job_dir)
            with self._lock:
                job = self._jobs[job_id]
                job.zip_relative_path = zip_relative_path
                job.status = "failed"
                job.error = str(exc)
                job.warning = _combine_messages([job.warning, *warnings])
                job.updated_at = time()

    def _get_request(self, job_id: str) -> BatchGenerationRequest | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.request if job else None

    def _add_images(self, job_id: str, images: list[SavedImage], warning: str | None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.saved_images.extend(images)
            job.completed_images = len(job.saved_images)
            if warning:
                job.warning = _combine_messages([job.warning, warning])
            job.updated_at = time()

    def _mark(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.updated_at = time()

    def _snapshot_locked(self, job: BatchJob) -> dict[str, object]:
        percent = int((job.completed_images / job.request.total_images) * 100) if job.request.total_images else 0
        preview_images = job.saved_images[-BATCH_PREVIEW_LIMIT:]
        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_images": job.request.total_images,
            "completed_images": job.completed_images,
            "percent": min(percent, 100),
            "chunk_size": job.request.chunk_size,
            "zip_url": make_public_url(job.zip_relative_path) if job.zip_relative_path else None,
            "warning": job.warning,
            "error": job.error,
            "cancel_requested": job.cancel_requested,
            "preview_images": [
                {
                    "file_name": image.file_name,
                    "preview_url": make_public_url(image.relative_path),
                    "download_url": make_public_url(image.relative_path),
                }
                for image in preview_images
            ],
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    def _write_prompt_summary(self, job_dir: Path, request: BatchGenerationRequest) -> None:
        summary = (
            f"Prompt:\n{request.prompt}\n\n"
            f"Negative prompt:\n{request.negative_prompt}\n\n"
            f"Settings:\n"
            f"- total_images: {request.total_images}\n"
            f"- chunk_size: {request.chunk_size}\n"
            f"- width: {request.width}\n"
            f"- height: {request.height}\n"
            f"- steps: {request.steps}\n"
            f"- guidance_scale: {request.guidance_scale}\n"
            f"- seed: {request.seed if request.seed is not None else 'random'}\n"
        )
        (job_dir / "batch-prompt.txt").write_text(summary, encoding="utf-8")

    def _write_zip(self, job_id: str, job_dir: Path) -> str | None:
        with self._lock:
            job = self._jobs.get(job_id)
            images = list(job.saved_images) if job else []

        if not images:
            return None

        zip_path = job_dir / "batch-results.zip"
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            prompt_file = job_dir / "batch-prompt.txt"
            if prompt_file.exists():
                archive.write(prompt_file, arcname=prompt_file.name)
            for image in images:
                archive.write(job_dir / image.file_name, arcname=image.file_name)
        return f"{job_id}/{zip_path.name}"
