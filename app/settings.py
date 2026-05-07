from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


MIN_IMAGE_DIMENSION = 256
MAX_IP_ADAPTER_SCALE = 1.5


def _load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_directory(project_root: Path, env_name: str, default: str) -> Path:
    raw_value = os.getenv(env_name, default)
    path = Path(raw_value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _require_dimension(name: str, value: int, maximum: int | None = None) -> None:
    if value < MIN_IMAGE_DIMENSION:
        raise ValueError(f"{name} must be at least {MIN_IMAGE_DIMENSION}.")
    if value % 64 != 0:
        raise ValueError(f"{name} must be a multiple of 64.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be no more than {maximum}.")


@dataclass(frozen=True)
class Settings:
    project_root: Path
    template_dir: Path
    output_dir: Path
    cache_dir: Path
    host: str
    port: int
    base_model: str
    ip_adapter_source: str
    ip_adapter_weight: str
    hf_token: str | None
    local_files_only: bool
    default_negative_prompt: str
    default_width: int
    default_height: int
    default_steps: int
    default_guidance_scale: float
    default_ip_adapter_scale: float
    default_num_images: int
    max_width: int
    max_height: int
    max_steps: int
    max_num_images: int
    max_guidance_scale: float
    default_batch_total_images: int
    max_batch_total_images: int
    default_batch_chunk_size: int
    max_batch_chunk_size: int
    max_face_image_mb: int
    insightface_model_name: str
    insightface_det_size: int

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("PORT must be between 1 and 65535.")

        _require_dimension("MAX_WIDTH", self.max_width)
        _require_dimension("MAX_HEIGHT", self.max_height)
        _require_dimension("DEFAULT_WIDTH", self.default_width, self.max_width)
        _require_dimension("DEFAULT_HEIGHT", self.default_height, self.max_height)
        _require_positive("MAX_STEPS", self.max_steps)
        _require_positive("DEFAULT_STEPS", self.default_steps)
        if self.default_steps > self.max_steps:
            raise ValueError("DEFAULT_STEPS must be no more than MAX_STEPS.")
        _require_positive("MAX_NUM_IMAGES", self.max_num_images)
        _require_positive("DEFAULT_NUM_IMAGES", self.default_num_images)
        if self.default_num_images > self.max_num_images:
            raise ValueError("DEFAULT_NUM_IMAGES must be no more than MAX_NUM_IMAGES.")
        _require_positive("MAX_BATCH_TOTAL_IMAGES", self.max_batch_total_images)
        _require_positive("DEFAULT_BATCH_TOTAL_IMAGES", self.default_batch_total_images)
        if self.default_batch_total_images > self.max_batch_total_images:
            raise ValueError("DEFAULT_BATCH_TOTAL_IMAGES must be no more than MAX_BATCH_TOTAL_IMAGES.")
        _require_positive("MAX_BATCH_CHUNK_SIZE", self.max_batch_chunk_size)
        _require_positive("DEFAULT_BATCH_CHUNK_SIZE", self.default_batch_chunk_size)
        if self.default_batch_chunk_size > self.max_batch_chunk_size:
            raise ValueError("DEFAULT_BATCH_CHUNK_SIZE must be no more than MAX_BATCH_CHUNK_SIZE.")
        if self.default_batch_chunk_size > self.default_batch_total_images:
            raise ValueError("DEFAULT_BATCH_CHUNK_SIZE must be no more than DEFAULT_BATCH_TOTAL_IMAGES.")
        _require_positive("MAX_GUIDANCE_SCALE", self.max_guidance_scale)
        _require_positive("DEFAULT_GUIDANCE_SCALE", self.default_guidance_scale)
        if self.default_guidance_scale > self.max_guidance_scale:
            raise ValueError("DEFAULT_GUIDANCE_SCALE must be no more than MAX_GUIDANCE_SCALE.")
        if self.default_ip_adapter_scale < 0 or self.default_ip_adapter_scale > MAX_IP_ADAPTER_SCALE:
            raise ValueError(f"DEFAULT_IP_ADAPTER_SCALE must be between 0 and {MAX_IP_ADAPTER_SCALE}.")
        _require_positive("MAX_FACE_IMAGE_MB", self.max_face_image_mb)
        _require_positive("INSIGHTFACE_DET_SIZE", self.insightface_det_size)

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        _load_env_file(project_root / ".env")

        return cls(
            project_root=project_root,
            template_dir=project_root / "app" / "templates",
            output_dir=_resolve_directory(project_root, "OUTPUT_DIR", "outputs"),
            cache_dir=_resolve_directory(project_root, "CACHE_DIR", ".cache"),
            host=os.getenv("HOST", "127.0.0.1"),
            port=_read_int("PORT", 8000),
            base_model=os.getenv(
                "SD_BASE_MODEL",
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
            ),
            ip_adapter_source=os.getenv("IP_ADAPTER_SOURCE", "h94/IP-Adapter-FaceID"),
            ip_adapter_weight=os.getenv(
                "IP_ADAPTER_WEIGHT",
                "ip-adapter-faceid_sd15.bin",
            ),
            hf_token=os.getenv("HF_TOKEN") or None,
            local_files_only=_read_bool("LOCAL_FILES_ONLY", False),
            default_negative_prompt=os.getenv(
                "DEFAULT_NEGATIVE_PROMPT",
                "monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
            ),
            default_width=_read_int("DEFAULT_WIDTH", 512),
            default_height=_read_int("DEFAULT_HEIGHT", 512),
            default_steps=_read_int("DEFAULT_STEPS", 20),
            default_guidance_scale=_read_float("DEFAULT_GUIDANCE_SCALE", 7.5),
            default_ip_adapter_scale=_read_float("DEFAULT_IP_ADAPTER_SCALE", 0.6),
            default_num_images=_read_int("DEFAULT_NUM_IMAGES", 1),
            max_width=_read_int("MAX_WIDTH", 768),
            max_height=_read_int("MAX_HEIGHT", 768),
            max_steps=_read_int("MAX_STEPS", 50),
            max_num_images=_read_int("MAX_NUM_IMAGES", 4),
            max_guidance_scale=_read_float("MAX_GUIDANCE_SCALE", 20.0),
            default_batch_total_images=_read_int("DEFAULT_BATCH_TOTAL_IMAGES", 100),
            max_batch_total_images=_read_int("MAX_BATCH_TOTAL_IMAGES", 500),
            default_batch_chunk_size=_read_int("DEFAULT_BATCH_CHUNK_SIZE", 1),
            max_batch_chunk_size=_read_int("MAX_BATCH_CHUNK_SIZE", 4),
            max_face_image_mb=_read_int("MAX_FACE_IMAGE_MB", 10),
            insightface_model_name=os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l"),
            insightface_det_size=_read_int("INSIGHTFACE_DET_SIZE", 640),
        )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def max_face_image_bytes(self) -> int:
        return self.max_face_image_mb * 1024 * 1024
