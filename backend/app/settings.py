from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
DEFAULT_ALLOWED_ORIGINS = (
  "https://billete.rafaelvillca.site",
)


def _parse_origins(raw_value: str | None) -> tuple[str, ...]:
  if not raw_value:
    return DEFAULT_ALLOWED_ORIGINS

  return tuple(origin.strip() for origin in raw_value.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
  app_host: str
  app_port: int
  database_url: str | None
  allowed_origins: tuple[str, ...]
  max_upload_bytes: int
  rate_limit_burst: int
  rate_limit_window_seconds: int
  rate_limit_per_minute: int
  queue_max: int
  cpu_threads: int
  model_path: Path
  model_meta_path: Path

  @classmethod
  def from_env(cls) -> "Settings":
    model_path = Path(
      os.environ.get(
        "APP_MODEL_PATH",
        MODELS_DIR / "en_PP-OCRv5_mobile_rec.onnx",
      )
    )
    model_meta_path = Path(
      os.environ.get(
        "APP_MODEL_META_PATH",
        MODELS_DIR / "en_PP-OCRv5_mobile_rec.meta.json",
      )
    )

    return cls(
      app_host=os.environ.get("APP_HOST", "127.0.0.1"),
      app_port=int(os.environ.get("APP_PORT", "8000")),
      database_url=os.environ.get("APP_DATABASE_URL"),
      allowed_origins=_parse_origins(os.environ.get("APP_ALLOWED_ORIGINS")),
      max_upload_bytes=int(os.environ.get("APP_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024))),
      rate_limit_burst=int(os.environ.get("APP_RATE_LIMIT_BURST", "5")),
      rate_limit_window_seconds=int(os.environ.get("APP_RATE_LIMIT_WINDOW_SECONDS", "10")),
      rate_limit_per_minute=int(os.environ.get("APP_RATE_LIMIT_PER_MINUTE", "30")),
      queue_max=int(os.environ.get("APP_QUEUE_MAX", "8")),
      cpu_threads=int(os.environ.get("APP_CPU_THREADS", "4")),
      model_path=model_path,
      model_meta_path=model_meta_path,
    )
