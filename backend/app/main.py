from __future__ import annotations

from fastapi import FastAPI
import uvicorn

from .api import router
from .engine.base import OcrEngine
from .engine.onnx_recognizer import OnnxSerialRecognizer
from .limiter import RateLimiter, RequestQueue
from .middleware import install_http_middleware
from .settings import Settings


def create_app(settings: Settings | None = None, engine: OcrEngine | None = None) -> FastAPI:
  resolved_settings = settings or Settings.from_env()
  resolved_engine = engine or OnnxSerialRecognizer(resolved_settings)

  app = FastAPI(title="Billete OCR Backend", version="0.1.0")
  app.state.settings = resolved_settings
  app.state.engine = resolved_engine
  app.state.rate_limiter = RateLimiter(
    burst_limit=resolved_settings.rate_limit_burst,
    burst_window_seconds=resolved_settings.rate_limit_window_seconds,
    sustained_limit=resolved_settings.rate_limit_per_minute,
  )
  app.state.request_queue = RequestQueue(max_pending=resolved_settings.queue_max)
  install_http_middleware(app, resolved_settings)
  app.include_router(router)
  return app


app = create_app()


def run() -> None:
  settings = Settings.from_env()
  uvicorn.run(
    "app.main:app",
    host=settings.app_host,
    port=settings.app_port,
    workers=1,
  )
