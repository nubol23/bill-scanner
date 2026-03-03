from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from .api import router
from .engine.base import OcrEngine
from .engine.onnx_recognizer import OnnxSerialRecognizer
from .limiter import RateLimiter, RequestQueue
from .middleware import install_http_middleware
from .repository import (
  BaseTelemetryRepository,
  NullTelemetryRepository,
  PostgresTelemetryRepository,
)
from .settings import Settings


def _resolve_telemetry_repository(
  settings: Settings,
  telemetry_repository: BaseTelemetryRepository | None,
) -> BaseTelemetryRepository:
  if telemetry_repository is not None:
    return telemetry_repository

  if settings.database_url:
    return PostgresTelemetryRepository(settings.database_url)

  return NullTelemetryRepository("APP_DATABASE_URL is not configured.")


def create_app(
  settings: Settings | None = None,
  engine: OcrEngine | None = None,
  telemetry_repository: BaseTelemetryRepository | None = None,
) -> FastAPI:
  resolved_settings = settings or Settings.from_env()
  resolved_engine = engine or OnnxSerialRecognizer(resolved_settings)
  resolved_telemetry = _resolve_telemetry_repository(
    resolved_settings,
    telemetry_repository,
  )

  @asynccontextmanager
  async def lifespan(_: FastAPI):
    await resolved_telemetry.startup()
    try:
      yield
    finally:
      await resolved_telemetry.shutdown()

  app = FastAPI(
    title="Billete OCR Backend",
    version="0.1.0",
    lifespan=lifespan,
  )
  app.state.settings = resolved_settings
  app.state.engine = resolved_engine
  app.state.telemetry = resolved_telemetry
  app.state.rate_limiter = RateLimiter(
    burst_limit=resolved_settings.rate_limit_burst,
    burst_window_seconds=resolved_settings.rate_limit_window_seconds,
    sustained_limit=resolved_settings.rate_limit_per_minute,
  )
  app.state.feedback_ip_limiter = RateLimiter(
    burst_limit=20,
    burst_window_seconds=24 * 60 * 60,
    sustained_limit=20,
    sustained_window_seconds=24 * 60 * 60,
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
