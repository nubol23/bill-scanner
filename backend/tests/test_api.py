from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.engine.base import OcrEngine, OcrResult, RecognizedCandidate
from app.main import create_app
from app.settings import Settings


class StubEngine(OcrEngine):
  name = "stub"

  def recognize(self, image_bytes: bytes) -> OcrResult:
    return OcrResult(
      raw_text="024613645 B",
      serial="024613645",
      candidates=[RecognizedCandidate(text="024613645", confidence=0.99, series="B")],
      series="B",
      raw_confidence=0.99,
    )


def _build_settings() -> Settings:
  return Settings(
    app_host="127.0.0.1",
    app_port=8000,
    allowed_origins=("http://127.0.0.1:4173",),
    max_upload_bytes=1024,
    rate_limit_burst=10,
    rate_limit_window_seconds=10,
    rate_limit_per_minute=30,
    queue_max=8,
    cpu_threads=1,
    model_path=Settings.from_env().model_path,
    model_meta_path=Settings.from_env().model_meta_path,
  )


def _png_bytes(width: int = 120, height: int = 40) -> bytes:
  image = Image.new("RGB", (width, height), "white")
  buffer = BytesIO()
  image.save(buffer, format="PNG")
  return buffer.getvalue()


def test_healthz_returns_engine_status():
  client = TestClient(create_app(settings=_build_settings(), engine=StubEngine()))
  response = client.get("/healthz")

  assert response.status_code == 200
  assert response.json()["status"] == "ok"
  assert response.json()["engine"] == "stub"


def test_recognize_returns_serial_payload():
  client = TestClient(create_app(settings=_build_settings(), engine=StubEngine()))
  response = client.post(
    "/api/v1/recognize",
    files={"image": ("serial.png", _png_bytes(), "image/png")},
  )

  assert response.status_code == 200
  payload = response.json()
  assert payload["status"] == "ok"
  assert payload["serial"] == "024613645"
  assert payload["series"] == "B"
  assert payload["candidates"][0]["text"] == "024613645"


def test_recognize_returns_cors_header_for_allowed_origin():
  client = TestClient(create_app(settings=_build_settings(), engine=StubEngine()))
  response = client.post(
    "/api/v1/recognize",
    headers={"Origin": "http://127.0.0.1:4173"},
    files={"image": ("serial.png", _png_bytes(), "image/png")},
  )

  assert response.status_code == 200
  assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:4173"


def test_rejects_unsupported_content_type():
  client = TestClient(create_app(settings=_build_settings(), engine=StubEngine()))
  response = client.post(
    "/api/v1/recognize",
    files={"image": ("serial.txt", b"hello", "text/plain")},
  )

  assert response.status_code == 415


def test_rejects_large_uploads():
  client = TestClient(create_app(settings=_build_settings(), engine=StubEngine()))
  response = client.post(
    "/api/v1/recognize",
    files={"image": ("serial.png", b"x" * 2048, "image/png")},
  )

  assert response.status_code == 413


def test_rate_limit_returns_429():
  settings = Settings(
    app_host="127.0.0.1",
    app_port=8000,
    allowed_origins=("http://127.0.0.1:4173",),
    max_upload_bytes=1024,
    rate_limit_burst=1,
    rate_limit_window_seconds=10,
    rate_limit_per_minute=1,
    queue_max=8,
    cpu_threads=1,
    model_path=Settings.from_env().model_path,
    model_meta_path=Settings.from_env().model_meta_path,
  )
  client = TestClient(create_app(settings=settings, engine=StubEngine()))

  first = client.post(
    "/api/v1/recognize",
    files={"image": ("serial.png", _png_bytes(), "image/png")},
  )
  second = client.post(
    "/api/v1/recognize",
    files={"image": ("serial.png", _png_bytes(), "image/png")},
  )

  assert first.status_code == 200
  assert second.status_code == 429
