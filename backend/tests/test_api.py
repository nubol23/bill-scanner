from __future__ import annotations

from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from app.engine.base import OcrEngine, OcrResult, RecognizedCandidate
from app.main import create_app
from app.repository import InMemoryTelemetryRepository
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


class StubNotFoundEngine(OcrEngine):
  name = "stub-not-found"

  def recognize(self, image_bytes: bytes) -> OcrResult:
    return OcrResult(
      raw_text="noise",
      serial=None,
      candidates=[RecognizedCandidate(text="024613645", confidence=0.4, series="B")],
      series=None,
      raw_confidence=0.4,
    )


def _build_settings() -> Settings:
  return Settings(
    app_host="127.0.0.1",
    app_port=8000,
    database_url=None,
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


def _create_client(
  *,
  engine: OcrEngine | None = None,
  settings: Settings | None = None,
  repository: InMemoryTelemetryRepository | None = None,
) -> tuple[TestClient, InMemoryTelemetryRepository]:
  resolved_repository = repository or InMemoryTelemetryRepository()
  client = TestClient(
    create_app(
      settings=settings or _build_settings(),
      engine=engine or StubEngine(),
      telemetry_repository=resolved_repository,
    )
  )
  return client, resolved_repository


def _analytics_payload(*, event_count: int = 1, event_name: str = "app_opened") -> dict:
  return {
    "device_id": str(uuid4()),
    "session_id": str(uuid4()),
    "page_load_id": str(uuid4()),
    "app_version": "0.0.0",
    "device_class": "mobile",
    "browser_family": "Mobile Safari",
    "os_family": "iOS",
    "viewport_bucket": "sm",
    "referrer_domain": "example.com",
    "events": [
      {
        "name": event_name,
        "occurred_at": "2026-03-03T12:00:00Z",
        "request_id": None,
        "denomination": "10",
        "method": "camera",
        "outcome": "ok",
        "meta": {"step": index},
      }
      for index in range(event_count)
    ],
  }


def test_healthz_returns_engine_status_and_request_id():
  client, _ = _create_client()
  with client:
    response = client.get("/healthz")

  assert response.status_code == 200
  assert response.json()["status"] == "ok"
  assert response.json()["engine"] == "stub"
  assert response.headers["x-request-id"]


def test_recognize_returns_serial_payload_and_persists_ocr_record():
  client, repository = _create_client()

  with client:
    response = client.post(
      "/api/v1/recognize",
      data={
        "context": (
          '{"device_id":"11111111-1111-1111-1111-111111111111",'
          '"session_id":"22222222-2222-2222-2222-222222222222",'
          '"page_load_id":"33333333-3333-3333-3333-333333333333",'
          '"denomination":"20","torch_enabled":true,'
          '"client_started_at":"2026-03-03T12:00:00Z"}'
        )
      },
      files={"image": ("serial.png", _png_bytes(), "image/png")},
      headers={"CF-Connecting-IP": "200.1.1.8"},
    )

  assert response.status_code == 200
  payload = response.json()
  assert payload["status"] == "ok"
  assert payload["serial"] == "024613645"
  assert payload["series"] == "B"
  assert payload["candidates"][0]["text"] == "024613645"
  assert response.headers["x-request-id"] == payload["request_id"]
  assert len(repository.ocr_inferences) == 1
  stored = repository.ocr_inferences[0]
  assert stored.outcome == "ok"
  assert stored.serial == "024613645"
  assert stored.response_payload["serial"] == "024613645"


def test_recognize_not_found_persists_not_found_outcome():
  client, repository = _create_client(engine=StubNotFoundEngine())

  with client:
    response = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.png", _png_bytes(), "image/png")},
    )

  assert response.status_code == 200
  assert response.json()["status"] == "not_found"
  assert len(repository.ocr_inferences) == 1
  assert repository.ocr_inferences[0].outcome == "not_found"
  assert repository.ocr_inferences[0].raw_text == "noise"


def test_recognize_returns_cors_header_for_allowed_origin():
  client, _ = _create_client()

  with client:
    response = client.post(
      "/api/v1/recognize",
      headers={"Origin": "http://127.0.0.1:4173"},
      files={"image": ("serial.png", _png_bytes(), "image/png")},
    )

  assert response.status_code == 200
  assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:4173"
  assert response.headers["access-control-expose-headers"] == "X-Request-Id"


def test_rejects_unsupported_content_type_and_persists_invalid_image():
  client, repository = _create_client()

  with client:
    response = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.txt", b"hello", "text/plain")},
      headers={"CF-Connecting-IP": "200.1.1.8"},
    )

  assert response.status_code == 415
  assert response.headers["x-request-id"]
  assert repository.ocr_inferences[0].outcome == "invalid_image"
  assert repository.ocr_inferences[0].response_payload["detail"] == "Unsupported image type."


def test_rejects_large_uploads():
  client, repository = _create_client()

  with client:
    response = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.png", b"x" * 2048, "image/png")},
      headers={"CF-Connecting-IP": "200.1.1.8"},
    )

  assert response.status_code == 413
  assert repository.ocr_inferences[0].outcome == "invalid_image"


def test_rate_limit_returns_429_and_persists_rate_limited_outcome():
  settings = Settings(
    app_host="127.0.0.1",
    app_port=8000,
    database_url=None,
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
  client, repository = _create_client(settings=settings)

  with client:
    first = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.png", _png_bytes(), "image/png")},
      headers={"CF-Connecting-IP": "200.1.1.8"},
    )
    second = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.png", _png_bytes(), "image/png")},
      headers={"CF-Connecting-IP": "200.1.1.8"},
    )

  assert first.status_code == 200
  assert second.status_code == 429
  assert repository.ocr_inferences[-1].outcome == "rate_limited"


def test_telemetry_insert_failure_does_not_fail_ocr_request():
  repository = InMemoryTelemetryRepository()
  repository.fail_ocr_insert = True
  client, repository = _create_client(repository=repository)

  with client:
    response = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.png", _png_bytes(), "image/png")},
    )

  assert response.status_code == 200
  assert response.json()["status"] == "ok"
  assert repository.ocr_inferences == []


def test_geoip_lookup_stores_matching_department():
  repository = InMemoryTelemetryRepository()
  repository.add_geoip_network(
    "200.1.1.0/24",
    department_code="LP",
    department_name="La Paz",
  )
  client, repository = _create_client(repository=repository)

  with client:
    response = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.png", _png_bytes(), "image/png")},
      headers={"CF-Connecting-IP": "200.1.1.8"},
    )

  assert response.status_code == 200
  assert repository.ocr_inferences[0].region is not None
  assert repository.ocr_inferences[0].region.department_name == "La Paz"


def test_geoip_lookup_keeps_null_region_when_no_match():
  client, repository = _create_client()

  with client:
    response = client.post(
      "/api/v1/recognize",
      files={"image": ("serial.png", _png_bytes(), "image/png")},
      headers={"CF-Connecting-IP": "200.1.1.8"},
    )

  assert response.status_code == 200
  assert repository.ocr_inferences[0].region is None


def test_events_batch_accepts_valid_payload():
  client, repository = _create_client()

  with client:
    response = client.post("/api/v1/events/batch", json=_analytics_payload())

  assert response.status_code == 204
  assert len(repository.analytics_events) == 1
  assert repository.analytics_events[0].event_name == "app_opened"


def test_events_batch_rejects_more_than_ten_events():
  client, _ = _create_client()

  with client:
    response = client.post(
      "/api/v1/events/batch",
      json=_analytics_payload(event_count=11),
    )

  assert response.status_code == 413


def test_events_batch_rejects_invalid_event_names():
  client, _ = _create_client()
  payload = _analytics_payload(event_name="invalid_event")

  with client:
    response = client.post("/api/v1/events/batch", json=payload)

  assert response.status_code == 400


def test_feedback_accepts_trimmed_comment_and_null_request_id():
  client, repository = _create_client()
  payload = {
    "device_id": str(uuid4()),
    "session_id": str(uuid4()),
    "page_load_id": str(uuid4()),
    "request_id": None,
    "rating": "up",
    "comment": "  todo bien  ",
    "prompted_after_scan_count": 5,
  }

  with client:
    response = client.post("/api/v1/feedback", json=payload)

  assert response.status_code == 204
  assert len(repository.feedback_entries) == 1
  assert repository.feedback_entries[0].comment == "todo bien"
  assert repository.feedback_entries[0].request_id is None


def test_feedback_rejects_comment_over_500_characters():
  client, _ = _create_client()
  payload = {
    "device_id": str(uuid4()),
    "session_id": str(uuid4()),
    "page_load_id": str(uuid4()),
    "request_id": None,
    "rating": "down",
    "comment": "x" * 501,
    "prompted_after_scan_count": 5,
  }

  with client:
    response = client.post("/api/v1/feedback", json=payload)

  assert response.status_code == 400


def test_feedback_rate_limit_is_enforced_per_device():
  client, _ = _create_client()
  device_id = str(uuid4())
  payload = {
    "device_id": device_id,
    "session_id": str(uuid4()),
    "page_load_id": str(uuid4()),
    "request_id": None,
    "rating": "up",
    "comment": None,
    "prompted_after_scan_count": 5,
  }

  with client:
    responses = [client.post("/api/v1/feedback", json=payload) for _ in range(6)]

  assert [response.status_code for response in responses[:5]] == [204, 204, 204, 204, 204]
  assert responses[5].status_code == 429
