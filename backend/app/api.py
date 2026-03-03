from __future__ import annotations

import asyncio
from datetime import timezone
import json
import logging
from time import perf_counter
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import ValidationError

from .engine.base import InvalidImageError
from .limiter import RequestCapacityExceeded
from .repository import (
  AnalyticsEventRecord,
  FeedbackRecord,
  OcrInferenceRecord,
  TelemetryUnavailableError,
)
from .schemas import (
  AnalyticsEventEnvelope,
  ErrorResponse,
  FeedbackRequest,
  HealthResponse,
  RecognizeClientContext,
  RecognizeResponse,
  RecognizedCandidatePayload,
)


logger = logging.getLogger(__name__)

ACCEPTED_CONTENT_TYPES = {
  "image/jpeg",
  "image/png",
  "image/webp",
}

router = APIRouter()


def _resolve_client_ip(request: Request) -> str:
  forwarded_ip = request.headers.get("CF-Connecting-IP")
  if forwarded_ip:
    return forwarded_ip
  if request.client:
    return request.client.host
  return "unknown"


def _resolve_request_id(request: Request) -> UUID:
  return UUID(str(request.state.request_id))


def _error_payload(detail: str) -> dict[str, str]:
  return {"detail": detail}


def _coerce_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
  return json.loads(json.dumps(payload))


def _coerce_recognize_context(raw_context: str | None) -> RecognizeClientContext | None:
  if raw_context is None:
    return None

  try:
    return RecognizeClientContext.model_validate(json.loads(raw_context))
  except (ValidationError, json.JSONDecodeError) as error:
    raise HTTPException(status_code=400, detail="Invalid recognize context payload.") from error


async def _resolve_region(request: Request, client_ip: str):
  telemetry = request.app.state.telemetry
  if not telemetry.is_available:
    return None

  try:
    return await telemetry.resolve_region(client_ip)
  except Exception as error:
    logger.exception("Failed to resolve GeoIP region: %s", error)
    return None


async def _safe_record_ocr_inference(
  request: Request,
  *,
  context: RecognizeClientContext | None,
  outcome: str,
  response_payload: dict[str, Any],
  client_ip: str,
  engine: str | None = None,
  latency_ms: int | None = None,
  raw_text: str | None = None,
  serial: str | None = None,
  series: str | None = None,
  confidence: float | None = None,
  candidates: list[dict[str, Any]] | None = None,
) -> None:
  telemetry = request.app.state.telemetry
  if not telemetry.is_available:
    return

  try:
    region = await _resolve_region(request, client_ip)
    await telemetry.record_ocr_inference(
      OcrInferenceRecord(
        request_id=_resolve_request_id(request),
        device_id=context.device_id if context else None,
        session_id=context.session_id if context else None,
        page_load_id=context.page_load_id if context else None,
        denomination=context.denomination if context else None,
        torch_enabled=context.torch_enabled if context else None,
        client_started_at=(
          context.client_started_at.astimezone(timezone.utc) if context else None
        ),
        outcome=outcome,
        engine=engine,
        latency_ms=latency_ms,
        raw_text=raw_text,
        serial=serial,
        series=series,
        confidence=confidence,
        candidates=candidates or [],
        response_payload=_coerce_response_payload(response_payload),
        region=region,
      )
    )
  except Exception as error:
    logger.exception("Failed to persist OCR telemetry: %s", error)


def _raise_http_error(status_code: int, detail: str) -> None:
  raise HTTPException(status_code=status_code, detail=detail)


async def _record_error_and_raise(
  request: Request,
  *,
  context: RecognizeClientContext | None,
  client_ip: str,
  outcome: str,
  status_code: int,
  detail: str,
) -> None:
  await _safe_record_ocr_inference(
    request,
    context=context,
    outcome=outcome,
    response_payload=_error_payload(detail),
    client_ip=client_ip,
  )
  _raise_http_error(status_code, detail)


def _ensure_telemetry_available(request: Request):
  telemetry = request.app.state.telemetry
  if not telemetry.is_available:
    raise HTTPException(status_code=503, detail=telemetry.unavailable_reason)
  return telemetry


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
  return HealthResponse(
    status="ok",
    model_loaded=True,
    engine=request.app.state.engine.name,
    version="0.1.0",
  )


@router.post("/api/v1/recognize", response_model=RecognizeResponse)
async def recognize(
  request: Request,
  image: UploadFile = File(...),
  context: str | None = Form(default=None),
) -> RecognizeResponse:
  client_ip = _resolve_client_ip(request)
  parsed_context: RecognizeClientContext | None = None

  try:
    parsed_context = _coerce_recognize_context(context)
  except HTTPException as error:
    await image.close()
    await _safe_record_ocr_inference(
      request,
      context=None,
      outcome="invalid_image",
      response_payload=_error_payload(str(error.detail)),
      client_ip=client_ip,
    )
    raise

  if image.content_type not in ACCEPTED_CONTENT_TYPES:
    await image.close()
    await _record_error_and_raise(
      request,
      context=parsed_context,
      client_ip=client_ip,
      outcome="invalid_image",
      status_code=415,
      detail="Unsupported image type.",
    )

  if not request.app.state.rate_limiter.allow(client_ip):
    await image.close()
    await _record_error_and_raise(
      request,
      context=parsed_context,
      client_ip=client_ip,
      outcome="rate_limited",
      status_code=429,
      detail="Rate limit exceeded.",
    )

  payload = await image.read()
  await image.close()

  if not payload:
    await _record_error_and_raise(
      request,
      context=parsed_context,
      client_ip=client_ip,
      outcome="invalid_image",
      status_code=400,
      detail="No image payload was sent.",
    )

  if len(payload) > request.app.state.settings.max_upload_bytes:
    await _record_error_and_raise(
      request,
      context=parsed_context,
      client_ip=client_ip,
      outcome="invalid_image",
      status_code=413,
      detail="Image payload is too large.",
    )

  started_at = perf_counter()

  try:
    async with request.app.state.request_queue.acquire():
      result = await asyncio.to_thread(request.app.state.engine.recognize, payload)
  except RequestCapacityExceeded as error:
    detail = str(error)
    await _record_error_and_raise(
      request,
      context=parsed_context,
      client_ip=client_ip,
      outcome="rate_limited",
      status_code=429,
      detail=detail,
    )
  except InvalidImageError as error:
    detail = str(error)
    await _record_error_and_raise(
      request,
      context=parsed_context,
      client_ip=client_ip,
      outcome="invalid_image",
      status_code=400,
      detail=detail,
    )
  except Exception as error:
    logger.exception("OCR inference failed: %s", error)
    await _record_error_and_raise(
      request,
      context=parsed_context,
      client_ip=client_ip,
      outcome="server_error",
      status_code=500,
      detail="OCR inference failed.",
    )

  latency_ms = int((perf_counter() - started_at) * 1000)
  response_payload = RecognizeResponse(
    status="ok" if result.serial else "not_found",
    serial=result.serial,
    series=result.series if result.serial else None,
    raw_text=result.raw_text,
    confidence=result.raw_confidence if result.serial else None,
    candidates=[
      RecognizedCandidatePayload(text=candidate.text, confidence=candidate.confidence)
      for candidate in result.candidates
    ],
    engine=request.app.state.engine.name,
    latency_ms=latency_ms,
    request_id=str(request.state.request_id),
  )

  await _safe_record_ocr_inference(
    request,
    context=parsed_context,
    outcome="ok" if result.serial else "not_found",
    response_payload=response_payload.model_dump(mode="json"),
    client_ip=client_ip,
    engine=response_payload.engine,
    latency_ms=response_payload.latency_ms,
    raw_text=response_payload.raw_text,
    serial=response_payload.serial,
    series=response_payload.series,
    confidence=response_payload.confidence,
    candidates=[candidate.model_dump(mode="json") for candidate in response_payload.candidates],
  )

  return response_payload


@router.post("/api/v1/events/batch", status_code=204)
async def events_batch(
  request: Request,
  body: dict[str, Any] = Body(...),
) -> Response:
  telemetry = _ensure_telemetry_available(request)

  try:
    envelope = AnalyticsEventEnvelope.model_validate(body)
  except ValidationError as error:
    messages = [entry["msg"] for entry in error.errors()]
    if any("Too many events" in message or "too large" in message for message in messages):
      raise HTTPException(status_code=413, detail="Analytics batch is too large.") from error
    raise HTTPException(status_code=400, detail="Invalid analytics payload.") from error

  try:
    await telemetry.record_analytics_events(
      [
        AnalyticsEventRecord(
          occurred_at=event.occurred_at.astimezone(timezone.utc),
          device_id=envelope.device_id,
          session_id=envelope.session_id,
          page_load_id=envelope.page_load_id,
          event_name=event.name,
          request_id=event.request_id,
          denomination=event.denomination,
          method=event.method,
          outcome=event.outcome,
          app_version=envelope.app_version,
          device_class=envelope.device_class,
          browser_family=envelope.browser_family,
          os_family=envelope.os_family,
          viewport_bucket=envelope.viewport_bucket,
          referrer_domain=envelope.referrer_domain,
          meta=event.meta,
        )
        for event in envelope.events
      ]
    )
  except TelemetryUnavailableError as error:
    raise HTTPException(status_code=503, detail=str(error)) from error
  except Exception as error:
    logger.exception("Failed to persist analytics events: %s", error)
    raise HTTPException(status_code=503, detail="Telemetry storage is unavailable.") from error

  return Response(status_code=204)


@router.post("/api/v1/feedback", status_code=204, responses={400: {"model": ErrorResponse}})
async def feedback(
  request: Request,
  body: dict[str, Any] = Body(...),
) -> Response:
  telemetry = _ensure_telemetry_available(request)

  try:
    payload = FeedbackRequest.model_validate(body)
  except ValidationError as error:
    raise HTTPException(status_code=400, detail="Invalid feedback payload.") from error

  client_ip = _resolve_client_ip(request)
  if not request.app.state.feedback_ip_limiter.allow(client_ip):
    raise HTTPException(status_code=429, detail="Feedback rate limit exceeded.")

  try:
    recent_submissions = await telemetry.count_recent_feedback_by_device(
      payload.device_id,
      hours=24,
    )
  except TelemetryUnavailableError as error:
    raise HTTPException(status_code=503, detail=str(error)) from error
  except Exception as error:
    logger.exception("Failed to query feedback throttling: %s", error)
    raise HTTPException(status_code=503, detail="Telemetry storage is unavailable.") from error

  if recent_submissions >= 5:
    raise HTTPException(status_code=429, detail="Feedback rate limit exceeded.")

  try:
    await telemetry.record_feedback(
      FeedbackRecord(
        device_id=payload.device_id,
        session_id=payload.session_id,
        page_load_id=payload.page_load_id,
        request_id=payload.request_id,
        rating=payload.rating,
        comment=payload.comment,
        prompted_after_scan_count=payload.prompted_after_scan_count,
      )
    )
  except TelemetryUnavailableError as error:
    raise HTTPException(status_code=503, detail=str(error)) from error
  except Exception as error:
    logger.exception("Failed to persist feedback: %s", error)
    raise HTTPException(status_code=503, detail="Telemetry storage is unavailable.") from error

  return Response(status_code=204)
