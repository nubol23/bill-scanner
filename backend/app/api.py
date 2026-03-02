from __future__ import annotations

import asyncio
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from .engine.base import InvalidImageError
from .limiter import RequestCapacityExceeded
from .schemas import HealthResponse, RecognizeResponse, RecognizedCandidatePayload


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


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
  return HealthResponse(
    status="ok",
    model_loaded=True,
    engine=request.app.state.engine.name,
    version="0.1.0",
  )


@router.post("/api/v1/recognize", response_model=RecognizeResponse)
async def recognize(request: Request, image: UploadFile = File(...)) -> RecognizeResponse:
  if image.content_type not in ACCEPTED_CONTENT_TYPES:
    raise HTTPException(status_code=415, detail="Unsupported image type.")

  client_ip = _resolve_client_ip(request)
  if not request.app.state.rate_limiter.allow(client_ip):
    raise HTTPException(status_code=429, detail="Rate limit exceeded.")

  payload = await image.read()
  await image.close()

  if not payload:
    raise HTTPException(status_code=400, detail="No image payload was sent.")
  if len(payload) > request.app.state.settings.max_upload_bytes:
    raise HTTPException(status_code=413, detail="Image payload is too large.")

  request_id = str(uuid4())
  started_at = perf_counter()

  try:
    async with request.app.state.request_queue.acquire():
      result = await asyncio.to_thread(request.app.state.engine.recognize, payload)
  except RequestCapacityExceeded as error:
    raise HTTPException(status_code=429, detail=str(error)) from error
  except InvalidImageError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  except Exception as error:
    raise HTTPException(status_code=500, detail="OCR inference failed.") from error

  latency_ms = int((perf_counter() - started_at) * 1000)

  return RecognizeResponse(
    status="ok" if result.serial else "not_found",
    serial=result.serial,
    raw_text=result.raw_text,
    confidence=result.raw_confidence if result.serial else None,
    candidates=[
      RecognizedCandidatePayload(text=candidate.text, confidence=candidate.confidence)
      for candidate in result.candidates
    ],
    engine=request.app.state.engine.name,
    latency_ms=latency_ms,
    request_id=request_id,
  )
