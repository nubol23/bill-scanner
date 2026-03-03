from __future__ import annotations

from pydantic import BaseModel, Field


class RecognizedCandidatePayload(BaseModel):
  text: str = Field(min_length=8, max_length=9)
  confidence: float = Field(ge=0.0, le=1.0)


class RecognizeResponse(BaseModel):
  status: str
  serial: str | None
  series: str | None = Field(default=None, min_length=1, max_length=1)
  raw_text: str
  confidence: float | None = Field(default=None, ge=0.0, le=1.0)
  candidates: list[RecognizedCandidatePayload]
  engine: str
  latency_ms: int = Field(ge=0)
  request_id: str


class HealthResponse(BaseModel):
  status: str
  model_loaded: bool
  engine: str
  version: str
