from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BillDenomination = Literal["10", "20", "50"]
DeviceClass = Literal["mobile", "tablet", "desktop", "unknown"]
ViewportBucket = Literal["xs", "sm", "md", "lg", "xl"]
FeedbackRating = Literal["up", "down"]
AnalyticsEventName = Literal[
  "app_opened",
  "denomination_selected",
  "method_selected",
  "camera_open_requested",
  "camera_open_result",
  "torch_toggled",
  "camera_capture_attempted",
  "scan_client_error",
  "retry_used",
  "fallback_used",
]


class RecognizedCandidatePayload(BaseModel):
  model_config = ConfigDict(extra="forbid")

  text: str = Field(min_length=8, max_length=9)
  confidence: float = Field(ge=0.0, le=1.0)


class RecognizeResponse(BaseModel):
  model_config = ConfigDict(extra="forbid")

  status: str
  serial: str | None
  series: str | None = Field(default=None, min_length=1, max_length=1)
  raw_text: str
  confidence: float | None = Field(default=None, ge=0.0, le=1.0)
  candidates: list[RecognizedCandidatePayload]
  engine: str
  latency_ms: int = Field(ge=0)
  request_id: str


class RecognizeClientContext(BaseModel):
  model_config = ConfigDict(extra="forbid")

  device_id: UUID
  session_id: UUID
  page_load_id: UUID
  denomination: BillDenomination
  torch_enabled: bool | None = None
  client_started_at: datetime


class AnalyticsEventItem(BaseModel):
  model_config = ConfigDict(extra="forbid")

  name: AnalyticsEventName
  occurred_at: datetime
  request_id: UUID | None = None
  denomination: BillDenomination | None = None
  method: Literal["camera", "manual"] | None = None
  outcome: str | None = Field(default=None, max_length=64)
  meta: dict[str, Any] = Field(default_factory=dict)

  @field_validator("meta")
  @classmethod
  def validate_meta_size(cls, value: dict[str, Any]) -> dict[str, Any]:
    if len(json.dumps(value)) > 1024:
      raise ValueError("Event metadata is too large.")
    return value


class AnalyticsEventEnvelope(BaseModel):
  model_config = ConfigDict(extra="forbid")

  device_id: UUID
  session_id: UUID
  page_load_id: UUID
  app_version: str = Field(min_length=1, max_length=32)
  device_class: DeviceClass
  browser_family: str = Field(min_length=1, max_length=64)
  os_family: str = Field(min_length=1, max_length=64)
  viewport_bucket: ViewportBucket
  referrer_domain: str | None = Field(default=None, max_length=255)
  events: list[AnalyticsEventItem]

  @field_validator("events")
  @classmethod
  def validate_event_count(cls, value: list[AnalyticsEventItem]) -> list[AnalyticsEventItem]:
    if not value:
      raise ValueError("At least one event is required.")
    if len(value) > 10:
      raise ValueError("Too many events in a single batch.")
    return value


class FeedbackRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  device_id: UUID
  session_id: UUID
  page_load_id: UUID
  request_id: UUID | None = None
  rating: FeedbackRating
  comment: str | None = None
  prompted_after_scan_count: int = Field(ge=1, le=1_000_000)

  @model_validator(mode="after")
  def normalize_comment(self) -> "FeedbackRequest":
    if self.comment is None:
      return self

    normalized = self.comment.strip()
    if len(normalized) > 500:
      raise ValueError("Comment is too long.")

    self.comment = normalized or None
    return self


class ErrorResponse(BaseModel):
  model_config = ConfigDict(extra="forbid")

  detail: str


class HealthResponse(BaseModel):
  model_config = ConfigDict(extra="forbid")

  status: str
  model_loaded: bool
  engine: str
  version: str
