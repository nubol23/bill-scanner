from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address, ip_network
import logging
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .db import create_async_pool


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoRegion:
  department_code: str
  department_name: str


@dataclass(frozen=True)
class OcrInferenceRecord:
  request_id: UUID
  device_id: UUID | None
  session_id: UUID | None
  page_load_id: UUID | None
  denomination: str | None
  torch_enabled: bool | None
  client_started_at: datetime | None
  outcome: str
  engine: str | None
  latency_ms: int | None
  raw_text: str | None
  serial: str | None
  series: str | None
  confidence: float | None
  candidates: list[dict[str, Any]]
  response_payload: dict[str, Any]
  region: GeoRegion | None


@dataclass(frozen=True)
class AnalyticsEventRecord:
  occurred_at: datetime
  device_id: UUID
  session_id: UUID
  page_load_id: UUID
  event_name: str
  request_id: UUID | None
  denomination: str | None
  method: str | None
  outcome: str | None
  app_version: str
  device_class: str
  browser_family: str
  os_family: str
  viewport_bucket: str
  referrer_domain: str | None
  meta: dict[str, Any]


@dataclass(frozen=True)
class FeedbackRecord:
  device_id: UUID
  session_id: UUID
  page_load_id: UUID
  request_id: UUID | None
  rating: str
  comment: str | None
  prompted_after_scan_count: int
  recorded_at: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc),
    compare=False,
  )


class TelemetryUnavailableError(RuntimeError):
  pass


class BaseTelemetryRepository:
  is_available = False
  unavailable_reason = "Telemetry storage is unavailable."

  async def startup(self) -> None:
    return None

  async def shutdown(self) -> None:
    return None

  async def resolve_region(self, client_ip: str | None) -> GeoRegion | None:
    return None

  async def record_ocr_inference(self, record: OcrInferenceRecord) -> None:
    return None

  async def record_analytics_events(self, records: list[AnalyticsEventRecord]) -> None:
    raise TelemetryUnavailableError(self.unavailable_reason)

  async def count_recent_feedback_by_device(self, device_id: UUID, *, hours: int) -> int:
    raise TelemetryUnavailableError(self.unavailable_reason)

  async def record_feedback(self, record: FeedbackRecord) -> None:
    raise TelemetryUnavailableError(self.unavailable_reason)


class NullTelemetryRepository(BaseTelemetryRepository):
  def __init__(self, reason: str = "Telemetry storage is unavailable.") -> None:
    self.unavailable_reason = reason


class InMemoryTelemetryRepository(BaseTelemetryRepository):
  def __init__(self) -> None:
    self.is_available = True
    self.unavailable_reason = ""
    self.analytics_events: list[AnalyticsEventRecord] = []
    self.ocr_inferences: list[OcrInferenceRecord] = []
    self.feedback_entries: list[FeedbackRecord] = []
    self.geoip_networks: list[tuple[Any, GeoRegion]] = []
    self.fail_ocr_insert = False
    self.fail_analytics_insert = False
    self.fail_feedback_insert = False

  def add_geoip_network(
    self,
    network: str,
    *,
    department_code: str,
    department_name: str,
  ) -> None:
    self.geoip_networks.append(
      (
        ip_network(network),
        GeoRegion(
          department_code=department_code,
          department_name=department_name,
        ),
      )
    )

  async def resolve_region(self, client_ip: str | None) -> GeoRegion | None:
    if not client_ip:
      return None

    try:
      parsed_ip = ip_address(client_ip)
    except ValueError:
      return None

    best_match: tuple[int, GeoRegion] | None = None
    for network, region in self.geoip_networks:
      if parsed_ip in network:
        prefix = network.prefixlen
        if best_match is None or prefix > best_match[0]:
          best_match = (prefix, region)

    return best_match[1] if best_match else None

  async def record_ocr_inference(self, record: OcrInferenceRecord) -> None:
    if self.fail_ocr_insert:
      raise RuntimeError("Failed to persist OCR inference.")
    self.ocr_inferences.append(record)

  async def record_analytics_events(self, records: list[AnalyticsEventRecord]) -> None:
    if self.fail_analytics_insert:
      raise RuntimeError("Failed to persist analytics events.")
    self.analytics_events.extend(records)

  async def count_recent_feedback_by_device(self, device_id: UUID, *, hours: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return sum(
      1
      for entry in self.feedback_entries
      if entry.device_id == device_id and entry.recorded_at >= cutoff
    )

  async def record_feedback(self, record: FeedbackRecord) -> None:
    if self.fail_feedback_insert:
      raise RuntimeError("Failed to persist feedback.")

    linked_request_id = record.request_id
    if linked_request_id and not any(
      inference.request_id == linked_request_id for inference in self.ocr_inferences
    ):
      linked_request_id = None

    stored_record = FeedbackRecord(
      device_id=record.device_id,
      session_id=record.session_id,
      page_load_id=record.page_load_id,
      request_id=linked_request_id,
      rating=record.rating,
      comment=record.comment,
      prompted_after_scan_count=record.prompted_after_scan_count,
      recorded_at=datetime.now(timezone.utc),
    )
    self.feedback_entries.append(stored_record)


class PostgresTelemetryRepository(BaseTelemetryRepository):
  def __init__(self, database_url: str) -> None:
    self._database_url = database_url
    self._pool = None
    self.unavailable_reason = "Telemetry storage is unavailable."
    self.is_available = False

  async def startup(self) -> None:
    if not self._database_url:
      self.unavailable_reason = "APP_DATABASE_URL is not configured."
      return

    pool = create_async_pool(self._database_url)

    try:
      await pool.open()
      async with pool.connection() as conn:
        async with conn.cursor() as cursor:
          await cursor.execute("SELECT 1")
    except Exception as error:
      self.unavailable_reason = "Telemetry storage is unavailable."
      logger.exception("Telemetry startup failed: %s", error)
      await pool.close()
      self._pool = None
      self.is_available = False
      return

    self._pool = pool
    self.is_available = True
    self.unavailable_reason = ""

  async def shutdown(self) -> None:
    if self._pool is not None:
      await self._pool.close()
      self._pool = None
      self.is_available = False

  async def resolve_region(self, client_ip: str | None) -> GeoRegion | None:
    if self._pool is None or not client_ip:
      return None

    try:
      parsed_ip = ip_address(client_ip)
    except ValueError:
      return None

    async with self._pool.connection() as conn:
      async with conn.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
          """
          SELECT department_code, department_name
          FROM geoip_bo_networks
          WHERE %s::inet <<= network
          ORDER BY masklen(network) DESC
          LIMIT 1
          """,
          (str(parsed_ip),),
        )
        row = await cursor.fetchone()

    if not row:
      return None

    return GeoRegion(
      department_code=row["department_code"],
      department_name=row["department_name"],
    )

  async def record_ocr_inference(self, record: OcrInferenceRecord) -> None:
    if self._pool is None:
      raise TelemetryUnavailableError(self.unavailable_reason)

    async with self._pool.connection() as conn:
      async with conn.cursor() as cursor:
        await cursor.execute(
          """
          INSERT INTO ocr_inferences (
            request_id,
            device_id,
            session_id,
            page_load_id,
            denomination,
            torch_enabled,
            client_started_at,
            outcome,
            engine,
            latency_ms,
            raw_text,
            serial,
            series,
            confidence,
            candidates,
            response_payload,
            region_department_code,
            region_department_name
          ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
          )
          ON CONFLICT (request_id) DO NOTHING
          """,
          (
            record.request_id,
            record.device_id,
            record.session_id,
            record.page_load_id,
            int(record.denomination) if record.denomination else None,
            record.torch_enabled,
            record.client_started_at,
            record.outcome,
            record.engine,
            record.latency_ms,
            record.raw_text,
            record.serial,
            record.series,
            record.confidence,
            Jsonb(record.candidates),
            Jsonb(record.response_payload),
            record.region.department_code if record.region else None,
            record.region.department_name if record.region else None,
          ),
        )

  async def record_analytics_events(self, records: list[AnalyticsEventRecord]) -> None:
    if self._pool is None:
      raise TelemetryUnavailableError(self.unavailable_reason)

    if not records:
      return

    params = [
      (
        record.occurred_at,
        record.device_id,
        record.session_id,
        record.page_load_id,
        record.event_name,
        record.request_id,
        int(record.denomination) if record.denomination else None,
        record.method,
        record.outcome,
        record.app_version,
        record.device_class,
        record.browser_family,
        record.os_family,
        record.viewport_bucket,
        record.referrer_domain,
        Jsonb(record.meta),
      )
      for record in records
    ]

    async with self._pool.connection() as conn:
      async with conn.cursor() as cursor:
        await cursor.executemany(
          """
          INSERT INTO analytics_events (
            occurred_at,
            device_id,
            session_id,
            page_load_id,
            event_name,
            request_id,
            denomination,
            method,
            outcome,
            app_version,
            device_class,
            browser_family,
            os_family,
            viewport_bucket,
            referrer_domain,
            meta
          ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
          )
          """,
          params,
        )

  async def count_recent_feedback_by_device(self, device_id: UUID, *, hours: int) -> int:
    if self._pool is None:
      raise TelemetryUnavailableError(self.unavailable_reason)

    async with self._pool.connection() as conn:
      async with conn.cursor() as cursor:
        await cursor.execute(
          """
          SELECT COUNT(*)
          FROM feedback_entries
          WHERE device_id = %s
          AND received_at >= NOW() - (%s * INTERVAL '1 hour')
          """,
          (device_id, hours),
        )
        row = await cursor.fetchone()

    return int(row[0] if row else 0)

  async def record_feedback(self, record: FeedbackRecord) -> None:
    if self._pool is None:
      raise TelemetryUnavailableError(self.unavailable_reason)

    async with self._pool.connection() as conn:
      async with conn.cursor() as cursor:
        await cursor.execute(
          """
          INSERT INTO feedback_entries (
            device_id,
            session_id,
            page_load_id,
            request_id,
            rating,
            comment,
            prompted_after_scan_count
          ) VALUES (
            %s, %s, %s,
            (SELECT request_id FROM ocr_inferences WHERE request_id = %s),
            %s, %s, %s
          )
          """,
          (
            record.device_id,
            record.session_id,
            record.page_load_id,
            record.request_id,
            record.rating,
            record.comment,
            record.prompted_after_scan_count,
          ),
        )
