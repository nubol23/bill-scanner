from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from time import monotonic


class RequestCapacityExceeded(Exception):
  pass


class RateLimiter:
  def __init__(
    self,
    *,
    burst_limit: int,
    burst_window_seconds: int,
    sustained_limit: int,
    sustained_window_seconds: int = 60,
  ) -> None:
    self._burst_limit = burst_limit
    self._burst_window_seconds = burst_window_seconds
    self._sustained_limit = sustained_limit
    self._sustained_window_seconds = sustained_window_seconds
    self._burst_buckets: dict[str, deque[float]] = defaultdict(deque)
    self._sustained_buckets: dict[str, deque[float]] = defaultdict(deque)
    self._lock = Lock()

  def allow(self, key: str) -> bool:
    if self._is_loopback(key):
      return True

    now = monotonic()
    with self._lock:
      burst_bucket = self._burst_buckets[key]
      sustained_bucket = self._sustained_buckets[key]

      self._prune(burst_bucket, now, self._burst_window_seconds)
      self._prune(sustained_bucket, now, self._sustained_window_seconds)

      if len(burst_bucket) >= self._burst_limit or len(sustained_bucket) >= self._sustained_limit:
        return False

      burst_bucket.append(now)
      sustained_bucket.append(now)
      return True

  @staticmethod
  def _is_loopback(key: str) -> bool:
    return key in {"127.0.0.1", "::1", "localhost"}

  @staticmethod
  def _prune(bucket: deque[float], now: float, window_seconds: int) -> None:
    cutoff = now - window_seconds
    while bucket and bucket[0] <= cutoff:
      bucket.popleft()


@dataclass
class RequestQueue:
  max_pending: int
  max_inflight: int = 1

  def __post_init__(self) -> None:
    self._semaphore = asyncio.Semaphore(self.max_inflight)
    self._pending = 0
    self._lock = asyncio.Lock()

  @asynccontextmanager
  async def acquire(self) -> AsyncIterator[None]:
    async with self._lock:
      if self._pending >= self.max_pending:
        raise RequestCapacityExceeded("OCR queue is full.")
      self._pending += 1

    try:
      await self._semaphore.acquire()
      try:
        yield
      finally:
        self._semaphore.release()
    finally:
      async with self._lock:
        self._pending -= 1
