from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class InvalidImageError(Exception):
  pass


@dataclass(frozen=True)
class RecognizedCandidate:
  text: str
  confidence: float


@dataclass(frozen=True)
class OcrResult:
  raw_text: str
  serial: str | None
  candidates: list[RecognizedCandidate]
  raw_confidence: float | None = None


class OcrEngine(ABC):
  name: str

  @abstractmethod
  def recognize(self, image_bytes: bytes) -> OcrResult:
    raise NotImplementedError
