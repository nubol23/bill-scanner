from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import onnxruntime as ort

from .base import OcrEngine, OcrResult, RecognizedCandidate
from .postprocess import decode_ctc, extract_serial_candidates, load_characters
from .preprocess import build_candidate_images, load_image_bytes, prepare_onnx_input
from ..settings import Settings


@dataclass(frozen=True)
class ModelMetadata:
  engine_name: str
  input_height: int
  min_width: int
  channel_order: str
  mean: tuple[float, float, float]
  std: tuple[float, float, float]
  pad_value: float
  top_crop_ratio: float
  use_space_char: bool
  charset_path: Path

  @classmethod
  def load(cls, path: Path) -> "ModelMetadata":
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cls(
      engine_name=payload["engine_name"],
      input_height=int(payload["input_height"]),
      min_width=int(payload["min_width"]),
      channel_order=payload["channel_order"],
      mean=tuple(float(value) for value in payload["mean"]),
      std=tuple(float(value) for value in payload["std"]),
      pad_value=float(payload["pad_value"]),
      top_crop_ratio=float(payload["top_crop_ratio"]),
      use_space_char=bool(payload["use_space_char"]),
      charset_path=(path.parent / payload["charset_path"]).resolve(),
    )


@dataclass(frozen=True)
class _ViewResult:
  name: str
  raw_text: str
  raw_confidence: float
  candidates: list[RecognizedCandidate]


class OnnxSerialRecognizer(OcrEngine):
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._metadata = ModelMetadata.load(settings.model_meta_path)
    self._characters = load_characters(
      self._metadata.charset_path,
      use_space_char=self._metadata.use_space_char,
    )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = settings.cpu_threads
    session_options.inter_op_num_threads = 1

    self._session = ort.InferenceSession(
      str(settings.model_path),
      sess_options=session_options,
      providers=["CPUExecutionProvider"],
    )
    self._input_name = self._session.get_inputs()[0].name
    self._output_name = self._session.get_outputs()[0].name
    self.name = self._metadata.engine_name

  def recognize(self, image_bytes: bytes) -> OcrResult:
    image = load_image_bytes(image_bytes)
    view_results: list[_ViewResult] = []

    for prepared in build_candidate_images(image, top_crop_ratio=self._metadata.top_crop_ratio):
      tensor = prepare_onnx_input(
        prepared.image,
        image_height=self._metadata.input_height,
        min_width=self._metadata.min_width,
        channel_order=self._metadata.channel_order,
        mean=self._metadata.mean,
        std=self._metadata.std,
        pad_value=self._metadata.pad_value,
      )
      prediction = self._session.run([self._output_name], {self._input_name: tensor})[0]
      decoded = decode_ctc(prediction, self._characters)
      candidates = extract_serial_candidates(decoded.text, decoded.char_confidences)
      view_results.append(
        _ViewResult(
          name=prepared.name,
          raw_text=decoded.text,
          raw_confidence=decoded.confidence,
          candidates=candidates,
        )
      )

    serial_candidates = self._merge_candidates(view_results)
    if serial_candidates:
      best_serial = serial_candidates[0]
      source_view = self._find_source_view(view_results, best_serial.text)
      return OcrResult(
        raw_text=source_view.raw_text,
        serial=best_serial.text,
        candidates=serial_candidates,
        raw_confidence=best_serial.confidence,
      )

    fallback_view = max(view_results, key=lambda item: item.raw_confidence)
    return OcrResult(
      raw_text=fallback_view.raw_text,
      serial=None,
      candidates=[],
      raw_confidence=None,
    )

  @staticmethod
  def _merge_candidates(view_results: list[_ViewResult]) -> list[RecognizedCandidate]:
    best_by_text: dict[str, tuple[int, RecognizedCandidate]] = {}

    for view_result in view_results:
      for rank, candidate in enumerate(view_result.candidates):
        current = best_by_text.get(candidate.text)
        if current and current[0] < rank:
          continue
        if current and current[0] == rank and current[1].confidence >= candidate.confidence:
          continue
        best_by_text[candidate.text] = (rank, candidate)

    ordered_rows = sorted(
      best_by_text.values(),
      key=lambda row: (row[0], -len(row[1].text), -row[1].confidence, row[1].text),
    )

    return [row[1] for row in ordered_rows]

  @staticmethod
  def _find_source_view(view_results: list[_ViewResult], serial: str) -> _ViewResult:
    for view_result in view_results:
      if any(candidate.text == serial for candidate in view_result.candidates):
        return view_result
    return view_results[0]
