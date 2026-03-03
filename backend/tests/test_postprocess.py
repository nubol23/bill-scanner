from __future__ import annotations

from app.engine.postprocess import extract_serial_candidates


def test_extract_serial_candidates_preserves_trailing_series_letter():
  text = "058252703 B"
  confidences = tuple(0.99 for _ in text)

  result = extract_serial_candidates(text, confidences)

  assert result
  assert result[0].text == "058252703"
  assert result[0].series == "B"


def test_extract_serial_candidates_preserves_trailing_series_letter_without_space():
  text = "058252703B"
  confidences = tuple(0.99 for _ in text)

  result = extract_serial_candidates(text, confidences)

  assert result
  assert result[0].text == "058252703"
  assert result[0].series == "B"


def test_extract_serial_candidates_omits_series_when_no_letter_is_detected():
  text = "058252703"
  confidences = tuple(0.99 for _ in text)

  result = extract_serial_candidates(text, confidences)

  assert result
  assert result[0].text == "058252703"
  assert result[0].series is None
