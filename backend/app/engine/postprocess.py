from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import RecognizedCandidate


CONFUSABLE_DIGIT_MAP = {
  "O": "0",
  "Q": "0",
  "D": "0",
  "I": "1",
  "L": "1",
  "|": "1",
  "Z": "2",
  "S": "5",
  "G": "6",
  "B": "8",
}
CONFUSABLE_DIGIT_PENALTY = 0.05


@dataclass(frozen=True)
class DecodedSequence:
  text: str
  confidence: float
  char_confidences: tuple[float, ...]


def load_characters(dict_path: Path, *, use_space_char: bool) -> list[str]:
  characters = dict_path.read_text(encoding="utf-8").splitlines()
  if use_space_char:
    characters.append(" ")
  return ["blank", *characters]


def decode_ctc(prediction: np.ndarray, characters: list[str]) -> DecodedSequence:
  indices = prediction.argmax(axis=2)
  probabilities = prediction.max(axis=2)

  kept_chars: list[str] = []
  kept_confidences: list[float] = []
  previous: int | None = None

  for index, confidence in zip(indices[0], probabilities[0], strict=True):
    token = int(index)
    if token == 0:
      previous = token
      continue
    if kept_chars and token == previous:
      previous = token
      continue
    previous = token
    if token >= len(characters):
      continue
    kept_chars.append(characters[token])
    kept_confidences.append(float(confidence))

  average_confidence = float(np.mean(kept_confidences)) if kept_confidences else 0.0

  return DecodedSequence(
    text="".join(kept_chars),
    confidence=average_confidence,
    char_confidences=tuple(kept_confidences),
  )


def normalize_confusable_text(text: str) -> str:
  normalized_characters: list[str] = []

  for character in text:
    normalized_characters.append(_normalize_character(character)[0])

  return "".join(normalized_characters)


def _normalize_character(character: str) -> tuple[str, bool]:
  if character.isdigit():
    return character, False

  upper_character = character.upper()
  replacement = CONFUSABLE_DIGIT_MAP.get(upper_character)
  if replacement is not None:
    return replacement, True

  return upper_character, False


def _extract_series_letter(text: str, start_index: int) -> str | None:
  series_letter: str | None = None

  for character in text[start_index:]:
    if character.isspace():
      continue
    if series_letter is not None:
      return None
    if not character.isalpha():
      return None
    series_letter = character.upper()

  return series_letter


def extract_serial_candidates(
  text: str,
  char_confidences: tuple[float, ...],
  *,
  min_digits: int = 8,
  max_digits: int = 9,
) -> list[RecognizedCandidate]:
  normalized_rows = [_normalize_character(character) for character in text]
  normalized_text = "".join(value for value, _ in normalized_rows)
  candidate_rows: dict[str, tuple[float, float, int, int, int, str | None]] = {}

  run_start = -1

  for index, (character, _) in enumerate(normalized_rows + [(" ", False)]):
    if character.isdigit():
      if run_start < 0:
        run_start = index
      continue
    if run_start < 0:
      continue

    run_end = index
    run_length = run_end - run_start
    if run_length < min_digits:
      run_start = -1
      continue

    candidate_lengths: tuple[int, ...]
    if run_length >= max_digits:
      candidate_lengths = (max_digits,)
    else:
      candidate_lengths = (run_length,)

    for candidate_length in candidate_lengths:
      for offset in range(0, run_length - candidate_length + 1):
        start = run_start + offset
        end = start + candidate_length
        digits = normalized_text[start:end]
        confidence_slice = char_confidences[start:end]
        if not confidence_slice:
          continue

        candidate_confidence = float(sum(confidence_slice) / len(confidence_slice))
        confusable_count = sum(1 for _, was_confusable in normalized_rows[start:end] if was_confusable)
        effective_confidence = candidate_confidence - (confusable_count * CONFUSABLE_DIGIT_PENALTY)
        series = _extract_series_letter(text, end)

        previous = candidate_rows.get(digits)
        if previous:
          previous_effective_confidence, previous_confidence, _, _, _, previous_series = previous
          if previous_effective_confidence > effective_confidence:
            continue
          if previous_effective_confidence == effective_confidence:
            if previous_confidence > candidate_confidence:
              continue
            if (
              previous_confidence == candidate_confidence
              and bool(previous_series)
              and not series
            ):
              continue
        candidate_rows[digits] = (
          effective_confidence,
          candidate_confidence,
          candidate_length,
          confusable_count,
          start,
          series,
        )

    run_start = -1

  sorted_candidates = sorted(
    candidate_rows.items(),
    key=lambda item: (-item[1][0], -item[1][2], item[1][3], item[1][4]),
  )

  return [
    RecognizedCandidate(text=text, confidence=confidence, series=series)
    for text, (_, confidence, _, _, _, series) in sorted_candidates[:5]
  ]
