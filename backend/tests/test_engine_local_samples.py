from __future__ import annotations

from pathlib import Path
import re

import pytest

from app.engine.onnx_recognizer import OnnxSerialRecognizer
from app.settings import Settings


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "imgs"


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="Local sample images are not available.")
def test_local_sample_images_match_filenames():
  engine = OnnxSerialRecognizer(Settings.from_env())

  for image_path in sorted(FIXTURE_DIR.glob("*.png")):
    expected_match = re.search(r"(\d{8,9})", image_path.stem)
    assert expected_match is not None
    result = engine.recognize(image_path.read_bytes())
    assert result.serial == expected_match.group(1), image_path.name
