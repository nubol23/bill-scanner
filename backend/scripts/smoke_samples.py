from __future__ import annotations

from pathlib import Path
import os
import re
import sys

import httpx


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "imgs"


def main() -> int:
  if not FIXTURE_DIR.exists():
    print("Skipping smoke test: ../imgs is not available.")
    return 0

  base_url = os.environ.get("BILLETE_BACKEND_URL", "http://127.0.0.1:8000")
  failures = []

  with httpx.Client(base_url=base_url, timeout=12.0) as client:
    for image_path in sorted(FIXTURE_DIR.glob("*.png")):
      expected_match = re.search(r"(\d{8,9})", image_path.stem)
      if not expected_match:
        continue

      response = client.post(
        "/api/v1/recognize",
        files={"image": (image_path.name, image_path.read_bytes(), "image/png")},
      )
      payload = response.json()
      serial = payload.get("serial")
      expected = expected_match.group(1)
      print(f"{image_path.name}: {serial}")
      if response.status_code != 200 or serial != expected:
        failures.append((image_path.name, response.status_code, serial, expected))

  if failures:
    for file_name, status_code, serial, expected in failures:
      print(
        f"FAILED {file_name}: status={status_code} serial={serial!r} expected={expected!r}",
        file=sys.stderr,
      )
    return 1

  print("All local sample images matched.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
