from __future__ import annotations

import argparse
import csv
import gzip
from io import BytesIO, StringIO
import os
from pathlib import Path
import sys
from ipaddress import ip_address, summarize_address_range
from urllib.parse import urlparse
import zipfile

import httpx
import psycopg


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from app.settings import Settings


DEFAULT_ENV_SOURCE_KEY = "APP_GEOIP_SOURCE_URL"
REQUIRED_NORMALIZED_COLUMNS = {
  "network",
  "department_code",
  "department_name",
}
DBIP_BOLIVIA_STATE_MAP = {
  "Beni Department": ("BE", "Beni"),
  "El Beni": ("BE", "Beni"),
  "Chuquisaca": ("CH", "Chuquisaca"),
  "Chuquisaca Department": ("CH", "Chuquisaca"),
  "Cochabamba": ("CB", "Cochabamba"),
  "La Paz": ("LP", "La Paz"),
  "La Paz Department": ("LP", "La Paz"),
  "Oruro": ("OR", "Oruro"),
  "Pando": ("PD", "Pando"),
  "Potosí Department": ("PT", "Potosi"),
  "Santa Cruz": ("SC", "Santa Cruz"),
  "Santa Cruz Department": ("SC", "Santa Cruz"),
  "Tarija Department": ("TJ", "Tarija"),
}


def _is_url(source: str) -> bool:
  parsed = urlparse(source)
  return parsed.scheme in {"http", "https"}


def _resolve_source() -> str:
  parser = argparse.ArgumentParser(
    description=(
      "Import GeoIP data into geoip_bo_networks. Supports either a normalized "
      "Bolivia CSV or the raw DB-IP City Lite monthly feed."
    )
  )
  parser.add_argument(
    "source",
    nargs="?",
    help=(
      "Local CSV/ZIP/GZ path or HTTP(S) URL. "
      f"If omitted, {DEFAULT_ENV_SOURCE_KEY} is used."
    ),
  )
  args = parser.parse_args()

  if args.source:
    return args.source

  source_from_env = os.environ.get(DEFAULT_ENV_SOURCE_KEY)
  if source_from_env:
    return source_from_env

  raise SystemExit(
    "No GeoIP source was provided. Pass a path/URL or set "
    f"{DEFAULT_ENV_SOURCE_KEY}."
  )


def _download_source(source_url: str) -> tuple[bytes, str]:
  with httpx.Client(follow_redirects=True, timeout=300.0) as client:
    response = client.get(source_url)
    response.raise_for_status()

  parsed = urlparse(source_url)
  file_name = Path(parsed.path).name or "geoip_source.csv"
  return response.content, file_name


def _read_local_source(source_path: str) -> tuple[bytes, str]:
  resolved_path = Path(source_path).expanduser().resolve()
  if not resolved_path.exists():
    raise FileNotFoundError(f"File not found: {resolved_path}")

  return resolved_path.read_bytes(), resolved_path.name


def _extract_csv_text(payload: bytes, source_name: str) -> str:
  lower_name = source_name.lower()

  if lower_name.endswith(".zip"):
    with zipfile.ZipFile(BytesIO(payload)) as archive:
      csv_members = sorted(
        member
        for member in archive.namelist()
        if member.lower().endswith(".csv") and not member.endswith("/")
      )
      if not csv_members:
        raise ValueError("ZIP archive does not contain a CSV file.")

      preferred_member = next(
        (
          member
          for member in csv_members
          if "geoip_bo" in Path(member).name.lower()
        ),
        csv_members[0],
      )
      return archive.read(preferred_member).decode("utf-8-sig")

  if lower_name.endswith(".gz"):
    return gzip.decompress(payload).decode("utf-8-sig")

  return payload.decode("utf-8-sig")


def _try_parse_normalized_rows(csv_text: str) -> list[dict[str, str]] | None:
  reader = csv.DictReader(StringIO(csv_text))
  if reader.fieldnames is None:
    return None

  if not REQUIRED_NORMALIZED_COLUMNS.issubset(reader.fieldnames):
    return None

  rows: list[dict[str, str]] = []
  for row in reader:
    if not row.get("network"):
      continue

    rows.append(
      {
        "network": row["network"],
        "department_code": row["department_code"],
        "department_name": row["department_name"],
        "source": row.get("source") or "normalized",
      }
    )

  return rows


def _parse_dbip_city_rows(csv_text: str) -> list[dict[str, str]]:
  reader = csv.reader(StringIO(csv_text))
  normalized_rows: list[dict[str, str]] = []

  for index, row in enumerate(reader, start=1):
    if len(row) < 5:
      continue

    if row[3] != "BO":
      continue

    state_name = row[4].strip()
    if not state_name:
      continue

    department = DBIP_BOLIVIA_STATE_MAP.get(state_name)
    if department is None:
      raise ValueError(
        f"Unmapped Bolivia department name in DB-IP feed on row {index}: {state_name!r}"
      )

    start_ip = ip_address(row[0].strip())
    end_ip = ip_address(row[1].strip())
    department_code, department_name = department

    for network in summarize_address_range(start_ip, end_ip):
      normalized_rows.append(
        {
          "network": str(network),
          "department_code": department_code,
          "department_name": department_name,
          "source": "dbip-city-lite",
        }
      )

  return normalized_rows


def _read_rows(source: str) -> list[dict[str, str]]:
  if _is_url(source):
    payload, source_name = _download_source(source)
  else:
    payload, source_name = _read_local_source(source)

  csv_text = _extract_csv_text(payload, source_name)

  normalized_rows = _try_parse_normalized_rows(csv_text)
  if normalized_rows is not None:
    return normalized_rows

  return _parse_dbip_city_rows(csv_text)


def main() -> int:
  source = _resolve_source()
  settings = Settings.from_env()
  if not settings.database_url:
    print("APP_DATABASE_URL is not configured.", file=sys.stderr)
    return 1

  try:
    rows = _read_rows(source)
  except Exception as error:
    print(f"Unable to load GeoIP source: {error}", file=sys.stderr)
    return 1

  if not rows:
    print("No rows were found in the GeoIP source.", file=sys.stderr)
    return 1

  with psycopg.connect(settings.database_url) as connection:
    with connection.cursor() as cursor:
      cursor.execute(
        """
        CREATE TEMP TABLE temp_geoip_bo_networks_import (
          network CIDR NOT NULL,
          department_code TEXT NOT NULL,
          department_name TEXT NOT NULL,
          source TEXT NOT NULL
        ) ON COMMIT DROP
        """
      )

      with cursor.copy(
        """
        COPY temp_geoip_bo_networks_import (
          network,
          department_code,
          department_name,
          source
        ) FROM STDIN
        """
      ) as copy:
        for row in rows:
          copy.write_row(
            (
              row["network"],
              row["department_code"],
              row["department_name"],
              row.get("source", "manual"),
            )
          )

      cursor.execute("TRUNCATE TABLE geoip_bo_networks")
      cursor.execute(
        """
        INSERT INTO geoip_bo_networks (
          network,
          department_code,
          department_name,
          source
        )
        SELECT
          network,
          department_code,
          department_name,
          source
        FROM temp_geoip_bo_networks_import
        ON CONFLICT (network) DO UPDATE
        SET department_code = EXCLUDED.department_code,
            department_name = EXCLUDED.department_name,
            source = EXCLUDED.source,
            updated_at = NOW()
        """
      )

    connection.commit()

  print(f"Imported {len(rows)} GeoIP rows from {source}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
