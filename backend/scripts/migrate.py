from __future__ import annotations

from pathlib import Path
import sys

import psycopg


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from app.settings import Settings


MIGRATIONS_DIR = ROOT / "migrations"


def main() -> int:
  settings = Settings.from_env()
  if not settings.database_url:
    print("APP_DATABASE_URL is not configured.", file=sys.stderr)
    return 1

  migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
  if not migration_files:
    print("No migration files found.")
    return 0

  with psycopg.connect(settings.database_url, autocommit=True) as connection:
    with connection.cursor() as cursor:
      cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY,
          applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
      )

      cursor.execute("SELECT version FROM schema_migrations")
      applied_versions = {row[0] for row in cursor.fetchall()}

      for migration_file in migration_files:
        version = migration_file.name
        if version in applied_versions:
          continue

        cursor.execute(migration_file.read_text())
        cursor.execute(
          "INSERT INTO schema_migrations (version) VALUES (%s)",
          (version,),
        )
        print(f"Applied {version}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
