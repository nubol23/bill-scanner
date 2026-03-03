from __future__ import annotations

from app.settings import Settings


def test_default_allowed_origins_are_production_only(monkeypatch):
  monkeypatch.delenv("APP_ALLOWED_ORIGINS", raising=False)

  settings = Settings.from_env()

  assert settings.allowed_origins == ("https://billete.rafaelvillca.site",)


def test_allowed_origins_can_be_overridden_from_env(monkeypatch):
  monkeypatch.setenv(
    "APP_ALLOWED_ORIGINS",
    "http://127.0.0.1:4175,http://localhost:4175",
  )

  settings = Settings.from_env()

  assert settings.allowed_origins == (
    "http://127.0.0.1:4175",
    "http://localhost:4175",
  )
