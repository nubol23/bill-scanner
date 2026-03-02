from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import Settings


def install_http_middleware(app: FastAPI, settings: Settings) -> None:
  app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
  )

  @app.middleware("http")
  async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response
