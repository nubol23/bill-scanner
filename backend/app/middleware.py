from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .settings import Settings


logger = logging.getLogger(__name__)


def install_http_middleware(app: FastAPI, settings: Settings) -> None:
  app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Request-Id"],
  )

  @app.middleware("http")
  async def assign_request_id(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers.setdefault("X-Request-Id", request_id)
    response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response

  @app.exception_handler(RequestValidationError)
  async def handle_validation_error(request: Request, exc: RequestValidationError):
    response = JSONResponse(
      status_code=400,
      content={"detail": "Invalid request payload."},
    )
    response.headers.setdefault("X-Request-Id", getattr(request.state, "request_id", str(uuid4())))
    response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response

  @app.exception_handler(Exception)
  async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    response = JSONResponse(
      status_code=500,
      content={"detail": "Internal server error."},
    )
    response.headers.setdefault("X-Request-Id", getattr(request.state, "request_id", str(uuid4())))
    response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response
