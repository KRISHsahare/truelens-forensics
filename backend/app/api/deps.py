"""Route helpers: the app state, and the one error envelope.

Every failure the API produces — a route's own rejection, a validation error, a rate limit, an
unhandled crash — leaves through ``error_response`` here, so a client parses exactly one error shape.
That is not cosmetic: a 429 whose body is slowapi's plain-text default and a 415 whose body is
FastAPI's ``{"detail": ...}`` are two more branches in the frontend, and the branch it forgets is the
one that shows the user ``[object Object]``.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.core.cache import ResultCache
from app.core.orchestrator import Orchestrator, Rejected
from app.core.ratelimit import client_ip
from app.core.signing import OverlaySigner
from app.schemas.common import Error

__all__ = ["error_response", "get_cache", "get_orchestrator", "get_settings", "get_signer",
           "rejected_response"]


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator  # type: ignore[no-any-return]


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_signer(request: Request) -> OverlaySigner:
    return request.app.state.signer  # type: ignore[no-any-return]


def get_cache(request: Request) -> ResultCache:
    return request.app.state.cache  # type: ignore[no-any-return]


def error_response(
    status: int,
    code: str,
    message: str,
    *,
    detail: str | None = None,
    retry_after: int | None = None,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the single error body. Validates through ``Error`` so a typo in a code is caught here.

    An unknown ``code`` raises rather than silently widening the enum: the enum is the contract the
    frontend switches on, and a value outside it would surface as an unhandled branch in production.
    """
    body = Error(
        error=code, message=message, detail=detail,
        retry_after_seconds=retry_after, extra=extra)
    out_headers = dict(headers or {})
    if retry_after is not None:
        out_headers["Retry-After"] = str(int(retry_after))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"),
                        headers=out_headers or None)


def rejected_response(exc: Rejected) -> JSONResponse:
    """The orchestrator's rejection, in wire form."""
    return error_response(
        exc.status, exc.code, exc.message,
        detail=exc.detail, retry_after=exc.retry_after, extra=exc.extra or None)


def rate_key(request: Request) -> str:
    """The per-client label used in log lines. Same rule the limiter uses, so they agree."""
    return client_ip(request)
