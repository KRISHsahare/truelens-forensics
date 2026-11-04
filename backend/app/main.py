"""The application: what exists when the process starts, and what it refuses to do.

Startup order is a correctness question, not a style one:

1. **Validate the settings.** ``validate_prod`` returns the configurations that cannot work — a
   result cache that outlives the files it references, two workers sharing an in-memory job store.
   In production these are fatal and the process refuses to boot, because both failure modes look
   like an intermittent bug in someone else's code: cache hits 410ing on their overlays, jobs 404ing
   on whichever worker did not create them.
2. **Build the state** (scratch, store, signer, cache) before the orchestrator, which takes all four.
3. **Start the pools.** ``orchestrator.startup()`` spawns the workers and loads torch, so the first
   request is not the one that pays for it.
4. **Then the sweeper.** It runs against state that exists, and it is cancelled before shutdown so a
   sweep cannot delete the job the last request is still reading.

The exception handlers exist so that *nothing* returns FastAPI's default ``{"detail": ...}``. A
client that has to handle two error grammars will handle the wrong one, and the unhandled-crash
handler is also the only place where a traceback is guaranteed not to reach the network.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.analysis.registry import TECHNIQUES
from app.api import (
    routes_analyze,
    routes_batch,
    routes_health,
    routes_jobs,
    routes_metrics,
    routes_overlays,
    routes_report,
)
from app.api.deps import error_response
from app.api.limits import limiter
from app.config import Settings, get_settings
from app.core.cache import ResultCache
from app.core.logconfig import configure_logging
from app.core.orchestrator import Orchestrator, Rejected
from app.core.ratelimit import rate_limit_exceeded_handler
from app.core.scratch import Scratch
from app.core.signing import OverlaySigner
from app.core.store import build_store

__all__ = ["app", "create_app"]

log = logging.getLogger("truelens.main")

_TAG_METADATA = [
    {"name": "analyze", "description": "Start an analysis from an upload or a URL."},
    {"name": "jobs", "description": "Poll a job for live progress and its final result."},
    {"name": "report", "description": "Render a finished job's result as a downloadable PDF."},
    {"name": "batch", "description": "Queue up to BATCH_MAX_IMAGES analyses in one request."},
    {"name": "overlays", "description": "Fetch a signed, expiling heatmap PNG."},
    {"name": "metrics", "description": "Model evaluation data for the Methodology page."},
    {"name": "health", "description": "Liveness, capacity, store and artifact status."},
]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    problems = settings.validate_prod()
    for message in problems:
        log.error("configuration problem: %s", message)
    if problems and settings.environment == "prod":
        raise RuntimeError(
            "Refusing to start with an impossible configuration: " + " ".join(problems))

    scratch = Scratch(settings.scratch_root or None)
    store = build_store(settings)
    signer = OverlaySigner(settings.overlay_signing_key, ttl_s=settings.overlay_ttl_s)
    cache = ResultCache(ttl_s=settings.result_cache_ttl_s,
                        max_entries=settings.result_cache_max_entries)
    orch = Orchestrator(settings, store=store, cache=cache, signer=signer, scratch=scratch)

    app.state.scratch = scratch
    app.state.store = store
    app.state.signer = signer
    app.state.cache = cache
    app.state.orchestrator = orch

    await orch.startup()
    sweeper = asyncio.create_task(_sweep_forever(orch, settings), name="truelens-sweeper")
    log.info("%s %s listening (profile=%s, store=%s, parallel=%s)",
             settings.app_name, settings.environment, settings.perf_profile,
             settings.store_backend, settings.analyzer_parallel)
    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except (asyncio.CancelledError, Exception):
            pass
        await orch.shutdown()
        await store.close()
        # Deleting the tree is the promise the privacy page makes: uploads and overlays do not
        # outlive the process. ``discard`` only removes a root this run created, so a configured
        # SCRATCH_ROOT that predates it is left where the operator put it.
        scratch.discard()


async def _sweep_forever(orch: Orchestrator, settings: Settings) -> None:
    """Delete expired jobs and overlays until cancelled.

    The TTL alone is not enough: a record is only *read* on a poll, and nothing reads a job that was
    abandoned. Without a sweeper the store's own capacity bound is the only thing keeping a
    long-running container honest, and the scratch directory would grow one folder per analysis
    forever.
    """
    interval = max(5, int(settings.sweeper_interval_s))
    while True:
        try:
            await asyncio.sleep(interval)
            records, directories = await orch.sweep()
            if records or directories:
                log.info("swept %d job record(s) and %d overlay dir(s)", records, directories)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("sweep iteration failed")
            await asyncio.sleep(interval)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=routes_health.app_version(),
        description=(
            f"Image authenticity analysis: {len(TECHNIQUES)} forensic techniques, calibrated "
            "fusion, and an explanation for every number. Results are probabilities, not proof."
        ),
        # Docs off in prod. They are accurate, and an accurate description of every endpoint and its
        # limits is a free map for anyone probing the deployment.
        docs_url=None if settings.environment == "prod" else "/api/docs",
        redoc_url=None,
        openapi_url=None if settings.environment == "prod" else "/api/openapi.json",
        lifespan=_lifespan,
        default_response_class=ORJSONResponse,
    )
    app.state.settings = settings

    # --- middleware -------------------------------------------------------------
    # Locked to the configured frontend origins, and *not* "*" with credentials: an allowlist that
    # contains the credentialed wildcard is the one CORS configuration that actually leaks. Nothing
    # here authenticates, so credentials are off entirely rather than merely restricted.
    origins = list(settings.cors_origins)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Accept", "Content-Type", "X-Requested-With"],
            # Overlay PNGs are fetched by <img> tags from another origin; without this the browser
            # cannot read the image back and every heatmap is a broken thumbnail in production.
            expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
            max_age=600,
        )
    app.state.limiter = limiter
    # No ASGI middleware: every limit here is declared per route, and the default limits are empty.
    # A middleware would re-check the same windows a second time per request for no additional
    # protection, and would make a 429 arrive from a layer the route does not know about.

    # --- handlers ---------------------------------------------------------------
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_exception_handler(Rejected, _rejected_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unhandled_handler)

    # --- routes -----------------------------------------------------------------
    for module in (routes_analyze, routes_batch, routes_jobs, routes_overlays, routes_report,
                   routes_metrics, routes_health):
        app.include_router(module.router)
    return app


async def _rejected_handler(request: Request, exc: Rejected) -> Any:
    return error_response(exc.status, exc.code, exc.message, detail=exc.detail,
                          retry_after=exc.retry_after, extra=exc.extra or None)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> Any:
    """FastAPI's own 404/405/413, in the envelope.

    A 404 from the router means "no such route"; a 404 from a job lookup means "no such job". Both
    are returned as ``not_found`` with the generic body, because a route list is not something a
    client needs to be told about and the alternative is a distinct error string to probe for.
    """
    code = {404: "not_found", 405: "not_found", 413: "too_large"}.get(
        exc.status_code, "internal" if exc.status_code >= 500 else "not_found")
    message = {
        404: "No such endpoint, job, or file.",
        405: "That method is not allowed on this endpoint.",
        413: "That upload is larger than the server accepts.",
    }.get(exc.status_code, str(exc.detail or "Request failed."))
    return error_response(exc.status_code, code, message)


async def _validation_handler(request: Request, exc: RequestValidationError) -> Any:
    """A malformed request, without echoing the value that was malformed.

    ``exc.errors()`` contains ``input``, which here is exactly the user-supplied string — and a
    response body that reflects input is a reflected-XSS helper. Only the location, the expectation,
    and a plain sentence are returned.
    """
    fields = sorted({
        # ``loc`` starts with the request section ("body", "query", "path"), so that first element is
        # dropped and the rest is the field path the client sent. Dropping the last element too — the
        # common reading of this slice — would leave "body" as the only thing ever reported, which
        # tells the integrator nothing about which field was wrong.
        ".".join(str(part) for part in err.get("loc", ())[1:]) or "body"
        for err in exc.errors()
    })
    return error_response(
        422, "invalid_image", "The request did not match what this endpoint accepts.",
        detail="Check the field names: " + ", ".join(f for f in fields if f)[:200],
        extra={"fields": fields})


async def _unhandled_handler(request: Request, exc: Exception) -> Any:
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return error_response(
        500, "internal",
        "The server could not complete this request.",
        detail="The failure was logged. Retrying may help if it was a busy moment.")


app = create_app()
