"""``GET /api/health`` — liveness plus the four things an operator checks first.

This is deliberately *not* a bare ``{"status": "ok"}``. When a deployment misbehaves the questions
are always the same: is the store answering, did the trained artifacts actually get baked into the
image, how many jobs are in flight, and is the process that answered this request the one that owns
the job I am polling. Every one of those is a field here, so a screenshot of ``/api/health`` is a bug
report.

Nothing in this response is secret. It names artifact *paths* — which are container paths already
fixed by the Dockerfile — and counts, never the signing key, the redis URL, or any client address.

The route is unauthenticated and uncached by design: a load balancer has to be able to ask it, and a
cached answer is worse than no answer when the thing being checked is whether the box is alive.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Request

from app.api.deps import get_cache, get_orchestrator, get_settings
from app.config import Settings
from app.core.store import JobStore
from app.schemas.common import Health

__all__ = ["app_version", "router"]

log = logging.getLogger("truelens.api.health")

router = APIRouter(prefix="/api", tags=["health"])

_VERSION_FALLBACK = "1.0.0"


def app_version() -> str:
    """The installed package version, or the fallback.

    ``importlib.metadata`` is the only trustworthy source once this is pip-installed in a container,
    because the pyproject file is not necessarily in the image. The fallback is a string rather than
    a raise because a health endpoint that 500s over a missing metadata file defeats its own purpose.
    """
    try:
        from importlib.metadata import version

        return version("truelens-backend")
    except Exception:
        return _VERSION_FALLBACK


@router.get("/health", response_model=Health, summary="Liveness, capacity and artifact presence")
async def health(request: Request) -> Health:
    settings: Settings = get_settings(request)
    orch = get_orchestrator(request)
    problems: list[str] = []

    store_stats = await _store_stats(orch.store, settings, problems)
    cache_stats: dict[str, Any] = {}
    try:
        cache_stats = get_cache(request).stats()
    except Exception:
        problems.append("result cache is not answering")

    for message in settings.validate_prod():
        problems.append(message)

    return Health(
        status="degraded" if problems else "ok",
        app=settings.app_name,
        version=app_version(),
        environment=settings.environment,
        problems=problems,
        perf_profile=settings.perf_profile,
        store=store_stats,
        cache=cache_stats,
        concurrency=orch.stats(),
        artifacts=_artifact_flags(settings),
        network={"cors_origins": settings.cors_origins, "trust_proxy": settings.trust_proxy},
        uptime_s=orch.uptime_s,
    )


async def _store_stats(
    store: JobStore, settings: Settings, problems: list[str]
) -> dict[str, Any]:
    """Store counters, or a problem statement.

    A store that will not answer is ``degraded`` rather than dead: reads of finished jobs fail, but
    an in-flight analysis still completes and an already-cached upload still serves, so the process
    is genuinely half-working and the health check should say so in those words.
    """
    try:
        return dict(await store.stats())
    except Exception as exc:
        log.warning("store stats failed: %s", type(exc).__name__)
        problems.append("job store is not answering")
        return {"backend": settings.store_backend, "reachable": False}


def _artifact_flags(settings: Settings) -> dict[str, bool]:
    """Which trained artifacts are present *right now*.

    Presence is not correctness — a corrupt or mismatched checkpoint exists too — but the two states
    an operator can distinguish from here are "the image was built without weights" and "the weights
    are there and the CNN still says unavailable", and the second one points at the log while the
    first points at the Dockerfile.
    """
    paths = {
        "cnn_weights": settings.cnn_weights_path,
        "fusion_model": settings.fusion_path,
        "metrics": settings.metrics_path,
        "clip_probe": settings.clip_probe_path if settings.enable_clip_probe else "",
        "face_model": settings.yunet_model_path,
    }
    return {k: bool(v) and os.path.exists(v) for k, v in paths.items()}
