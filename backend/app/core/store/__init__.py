"""Job store selection.

``build_store`` is the only place that knows which backend is in play, so the rest of the app holds
a ``JobStore`` and never branches. The ``redis`` module is imported inside the branch: naming it at
module scope would make a memory-backend deployment fail to boot when the optional extra is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.store.base import STORE_KEY_PREFIX, JobStore
from app.core.store.memory import MemoryJobStore

if TYPE_CHECKING:
    from app.config import Settings

__all__ = ["STORE_KEY_PREFIX", "JobStore", "MemoryJobStore", "build_store"]


def build_store(settings: Settings) -> JobStore:
    """Construct the backend named by ``STORE_BACKEND``.

    Raises ``RuntimeError``/``ValueError`` for a misconfiguration. That is deliberate: a deployment
    that asks for redis and silently falls back to per-process state works on one worker and
    produces mystery-404s on two, which is far worse to debug than a boot that refuses.
    """
    backend = settings.store_backend
    if backend == "memory":
        return MemoryJobStore(max_entries=settings.job_store_max_entries)
    if backend == "redis":
        from app.core.store.redis import RedisJobStore

        return RedisJobStore(settings.redis_url, ttl_s=settings.job_ttl_s)
    raise ValueError(f"unknown STORE_BACKEND: {backend!r}")
