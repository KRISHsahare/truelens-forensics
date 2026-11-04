"""The job store contract.

**Why snapshots and not a mutate callback.** A technique finishing inside a ``ProcessPool`` worker
fires a progress callback from a non-async thread. If that callback had to await a store write, the
orchestrator would need a loop handle in every worker thread, and a stalled redis call would stall
the analyser that produced it. So the owning process keeps one authoritative in-memory
``JobRecord``, mutates it synchronously under a lock, and *publishes* immutable snapshots to the
store. Readers never see a half-written record because they never see the record — they see the last
published serialisation of it.

That makes the store a broadcast channel rather than a database, and it is why there is no
compare-and-swap here. There is exactly one writer per job for the job's whole life, so
last-writer-wins is not a race to be solved; the ordering problem that *does* exist (a technique's
"done" competing with the watchdog's "timed out") is resolved in ``JobRecord.set_stage``, close to
the two callers, before anything is serialised.

``get`` may return a snapshot up to one publish interval stale. The UI polls, so the worst case is
one extra poll; nothing forensic depends on read-after-write freshness.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.job import JobRecord

__all__ = ["STORE_KEY_PREFIX", "JobStore"]

#: Redis key namespace. Two deployments sharing one database must not collide.
STORE_KEY_PREFIX = "truelens:job:"


@runtime_checkable
class JobStore(Protocol):
    """Minimal store surface: save a snapshot, read it back, let things expire."""

    async def save(self, record: JobRecord) -> None:
        """Insert or overwrite the snapshot for ``record.id``.

        Must not raise for a job that simply timed out in the backend; a failed publish is logged
        by the caller and the analysis continues, because losing progress visibility is not a
        reason to lose the result.
        """

    async def get(self, job_id: str) -> JobRecord | None:
        """Return the most recent snapshot, or ``None`` if it never existed or has expired."""

    async def delete(self, job_id: str) -> None:
        """Remove a job. Idempotent."""

    async def sweep(self, ttl_s: float) -> int:
        """Drop records older than ``ttl_s`` (by creation time) and return how many went.

        A no-op for backends with native expiry, present on the interface so the sweeper task can
        call it unconditionally.
        """

    async def stats(self) -> dict[str, Any]:
        """Counts for ``/api/health``: how many jobs are held, how many finished."""

    async def close(self) -> None:
        """Release connections / background tasks. Must be safe to call twice."""
