"""In-process job store.

Default backend, and the only one that needs an eviction policy: a container that runs for a week
with a TTL-only design holds every job created in the last TTL plus every job created *since* the
last sweep, so the sweeper interval has to be a real requirement rather than a formality. Two
independent bounds therefore exist — time (``job_ttl_s``) and count (``max_entries``) — because a
count bound alone still holds a kilobyte-scale result per job for a hundred jobs, and a time bound
alone survives a burst.

**All methods are event-loop only.** No lock: every caller is a coroutine on the loop, so the
GIL-free interleaving point is the ``await`` and dict operations between awaits are atomic for our
purposes. Pool-worker progress therefore travels through ``call_soon_threadsafe`` in the
orchestrator rather than reaching into the store from a non-async thread.

**This is per-process state.** With ``WEB_CONCURRENCY>1`` a job created in worker A is invisible to
worker B, so its polls 404. ``Settings.validate_prod`` makes that combination fatal instead of
letting it surface as intermittent missing jobs in production; on a single worker this backend is
strictly faster than redis and has no failure mode of its own.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.job import JobRecord

__all__ = ["MemoryJobStore"]

log = logging.getLogger("truelens.store.memory")


class MemoryJobStore:
    """Bounded, TTL-expiring dict of job snapshots."""

    def __init__(self, *, max_entries: int = 512) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        #: ``(created_at, first_terminal_at)``. A second index so eviction and sweeping never
        #: deserialise a value; mutated in the same statements as ``_records``.
        self._ages: dict[str, tuple[float, float | None]] = {}
        self.max_entries = max(1, int(max_entries))
        self._saved = 0
        self._evicted = 0

    # ------------------------------------------------------------------- writes
    async def save(self, record: JobRecord) -> None:
        payload = record.to_dict()
        previous = self._ages.get(record.id)
        existed = previous is not None
        # The first transition to terminal claims the timestamp, so a record republished after
        # finishing still sorts as "finished a while ago" for eviction.
        finished = previous[1] if (previous and previous[1]) else (
            time.time() if record.is_terminal() else None
        )
        self._records[record.id] = payload
        self._ages[record.id] = (record.created_at, finished)
        self._saved += 1
        if not existed:
            self._enforce_capacity()

    async def delete(self, job_id: str) -> None:
        self._records.pop(job_id, None)
        self._ages.pop(job_id, None)

    # -------------------------------------------------------------------- reads
    async def get(self, job_id: str) -> JobRecord | None:
        payload = self._records.get(job_id)
        if payload is None:
            return None
        try:
            return JobRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            # A record that cannot be read back is worse than a missing one: it would make every
            # poll a 500. Drop it and report 404.
            log.exception("unreadable job record %s; dropping", job_id)
            await self.delete(job_id)
            return None

    async def sweep(self, ttl_s: float) -> int:
        now = time.time()
        stale = [jid for jid, (created, _) in self._ages.items() if now - created > ttl_s]
        for jid in stale:
            self._records.pop(jid, None)
            self._ages.pop(jid, None)
        if stale:
            log.info("swept %d expired job(s)", len(stale))
        return len(stale)

    async def stats(self) -> dict[str, Any]:
        total = len(self._records)
        terminal = sum(1 for (_, f) in self._ages.values() if f is not None)
        return {
            "backend": "memory",
            "jobs_held": total,
            "jobs_terminal": terminal,
            "jobs_running": total - terminal,
            "capacity": self.max_entries,
            "saves": self._saved,
            "evictions": self._evicted,
            "shared_across_workers": False,
        }

    async def close(self) -> None:
        self._records.clear()
        self._ages.clear()

    # ------------------------------------------------------------------ internal
    def _enforce_capacity(self) -> None:
        """Evict until back inside ``max_entries``."""
        overflow = len(self._records) - self.max_entries
        if overflow <= 0:
            return
        # Finished first, longest-finished first. Only when a burst of *concurrent* jobs alone
        # exceeds the cap do we drop running records: losing an in-flight analysis is the right
        # sacrifice against unbounded growth, but it should never be the first resort.
        finished = sorted((f, c, jid) for jid, (c, f) in self._ages.items() if f is not None)
        victims = [jid for (_, _, jid) in finished[:overflow]]
        if len(victims) < overflow:
            running = sorted((c, jid) for jid, (c, f) in self._ages.items() if f is None)
            victims += [jid for (_, jid) in running[: overflow - len(victims)]]
        for jid in victims:
            self._records.pop(jid, None)
            self._ages.pop(jid, None)
            self._evicted += 1
        log.warning("evicted %d job record(s) at capacity %d", len(victims), self.max_entries)
