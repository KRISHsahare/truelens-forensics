"""Redis-backed job store: the multi-worker backend.

Needed for one reason — with ``WEB_CONCURRENCY>1`` a job minted in worker A must be readable by a
poll that lands in worker B. Everything else here is about not being surprised by redis.

**TTL is set on write, so expiry is enforced by redis, not by our sweeper.** ``sweep`` returns 0.
The consequence is precise: the key lives ``ttl`` seconds past the *last* publish, not past
creation. That is a ceiling on staleness rather than a promise about lifetime, which is the right
shape for a record that is rewritten continuously while a job runs — a creation-anchored TTL would
expire a job that is still legitimately working.

**Reads propagate errors; writes do not kill the analysis.** A ``get`` that fails because redis is
down must surface as 503, not as 404: telling the client "your job does not exist" when the truth is
"I cannot reach my database" invites the user to re-upload and doubles the load exactly when the
system is already struggling. A ``save`` that fails is the opposite case — the analysis is running
in this process and will produce a real result, so the publish is logged and dropped, degrading
progress visibility instead of throwing away two seconds of CPU.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.job import JobRecord
from app.core.store.base import STORE_KEY_PREFIX

__all__ = ["RedisJobStore"]

log = logging.getLogger("truelens.store.redis")

#: Upper bound on the SCAN used by ``stats``. Counting keys is an operator nicety; it must never
#: become the request that blocks the instance for every other client.
_SCAN_ROUNDS = 200
_SCAN_COUNT = 100


class RedisJobStore:
    """Job snapshots in redis, with native key expiry."""

    def __init__(self, url: str, *, ttl_s: float = 600.0, prefix: str = STORE_KEY_PREFIX) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - depends on the install extra
            raise RuntimeError(
                "STORE_BACKEND=redis needs the 'redis' extra: pip install 'redis>=5.2'"
            ) from exc
        if not url:
            raise ValueError("REDIS_URL must be set when STORE_BACKEND=redis")
        self._redis = Redis.from_url(url, decode_responses=True)
        self._ttl = int(max(1, ttl_s))
        self._prefix = prefix
        self._writes_failed = 0

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    # ------------------------------------------------------------------- writes
    async def save(self, record: JobRecord) -> None:
        payload = json.dumps(record.to_dict(), separators=(",", ":"))
        try:
            await self._redis.set(self._key(record.id), payload, ex=self._ttl)
        except Exception as exc:
            # Logged, never raised: see the module docstring.
            self._writes_failed += 1
            log.warning("job publish failed (%s): %s", record.id, exc)

    async def delete(self, job_id: str) -> None:
        try:
            await self._redis.delete(self._key(job_id))
        except Exception as exc:
            log.warning("job delete failed (%s): %s", job_id, exc)

    # -------------------------------------------------------------------- reads
    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self._redis.get(self._key(job_id))
        if raw is None:
            return None
        try:
            return JobRecord.from_dict(json.loads(raw))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            log.exception("unreadable job record %s; dropping", job_id)
            await self.delete(job_id)
            return None

    async def sweep(self, ttl_s: float) -> int:
        """No-op: key expiry is set on write and enforced by redis."""
        return 0

    async def stats(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "backend": "redis",
            "ttl_s": self._ttl,
            "shared_across_workers": True,
            "failed_publishes": self._writes_failed,
        }
        try:
            held, exact = await self._count()
            data["jobs_held"] = held
            data["jobs_held_exact"] = exact
        except Exception as exc:
            data["jobs_held"] = None
            data["count_error"] = str(exc)[:200]
        return data

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except AttributeError:  # pragma: no cover - older redis-py
            await self._redis.close()
        except Exception as exc:
            log.warning("redis close failed: %s", exc)

    # ------------------------------------------------------------------ internal
    async def _count(self) -> tuple[int, bool]:
        """Bounded ``SCAN`` over our prefix. Returns ``(count, exact)``.

        Deliberately capped. Counting keys with ``KEYS`` is the classic way to freeze a redis
        instance, and even ``SCAN`` is not free; an approximate number on a health endpoint is worth
        less than a responsive database, so when the cap is hit the answer says so.
        """
        cursor = 0
        total = 0
        for _ in range(_SCAN_ROUNDS):
            cursor, keys = await self._redis.scan(
                cursor=cursor, match=f"{self._prefix}*", count=_SCAN_COUNT
            )
            total += len(keys)
            if cursor == 0:
                return total, True
        return total, False
