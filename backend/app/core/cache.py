"""Result cache keyed by SHA-256 of the uploaded bytes.

The prototype re-analysed the same file on every page refresh, which is both slow and pointless:
every analyser here is a deterministic function of the bytes. Keying on the digest is safe for
exactly that reason — no analyser reads the clock, the filename, or the job id. What is re-minted on
a hit is the per-run surface: a new job id, new timestamps, and freshly signed overlay URLs. The
``timings`` inside the payload are the *original* run's, which is what the record's ``from_cache``
flag exists to say — a replayed analysis is honest about having taken 3.4 seconds once, not about
having taken 3.4 seconds now.

**The overlay problem.** A cached result references PNG files under the *original* job's scratch
directory, which the sweeper deletes after ``overlay_ttl_s``. ``Settings.validate_prod`` already
rejects ``result_cache_ttl_s >= overlay_ttl_s`` so an entry cannot outlive its files, and this module
finishes the job: each entry records the wall-clock deadline after which its overlays are gone, and
``resign`` either mints a URL whose signature expires *no later than* that deadline, or, once the
deadline has passed, returns the scores with overlays stripped and a note saying so. A stale cache
entry therefore degrades into "here is the verdict, the pictures expired" rather than a 410.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from app.core.signing import OverlaySigner

__all__ = ["CacheHit", "ResultCache"]

log = logging.getLogger("truelens.cache")


@dataclass(slots=True)
class CacheHit:
    """A cached result, already re-signed for the current clock."""

    result: dict[str, Any]
    #: Techniques whose overlay is still fetchable. Empty once the deadline has passed.
    overlays_available: list[str] = field(default_factory=list)
    expired_overlays: list[str] = field(default_factory=list)

    @property
    def overlays_expired(self) -> bool:
        return bool(self.expired_overlays)


class ResultCache:
    """TTL + LRU + bounded-size cache of finished analyses."""

    def __init__(self, *, ttl_s: float, max_entries: int = 256) -> None:
        self.ttl_s = float(ttl_s)
        self.max_entries = max(1, int(max_entries))
        #: ``sha256 -> (stored_at, payload)``. An ``OrderedDict`` gives O(1) LRU recency; TTL and
        #: LRU are two different reasons to evict and both are enforced on every touch.
        self._entries: OrderedDict[str, tuple[float, _Payload]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    # --------------------------------------------------------------------- api
    def put(
        self,
        sha256: str,
        result: dict[str, Any],
        overlay_refs: dict[str, str],
        *,
        overlay_deadline: float,
    ) -> None:
        """Store ``result``.

        ``overlay_refs`` maps technique name -> ``"{job_id}/{name}"``; the result dict itself must
        hold ``overlay_url = None`` for those techniques, since a URL is a signed *time* as much as
        a signed *path* and cannot be frozen. ``overlay_deadline`` is the unix time at which the
        scratch sweeper will have removed those files.
        """
        if not sha256:
            return
        payload = _Payload(
            # Copy so a later mutation of the caller's dict (e.g. attaching per-job timings) cannot
            # rewrite the cached answer.
            result=dict(result),
            overlay_refs=dict(overlay_refs),
            deadline=float(overlay_deadline),
        )
        self._entries.pop(sha256, None)
        self._entries[sha256] = (time.time(), payload)
        self._evict()

    def get(self, sha256: str, signer: OverlaySigner, *, now: float | None = None) -> CacheHit | None:
        """Return a re-signed hit, or ``None``.

        ``signer`` is a parameter rather than something the cache owns: the cache holds no secrets
        and knows nothing about URL shape, it only knows the deadline after which its own entries
        must stop advertising images. That keeps this testable without a key and keeps key material
        in one place.

        A single expired entry is removed on read instead of being swept lazily: an entry that has
        passed its TTL must never be servable, and waiting for the sweeper leaves a window where it
        is still in the dict and still correct-by-checking-code to return.
        """
        now = now if now is not None else time.time()
        found = self._entries.get(sha256)
        if found is None:
            self.misses += 1
            return None
        stored_at, payload = found
        if now - stored_at >= self.ttl_s:
            self._entries.pop(sha256, None)
            self.misses += 1
            return None
        # Touch for LRU recency only after the TTL check, so expired entries are not revived.
        self._entries.move_to_end(sha256)
        self.hits += 1
        return self._resign(payload, now, signer)

    def drop_overlays_for(self, job_id: str) -> int:
        """Invalidate every entry whose overlays lived in ``job_id``'s directory.

        Called when the scratch tree is pruned ahead of schedule — a crash cleanup, or an explicit
        deletion. The TTL invariant usually makes this unnecessary, but "usually" is not a reason to
        leave a path where a cache entry can name a deleted file.
        """
        doomed = [
            sha for sha, (_, p) in self._entries.items()
            if any(ref.startswith(f"{job_id}/") for ref in p.overlay_refs.values())
        ]
        for sha in doomed:
            self._entries.pop(sha, None)
        return len(doomed)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "capacity": self.max_entries,
            "ttl_s": self.ttl_s,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else None,
        }

    # --------------------------------------------------------------- internals
    def _resign(self, payload: _Payload, now: float, signer: OverlaySigner) -> CacheHit:
        """Rebuild the result with overlay URLs valid up to the files' deadline."""
        result = deepcopy(payload.result)
        techniques = result.get("techniques") or {}
        available: list[str] = []
        expired: list[str] = []
        for name, ref in payload.overlay_refs.items():
            view = techniques.get(name)
            if view is None or "overlay_url" not in view:
                continue
            if now >= payload.deadline:
                view["overlay_url"] = None
                expired.append(name)
                continue
            job_id, _, overlay_name = ref.partition("/")
            ttl = max(1, min(signer.ttl_s, int(payload.deadline - now)))
            view["overlay_url"] = signer.url(job_id, overlay_name, now=now, ttl_override=ttl)
            available.append(name)
        if expired:
            notes = result.setdefault("notes", [])
            note = (
                "Overlay images from this cached analysis have expired; the scores and verdict "
                "stand, but re-run the image to regenerate the pictures."
            )
            if note not in notes:
                notes.append(note)
        return CacheHit(result=result, overlays_available=available, expired_overlays=expired)

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries:
            evicted, _ = self._entries.popitem(last=False)
            log.debug("cache evicted %s", evicted[:16])


@dataclass(frozen=True, slots=True)
class _Payload:
    """What one cache slot holds."""

    result: dict[str, Any]
    #: technique name -> ``"{job_id}/{overlay_name}"``, relative to the scratch root.
    overlay_refs: dict[str, str]
    #: unix time after which the referenced PNGs are gone.
    deadline: float
