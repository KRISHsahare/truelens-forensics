"""Per-client rate limiting and the IP identity it keys on.

**The key function is the security-relevant part.** Limiting by ``X-Forwarded-For`` is what makes
the limit work at all behind Render or nginx — every request then arrives from one proxy IP and
socket-based limits would collapse into a single global bucket. It is also what makes the limit
trivially evadable when the header is not rewritten upstream: a client can send any chain it likes,
so each new value buys a fresh allowance. ``TRUST_PROXY`` therefore defaults to *false*, and when it
is enabled the leftmost entry is used only because that is the address the *client* claims, which a
correctly configured proxy prepends to and never overwrites.

**Limits are per-process.** The storage is in-memory, so ``WEB_CONCURRENCY=n`` multiplies the
effective allowance by ``n``. That is stated rather than engineered around because sharing a
counter across workers needs a round trip to redis on every request, and this limit exists to bound
CPU on a forensic pipeline, not to be a billing meter. A deployment that needs exactness should
enforce it at the edge.

**Memory the store uses** is bounded by *distinct addresses seen inside a live window*, not by total
traffic: a fixed-window store discards each bucket when its window closes. Worth being exact about
what that does *not* buy — a client that spoofs a thousand addresses gets a thousand buckets, so the
per-IP limit is not the defence against a distributed or spoofed flood. The real bound on work
accepted is ``max_concurrent_jobs`` plus the job deadlines, and the per-IP limit exists to stop one
ordinary user saturating the box by accident.
"""

from __future__ import annotations

import logging
from typing import Any

from slowapi import Limiter
from slowapi.extension import HEADERS
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import get_settings
from app.schemas.common import Error

__all__ = ["build_limiter", "client_ip", "rate_limit_exceeded_handler"]

log = logging.getLogger("truelens.ratelimit")


def client_ip(request: Request, *, trust_proxy: bool | None = None) -> str:
    """The address a rate limit should be attributed to.

    Returns ``"unknown"`` rather than ``""`` when nothing is available: an empty key would collapse
    every unidentified client into one shared bucket, which fails closed against the attacker but
    fails *open* for the honest ones — one scanner floods the bucket and the whole deployment starts
    429ing. Distinct and unlimitable is the safer bug.
    """
    settings = get_settings()
    if trust_proxy is None:
        trust_proxy = settings.trust_proxy
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        # The chain is "client, proxy1, proxy2": the first entry is what the client asserted, and
        # every later entry was written by a hop. Only the first is the identity we want to limit.
        candidate = forwarded.split(",")[0].strip()
        if candidate:
            return _normalise(candidate)
        # Header absent (direct connection to the app, or a proxy that strips it). Falling through
        # to the socket is correct; falling through to a constant would be worse.
    direct = request.client.host if request.client else ""
    return _normalise(direct) or "unknown"


def _normalise(value: str) -> str:
    """Bound the key's length and strip the IPv6 zone id.

    Without this, a hostile ``X-Forwarded-For`` of arbitrary length becomes an arbitrary-length
    dictionary key, and ``fe80::1%eth0`` would hash differently from the same address on another
    interface for no reason that serves the limit.
    """
    value = value.strip()
    if "%" in value:
        value = value.split("%", 1)[0]
    return value[:64]


def build_limiter() -> Limiter:
    """A limiter wired to the configured IP key and the in-memory window store.

    Routes decorate themselves with ``limiter.limit(lambda: get_settings().rate_limit_analyze)`` — a
    *callable* limit, so the value is read per request rather than captured at import. The
    difference matters for tests, where ``reset_settings_cache`` plus an env change has to actually
    move the limit, and it means one shared limiter serves every endpoint.

    ``swallow_errors`` is on deliberately. The store is in-process and should never fail; if it
    does, a counter that has broken should not turn the whole API into 500s. Failing open on a
    broken window is the cheaper bug, because the thing being protected — a CPU-bound pipeline — has
    the job concurrency cap and the deadlines as its second line of defence.

    **The renamed header is not cosmetic.** With ``headers_enabled``, slowapi writes ``Retry-After``
    onto every response from a limited route, computed as the remainder of *its* window, and it does
    so by overwriting whatever the route already set. On a 503 that is actively wrong: the route is
    saying "a slot frees in about three seconds" and the limiter replaces it with "wait sixty". The
    limiter's own countdown is genuinely useful information, so it is kept — under a namespaced
    header — while ``Retry-After`` stays owned by the layer that knows why the request was refused.
    """
    rate_limiter = Limiter(
        # A callable, so the proxy decision follows the live setting instead of the boot value.
        key_func=lambda request: client_ip(request),
        default_limits=[],
        # "fixed" is cheaper and, for a limit whose purpose is "no more than N analyses per minute
        # per address", answers the question being asked. A moving window would let a client that
        # slept for 30s burst at double rate across the boundary.
        strategy="fixed-window",
        storage_uri="memory://",
        headers_enabled=True,
        key_prefix="truelens",
        swallow_errors=True,
        enabled=True,
    )
    # There is no constructor argument for this; the mapping is built once inside ``__init__`` from
    # slowapi's defaults, and this is the only place the collision is worth fixing.
    rate_limiter._header_mapping[HEADERS.RETRY_AFTER] = "X-Ratelimit-Window-Retry-After"
    return rate_limiter


async def rate_limit_exceeded_handler(request: Request, exc: Exception) -> Any:
    """429 with a machine-readable body.

    The default slowapi handler renders the internal exception's text, which includes the route
    rule and the window — harmless here, but the client also needs ``retry_after`` as data rather
    than as a header alone, because the Analyzer page greys out the upload button on a real number
    instead of guessing.
    """
    limit = str(getattr(exc, "detail", "") or "")
    seconds = _retry_seconds(limit)
    log.info("rate limited %s %s", client_ip(request), request.url.path)
    body = Error(
        error="rate_limited",
        message=("Too many analyses from this address. The limit is per minute and the queue "
                 "clears automatically."),
        retry_after_seconds=seconds,
        extra={"limit": limit or None})
    return JSONResponse(
        status_code=429,
        content=body.model_dump(mode="json"),
        headers={"Retry-After": str(seconds)},
    )


def _retry_seconds(limit: str) -> int:
    """Best-effort "seconds until the window rolls" from a limit string like ``20/minute``.

    Exact only for whole-minute windows, which is what the configured limits are. Being off by
    seconds on a ``Retry-After`` is not a correctness question; returning nothing would be worse for
    the client than returning an upper bound.
    """
    unit = limit.rsplit("/", 1)[-1].strip().lower() if "/" in limit else ""
    return {"second": 1, "minute": 60, "hour": 3600, "day": 86400}.get(unit, 60)
