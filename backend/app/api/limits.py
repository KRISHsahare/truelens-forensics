"""The shared rate limiter.

One ``Limiter`` instance is imported by both the routes that decorate themselves and the app that
installs it in ``app.state.limiter`` — slowapi's decorator looks the limiter up on the request's app
at call time, so a second instance would count a second, independent set of windows and the limit
would silently double.

The limit *values* are callables, not strings. A string is evaluated when the module is imported,
which would freeze whatever ``RATE_LIMIT_ANALYZE`` happened to be when the package was first
touched; a callable is read per request, so an env change in a test actually moves the limit.

**Every decorated route takes a ``response: Response`` argument.** With ``headers_enabled`` on — and
it is on, because the client shows a quota meter from ``X-RateLimit-Remaining`` — slowapi's wrapper
attaches the window headers to whatever ``Response`` the endpoint returns. A route that returns a
Pydantic model returns no ``Response`` at that point, so slowapi looks for one among the handler's
arguments and raises if there is none. Declaring the parameter is the documented integration:
FastAPI treats ``Response`` as a non-field dependency, injects a throwaway instance, and merges its
headers onto the real response after serialisation.
"""

from __future__ import annotations

from collections.abc import Callable

from slowapi import Limiter

from app.config import get_settings
from app.core.ratelimit import build_limiter

__all__ = ["limit_analyze", "limit_batch", "limit_jobs", "limiter"]

limiter: Limiter = build_limiter()


def _field(name: str) -> Callable[[], str]:
    def read() -> str:
        return str(getattr(get_settings(), name))

    read.__name__ = f"limit_{name}"
    return read


#: Uploads are the expensive route: one request buys a full CPU-bound analysis.
limit_analyze = _field("rate_limit_analyze")
#: Polling is cheap but frequent, and a client that spins at 2 Hz should still not be refused.
limit_jobs = _field("rate_limit_jobs")
#: One batch request is up to ``batch_max_images`` analyses, so its window is proportionally tighter.
limit_batch = _field("rate_limit_batch")
