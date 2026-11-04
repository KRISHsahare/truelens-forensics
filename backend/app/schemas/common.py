"""Error and health schemas.

``Error`` exists so every failure the API returns has the same shape, including the ones FastAPI
produces before a route body runs (413 upload too large, 422 validation, 429 rate limit). A client
that has to parse three error grammars will get one of them wrong, and the one it gets wrong is the
one it shows the user.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Error", "Health", "NotFound"]

ErrorCode = Literal[
    "invalid_image",
    "too_large",
    "unsupported_media",
    "payload_too_large",
    "decode_failed",
    "unredactable",
    "too_many_images",
    "rate_limited",
    "server_busy",
    "not_found",
    "gone",
    "job_failed",
    #: A document asked for before the analysis finished. Different from ``job_failed``, which is
    #: terminal: this one is a "come back on the next poll".
    "job_not_ready",
    "internal",
]


class Error(BaseModel):
    """The single error envelope.

    ``detail`` is written for a person; ``code`` is written for a switch statement. Both are
    mandatory because a UI that strings-matches on ``detail`` breaks the first time the copy changes.
    """

    model_config = ConfigDict(frozen=True)

    error: ErrorCode
    message: str = Field(description="Human-readable, safe to display verbatim.")
    detail: str | None = Field(None, description="More specific context, e.g. the size limit.")
    retry_after_seconds: int | None = None
    #: Machine-checkable extra data (limit value, allowed formats). Never carries paths or tracebacks.
    extra: dict[str, Any] | None = None


class NotFound(BaseModel):
    """A 404 with no explanation of *why* the thing is missing.

    Job ids and overlay names are unguessable-by-construction, so the distinction between "never
    existed", "expired" and "signature invalid" tells an attacker nothing they do not already have to
    guess — and telling them which one it is would leak whether a job id is live.
    """

    model_config = ConfigDict(frozen=True)

    error: Literal["not_found"] = "not_found"
    message: str = (
        "Not found. Jobs and their overlays are deleted within ten minutes of upload, so this may "
        "simply have expired."
    )


class Health(BaseModel):
    """Liveness plus the facts an operator actually needs when a deploy misbehaves.

    Deliberately not secret-free-by-omission: nothing here is a credential, but the store backend,
    worker counts and artifact presence are the four things you check first when results look wrong.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    #: ``degraded`` means something is present but not working — a store that will not answer, a
    #: missing signing key in prod. Requests still succeed; the answers are just less than asked for.
    problems: list[str] = Field(default_factory=list)
    perf_profile: str
    store: dict[str, Any] = Field(default_factory=dict)
    cache: dict[str, Any] = Field(default_factory=dict)
    concurrency: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, bool] = Field(default_factory=dict)
    #: The resolved CORS allowlist and proxy trust. This is here because a wrong `FRONTEND_ORIGIN`
    #: produces the most misleading failure this service can have: every request succeeds with 200,
    #: and the browser simply refuses to hand the body to the page -- so the frontend reports an
    #: unreachable backend while the backend's own health check is green. Reading the list is the
    #: one-glance diagnosis; probing with curl is not something an operator should have to do.
    network: dict[str, Any] = Field(default_factory=dict)
    uptime_s: float = 0.0
