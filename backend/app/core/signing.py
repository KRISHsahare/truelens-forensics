"""Signed, expiring overlay URLs.

An overlay path encodes a job id the client already knows and a technique name it can guess. Left
unsigned, ``GET /api/overlays/{job_id}/{name}.png`` is a catalogue: job ids are minted in
millisecond-sortable order, so an attacker who knows roughly when one upload happened can walk the
prefix and pull heatmaps for other people's uploads. The privacy statement promises no server-side
image storage, and a guessable URL to the analysed output is a hole in that promise even though the
upload itself never touches disk.

So the URL carries an expiry and an HMAC over ``(job id, technique, expiry)``, and the route serves
only when both check out. ``expires_at`` is part of the signed message rather than a separate
deadline check, so a token cannot be re-dated without re-signing it.

Two details worth the code they cost:

* **Constant-time comparison.** ``hmac.compare_digest`` instead of ``==``. The message includes a
  server-side secret so a timing oracle here does not directly yield a valid signature, but the
  check is on a secret-deriving path and there is no reason to be the codebase that thought it was
  fine.
* **Signature length is not truncated.** A 32-byte digest costs 64 hex characters in a URL that is
  already carrying a 128-bit job nonce. Cutting it to "URL-friendly" lengths reintroduces exactly
  the forgery-probability question the signature exists to close.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode

__all__ = ["OverlaySigner", "verify_signature"]

_CANONICAL_SEP = "|"


def _key_bytes(key: str | bytes) -> bytes:
    return key.encode("utf-8") if isinstance(key, str) else bytes(key)


class OverlaySigner:
    """Mints and checks overlay URLs for one deployment."""

    def __init__(self, key: str, *, ttl_s: int = 540, base_url: str = "/api/overlays") -> None:
        if not key:
            raise ValueError("overlay signing key must not be empty")
        self._key = _key_bytes(key)
        self.ttl_s = int(ttl_s)
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------- signing
    def _message(self, job_id: str, name: str, expires_at: int) -> bytes:
        # Pipe-delimited and length-free: both segments are already constrained by
        # ``scratch.path_safe`` to character classes that cannot contain "|", so the fields cannot
        # be re-aligned by an attacker who controls one of them.
        return f"{job_id}{_CANONICAL_SEP}{name}{_CANONICAL_SEP}{expires_at}".encode()

    def signature(self, job_id: str, name: str, expires_at: int) -> str:
        return hmac.new(self._key, self._message(job_id, name, expires_at), hashlib.sha256).hexdigest()

    def url(
        self,
        job_id: str,
        name: str,
        *,
        now: float | None = None,
        ttl_override: int | None = None,
    ) -> str:
        """A full relative overlay URL, valid for ``ttl_s`` from ``now``.

        Relative, not absolute: the frontend knows its own API origin and the backend behind a
        Render/Compose healthcheck does not reliably know which hostname the browser used. Signing
        only the path and query keeps the URL portable across proxies.

        ``ttl_override`` shortens the lifetime when the caller knows the underlying file will not
        outlive the default — a cache replay, where the PNG is already partway through its own TTL.
        Only shortening is permitted: extending past the signer's configured lifetime would hand out
        a URL for a file the sweeper has certainly already removed.
        """
        if ttl_override is not None and ttl_override > self.ttl_s:
            raise ValueError("ttl_override may only shorten a signature lifetime")
        ttl = self.ttl_s if ttl_override is None else max(1, int(ttl_override))
        exp = int((now or time.time()) + ttl)
        query = urlencode({"exp": exp, "sig": self.signature(job_id, name, exp)})
        return f"{self.base_url}/{job_id}/{name}.png?{query}"

    # ----------------------------------------------------------------- verifying
    def check(self, job_id: str, name: str, expires_at: str, sig: str, *, now: float | None = None) -> bool:
        return verify_signature(self._key, job_id, name, expires_at, sig, now=now)

    def check_url(self, url: str, *, now: float | None = None) -> bool:
        """Verify a URL this signer produced. Convenience for tests and the smoke script."""
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        tail = parsed.path.rsplit("/", 2)
        if len(tail) < 3:
            return False
        job_id, file_and_ext = tail[-2], tail[-1]
        qs = parse_qs(parsed.query)
        try:
            return self.check(job_id, file_and_ext.removesuffix(".png"),
                              qs["exp"][0], qs["sig"][0], now=now)
        except (KeyError, IndexError):
            return False


def verify_signature(
    key: str | bytes,
    job_id: str,
    name: str,
    expires_at: str,
    sig: str,
    *,
    now: float | None = None,
) -> bool:
    """True when ``sig`` is authentic *and* unexpired.

    Takes strings, not ints, because both arrive from a query string. The digest is recomputed over
    the *parsed* integer rather than the raw text, so ``exp=000123`` and ``exp=123`` are the same
    token: the signature commits to a numeric deadline, not to one spelling of it. Nothing is gained
    by rejecting the other spellings, and normalising before comparing removes the question.
    """
    if not key or not sig:
        return False
    try:
        exp = int(expires_at)
    except (TypeError, ValueError):
        return False
    if exp <= int(now or time.time()):
        return False
    # Only lowercase hex of the right length can be one of our signatures. Rejecting anything else
    # up front keeps compare_digest off a path where the inputs differ wildly in length, and stops a
    # truncated or doubled digest from being treated as a candidate.
    if len(sig) != 64 or any(ch not in "0123456789abcdef" for ch in sig):
        return False
    expected = hmac.new(
        _key_bytes(key),
        f"{job_id}{_CANONICAL_SEP}{name}{_CANONICAL_SEP}{exp}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, sig)
