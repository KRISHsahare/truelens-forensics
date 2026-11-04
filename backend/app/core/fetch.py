"""Server-side image fetching for URL import, written so it cannot be pointed at the network it runs on.

``POST /api/analyze`` with a ``url`` makes the API perform an HTTP request that the client chose.
That is the definition of an SSRF primitive, and the boring version of this module — "check the
scheme, then ``httpx.get(url)``" — is vulnerable even though it looks validated:

* **DNS is not a one-time event.** Resolving ``evil.example`` to 93.x.x.x for the check and then
  letting the HTTP stack resolve it again for the connection is a time-of-check race, and a
  hostname with a round-robin record containing 169.254.169.254 hits the cloud metadata endpoint on
  some attempts. So the address is resolved **once**, vetted, and the socket connects to that exact
  address. The TLS handshake still uses the original hostname for SNI and certificate verification
  — pinning the IP must not silently downgrade to "no hostname checking".
* **Redirects are second hops, not free.** Every redirect restarts the vetting from the scheme, so
  a public URL that 302s to ``127.0.0.1`` is refused rather than followed.
* **The blacklist is IP-space, not hostname-space.** ``localhost``, ``127.0.0.1``, ``10.0.0.0/8``,
  ``169.254.169.254``, ``::1``, ``fc00::/7`` and the IPv4-mapped IPv6 spellings of the same
  addresses are all refused, because "is this a private host" is a question about the answer, not
  about the spelling.

Size and time are bounded because the caller controls both: a 4 GB "image" or a server that trickles
one byte per second must not become a job that never finishes.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import ssl
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from typing import Any
from urllib.parse import urlsplit

__all__ = ["FetchError", "FetchedImage", "fetch_image"]

log = logging.getLogger("truelens.fetch")

#: Bytes actually read, not the advertised Content-Length — a lying header must not be a memory
#: allocation. This is the read ceiling; the API's own cap is applied after.
READ_CEILING_SLACK = 1024 * 1024
MAX_REDIRECTS = 3
_CONNECT_TIMEOUT_S = 5.0
#: Idle time between chunks. A trickle is a denial of service unless the *total* is also bounded.
_READ_TIMEOUT_S = 10.0
_TOTAL_DEADLINE_S = 20.0

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_ALLOWED_CONTENT_TYPES = ("image/", "application/octet-stream")


class FetchError(Exception):
    """Anything that makes us refuse to fetch, or that made the fetch useless.

    ``message`` is what the client sees. It never contains the resolved address, the response
    headers, or a detail from the upstream server — a refusal that explains which internal IP was
    hit is a port scanner with better grammar.
    """

    def __init__(self, message: str, *, code: str = "url_fetch_failed") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True, slots=True)
class FetchedImage:
    data: bytes
    content_type: str
    #: The hostname we actually talked to, after following redirects. For the log line only.
    final_host: str


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Refuse anything that could reach the machine, its neighbours, or its cloud.

    ``is_global`` alone is not enough: it is False for the documentation ranges and the 6to4
    addresses too, which is a stricter and therefore safer answer. ``is_loopback``,
    ``is_link_local``, ``is_private`` and ``is_reserved`` are checked explicitly because the intent
    is readable at the call site, and a future caller must be able to see what is being refused
    without knowing ``is_global``'s exact semantics.
    """
    mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
    if mapped is not None:
        # ::ffff:169.254.169.254 is the metadata endpoint wearing an IPv6 spelling.
        ip = mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


def _resolve_and_vet(host: str, port: int) -> tuple[str, int]:
    """Resolve ``host`` and return the single address we will connect to, or raise.

    Every A/AAAA record is checked, not just the first: a hostname that returns a public address
    *and* a loopback address would otherwise pass the check and then connect to whichever record the
    resolver happened to hand the HTTP stack.
    """
    if not host:
        raise FetchError("The URL has no hostname.")
    try:
        # A literal IP is not exempt — getaddrinfo handles both spellings, so the vetting below
        # runs on it exactly as it does on a name.
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError("That hostname does not resolve.") from exc
    if not infos:
        raise FetchError("That hostname does not resolve.")
    for info in infos:
        sockaddr = info[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0].split("%")[0])
        except ValueError as exc:
            raise FetchError("That hostname resolved to an unusable address.") from exc
        if not _is_public(ip):
            log.info("refused %s: resolved to a non-public address", host)
            raise FetchError(
                "That URL points at a private or reserved address, which this service will not "
                "fetch. Only publicly reachable image URLs are supported.")
    # Prefer IPv4 when both exist: one address family means one vetted value to pin.
    v4 = next((i for i in infos if i[0] == socket.AF_INET), None)
    chosen = v4 or infos[0]
    return (chosen[4][0], int(chosen[4][1]))


class _PinnedHTTP(HTTPConnection):
    """Connects to a pre-vetted IP while keeping the real Host header."""

    def __init__(self, host: str, port: int, *, ip: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


class _PinnedHTTPS(HTTPSConnection):
    """TLS to a pre-vetted IP, with SNI and certificate checks still against the hostname.

    ``server_hostname=self.host`` is the line that matters. Passing the pinned IP there would turn
    this into a certificate check against an address, which any server on that IP can pass — so the
    identity we verify is the name the user asked for, exactly as a browser does.
    """

    def __init__(self, host: str, port: int, *, ip: str, timeout: float,
                 context: ssl.SSLContext) -> None:
        super().__init__(host, port, timeout=timeout, context=context)
        self._ip = ip

    def connect(self) -> None:
        raw = socket.create_connection((self._ip, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


def _request_once(
    url: str, *, max_bytes: int
) -> tuple[int, str, str, dict[str, str], bytes]:
    """One GET, no automatic redirects. Returns (status, content_type, location, headers, body)."""
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise FetchError("Only http and https URLs can be imported.")
    if parts.username or parts.password:
        # Credentials in a URL are honoured by some stacks and logged by most. Nothing here needs
        # them, and accepting them is a way to make a proxy log a secret.
        raise FetchError("Credentials in the URL are not supported.")
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)
    ip, resolved_port = _resolve_and_vet(host, port)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    # ``resolved_port`` is the port the socket is opened on, which for a service bound to a
    # non-standard port is the same value we vetted. It is used, not ignored: the address family
    # chosen by getaddrinfo can rewrite it.
    del resolved_port

    headers = {
        "Host": host if port in (80, 443) else f"{host}:{port}",
        "User-Agent": "TrueLens/1.0 (+image analysis; fetches only the URL you provide)",
        "Accept": "image/jpeg, image/png, image/webp, image/bmp;q=0.9, */*;q=0.1",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    if parts.scheme == "https":
        conn: Any = _PinnedHTTPS(host, port, ip=ip, timeout=_CONNECT_TIMEOUT_S,
                                 context=ssl.create_default_context())
    else:
        conn = _PinnedHTTP(host, port, ip=ip, timeout=_CONNECT_TIMEOUT_S)
    try:
        conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        for key, value in headers.items():
            conn.putheader(key, value)
        conn.endheaders()
        resp = conn.getresponse()
        status = resp.status
        content_type = (resp.getheader("Content-Type") or "").split(";")[0].strip().lower()
        location = resp.getheader("Location") or ""
        raw_headers = {k.lower(): v for k, v in resp.getheaders()}
        body = _read_capped(resp, max_bytes)
    except FetchError:
        raise
    except (ssl.SSLError, TimeoutError, ConnectionError, OSError) as exc:
        # One sentence, no exception text: an SSL error string names the certificate and the host.
        log.info("fetch of %s failed: %s", host, type(exc).__name__)
        raise FetchError("The image could not be downloaded from that URL.") from exc
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return status, content_type, location, raw_headers, body


def _read_capped(resp: Any, max_bytes: int) -> bytes:
    """Read the body, stopping at the cap rather than trusting Content-Length.

    ``resp.read(max_bytes + 1)`` is not enough on its own: a chunked response has no declared length
    at all, and a lying ``Content-Length`` would otherwise be an allocation request.
    """
    declared = resp.getheader("Content-Length")
    if declared:
        try:
            if int(declared) > max_bytes:
                raise FetchError("That image is larger than this service accepts.")
        except ValueError:
            pass  # A garbage header is ignored, not fatal; the read cap below still applies.
    chunks: list[bytes] = []
    total = 0
    while True:
        piece = resp.read(64 * 1024)
        if not piece:
            break
        chunks.append(piece)
        total += len(piece)
        if total > max_bytes:
            raise FetchError("That image is larger than this service accepts.")
    return b"".join(chunks)


def fetch_image(url: str, *, max_bytes: int) -> FetchedImage:
    """Fetch an image over http/https, refusing anything that is not a public image.

    Raises ``FetchError`` with a client-safe message. The caller still runs ``probe_header`` on the
    bytes: this module decides *whether to fetch*, the security layer decides *whether to analyse*.
    """
    import time

    deadline = time.monotonic() + _TOTAL_DEADLINE_S
    current = url.strip()
    seen_hops = 0
    while True:
        if time.monotonic() > deadline:
            raise FetchError("Downloading that image took too long.")
        status, content_type, location, _headers, body = _request_once(
            current, max_bytes=max_bytes + READ_CEILING_SLACK)
        if status in (301, 302, 303, 307, 308):
            seen_hops += 1
            if seen_hops > MAX_REDIRECTS:
                raise FetchError("That URL redirects too many times.")
            if not location:
                raise FetchError("The image could not be downloaded from that URL.")
            from urllib.parse import urljoin

            current = urljoin(current, location)
            continue
        if status >= 400:
            # The upstream status is reported as "could not be downloaded", not as "403 Forbidden
            # from cdn.example.com" — the difference between a hotlink block and a missing file is
            # the server's business, and echoing it turns this endpoint into a status oracle.
            raise FetchError("That URL did not return an image.")
        if content_type and not content_type.startswith(_ALLOWED_CONTENT_TYPES):
            raise FetchError(
                f"That URL is a {content_type}, not an image. Only JPEG, PNG, WebP and BMP files "
                "can be analysed.")
        if len(body) > max_bytes:
            raise FetchError("That image is larger than this service accepts.")
        if not body:
            raise FetchError("That URL returned an empty file.")
        parts = urlsplit(current)
        return FetchedImage(data=body, content_type=content_type or "image/jpeg",
                           final_host=parts.hostname or "")
