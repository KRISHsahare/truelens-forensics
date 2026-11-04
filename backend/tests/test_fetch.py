"""URL import: the SSRF refusals, without a network.

``POST /api/analyze`` with a ``url`` makes this process perform an HTTP request the client chose, so
the tests below are the ones that decide whether the endpoint is an image fetcher or a port scanner
pointed at whatever network the container sits on. Nothing here touches a socket: DNS is answered by
a fake resolver and the connection classes are replaced, which means the *vetting and the pinning*
are what run — including the real ``_request_once`` and the real redirect loop.

The property under test is not "the bad case is caught" but "the bad case is caught on the way it is
usually dressed up as good": a name with one public record and one loopback record, a public URL that
redirects inward, an IPv6 spelling of an IPv4 denial.
"""

from __future__ import annotations

import ipaddress
import socket
import time

import pytest

from app.core import fetch
from app.core.fetch import FetchError, _is_public, _resolve_and_vet, fetch_image

PUBLIC = "93.184.216.34"
V6 = "2606:4700:4700::1111"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
IMAGE_HEADERS = {"Content-Type": "image/png"}


class _Response:
    """The sliver of ``http.client.HTTPResponse`` that ``_read_capped`` uses."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self._headers = list(headers.items())
        self._body = body
        self._pos = 0

    def getheader(self, name: str, default=None):
        for key, value in self._headers:
            if key.lower() == name.lower():
                return value
        return default

    def getheaders(self):
        return list(self._headers)

    def read(self, size: int) -> bytes:
        chunk = self._body[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


class _Server:
    """``"host" + "request-path"`` -> canned response, plus a log of every dial made."""

    def __init__(self, routes: dict[str, tuple[int, dict[str, str], bytes]]) -> None:
        self.routes = routes
        self.connections: list[dict[str, object]] = []
        self.host_headers: list[str] = []
        self.paths: list[str] = []

    def serve(self, host, port, *, ip, timeout, **_kwargs):
        return _Conn(self, host, port, ip)


class _Conn:
    def __init__(self, server: _Server, host: str, port: int, ip: str) -> None:
        self._server = server
        self._host = host
        self._port = port
        self._ip = ip
        self._path = "/"
        self._response: _Response | None = None

    def putrequest(self, method, path, **_kwargs):
        assert method == "GET"
        self._path = path

    def putheader(self, key, value):
        if key.lower() == "host":
            self._server.host_headers.append(value)

    def endheaders(self):
        # Logged here rather than in the constructor: a connection object built but never used tells
        # the same "we reached this host" lie as one that was used.
        self._server.connections.append({"host": self._host, "port": self._port, "ip": self._ip})
        self._server.paths.append(self._path)
        route = self._server.routes.get(f"{self._host}{self._path.split('?')[0]}")
        if route is None:
            raise AssertionError(f"test served an unexpected URL: {self._host}{self._path}")
        status, headers, body = route
        self._response = _Response(status, dict(headers), body)

    def getresponse(self):
        assert self._response is not None
        return self._response

    def close(self):
        pass


def _dns(table: dict[str, list[str]]):
    """A ``getaddrinfo`` that answers from a table and never touches a resolver."""
    def fake(host, port, *_args, **_kwargs):
        try:
            # A numeric host is answered with itself, as the real resolver does. That is the point of
            # several tests below: an IP spelled into the URL is not exempt from vetting.
            ipaddress.ip_address(host)
            addrs = [host]
        except ValueError:
            addrs = table.get(host)
        if not addrs:
            raise socket.gaierror(8, "Name or service not known")
        return [(socket.AF_INET if ":" not in a else socket.AF_INET6,
                 socket.SOCK_STREAM, 0, "", (a, port)) for a in addrs]
    return fake


@pytest.fixture()
def net(monkeypatch):
    """Install the fake resolver and fake connections. ``net["install"]`` sets the per-test answers."""
    state: dict[str, object] = {}

    def install(*, routes: dict[str, tuple[int, dict[str, str], bytes]] | None = None,
                dns: dict[str, list[str]] | None = None) -> _Server:
        server = _Server(routes or {})
        monkeypatch.setattr(fetch.socket, "getaddrinfo", _dns(dns or {}))
        monkeypatch.setattr(fetch, "_PinnedHTTP", server.serve)
        monkeypatch.setattr(fetch, "_PinnedHTTPS", server.serve)
        return server

    state["install"] = install
    return state


def _ok(body: bytes = PNG, **headers) -> tuple[int, dict[str, str], bytes]:
    return (200, {**IMAGE_HEADERS, **headers}, body)


def _img(url: str = f"https://{PUBLIC}/photo.png", *, max_bytes: int = 1_000_000):
    return fetch_image(url, max_bytes=max_bytes)


# --------------------------------------------------------------------- the happy path


def test_a_public_image_is_fetched(net):
    server = net["install"](routes={f"{PUBLIC}/photo.png": _ok()})
    image = _img()
    assert image.data == PNG
    assert image.content_type == "image/png"
    assert image.final_host == PUBLIC
    assert server.connections == [{"host": PUBLIC, "port": 443, "ip": PUBLIC}]


def test_the_socket_goes_to_the_vetted_address_but_the_host_header_stays(net):
    """Pinning the IP must not turn into "we asked the address who it is"."""
    host = "cdn.example"
    server = net["install"](
        dns={host: [PUBLIC]},
        routes={f"{host}/a.jpg": _ok(), f"{host}/b.jpg": _ok()})
    _img(f"http://{host}/a.jpg")
    _img(f"http://{host}:8080/b.jpg")
    assert [c["ip"] for c in server.connections] == [PUBLIC, PUBLIC]
    assert [c["port"] for c in server.connections] == [80, 8080]
    assert server.host_headers == [host, f"{host}:8080"]


def test_a_binary_content_type_is_accepted_and_rechecked_by_the_pipeline(net):
    """Plenty of CDNs answer ``application/octet-stream``. The magic-byte gate is what decides."""
    net["install"](routes={f"{PUBLIC}/photo.png":
                           (200, {"Content-Type": "application/octet-stream"}, PNG)})
    assert _img().data == PNG


def test_the_port_we_connect_to_is_the_port_we_vetted(net):
    server = net["install"](routes={f"{PUBLIC}/photo.png": _ok()})
    _img(f"http://{PUBLIC}:8443/photo.png")
    assert server.connections[0]["port"] == 8443


# ------------------------------------------------------------------------ refusals


def test_a_non_image_content_type_is_refused(net):
    net["install"](routes={f"{PUBLIC}/photo.png":
                           (200, {"Content-Type": "text/html"}, b"<html>not an image")})
    with pytest.raises(FetchError, match="not an image"):
        _img()


def test_an_upstream_error_does_not_become_a_status_oracle(net):
    """403 versus 404 tells an attacker whether a private path exists. We say one sentence."""
    net["install"](routes={f"{PUBLIC}/photo.png": (404, IMAGE_HEADERS, b"")})
    with pytest.raises(FetchError) as exc:
        _img()
    assert "did not return an image" in exc.value.message
    assert "404" not in exc.value.message and PUBLIC not in exc.value.message


def test_an_empty_body_is_refused(net):
    net["install"](routes={f"{PUBLIC}/photo.png": (200, IMAGE_HEADERS, b"")})
    with pytest.raises(FetchError, match="empty file"):
        _img()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://10.0.0.1:70/x",
    "ftp://internal.example/photo.png",
    "/not/a/url.png",
])
def test_a_scheme_that_is_not_http_or_https_never_reaches_dns(url, net):
    server = net["install"]()
    with pytest.raises(FetchError, match="Only http and https"):
        fetch_image(url, max_bytes=1000)
    assert server.connections == []


def test_a_literal_loopback_address_is_refused_without_an_identity_crisis(net):
    """The IPv6 spelling of "your own machine" written straight into the URL."""
    server = net["install"]()
    with pytest.raises(FetchError, match="private or reserved"):
        fetch_image("http://[::1]/x.png", max_bytes=1000)
    assert server.connections == []


def test_credentials_in_a_url_are_refused_without_a_lookup(net):
    """Whatever the resolver would answer, we will not put a secret in a request line or a log."""
    server = net["install"](dns={"public.example": [PUBLIC]})
    with pytest.raises(FetchError, match="Credentials"):
        fetch_image("http://user:pw@public.example/a.png", max_bytes=1000)
    assert server.connections == []


def test_a_url_with_no_hostname_is_refused(net):
    net["install"]()
    with pytest.raises(FetchError, match="no hostname"):
        fetch_image("http://", max_bytes=1000)


# ------------------------------------------------------------------ address vetting


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "127.16.0.1", "0.0.0.0", "10.20.30.40", "172.16.0.1", "192.168.1.1",
    "169.254.169.254", "100.64.0.1", "127.0.0.2", "224.0.0.1", "240.0.0.1", "192.0.2.1",
    "::1", "fe80::1", "fc00::1", "fd12::34", "::", "ff02::1",
    "::ffff:169.254.169.254", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
])
def test_a_non_public_address_is_refused(ip, net):
    server = net["install"](dns={"any.example": [ip]})
    with pytest.raises(FetchError, match="private or reserved"):
        fetch_image(f"http://any.example/{ip.replace(':', '_')}.png", max_bytes=1000)
    assert server.connections == []


@pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", V6])
def test_a_public_address_is_allowed(ip):
    assert _is_public(ipaddress.ip_address(ip))


def test_a_literal_ip_is_not_exempt_from_vetting():
    """The most common mistake: "the URL is an address, so there is nothing to resolve"."""
    assert not _is_public(ipaddress.ip_address("127.0.0.1"))
    assert _is_public(ipaddress.ip_address("93.184.216.34"))


def test_dns_rebinding_one_public_and_one_loopback_record_is_refused(net):
    """Checking only the first answer is the whole attack.

    A round-robin name that answers 93.x *and* 127.0.0.1 gets the public URL past a naive check and
    then hits the machine on whichever attempt the resolver hands out the loopback record.
    """
    host = "rebind.example"
    server = net["install"](dns={host: [PUBLIC, "127.0.0.1"]})
    with pytest.raises(FetchError, match="private or reserved"):
        fetch_image(f"http://{host}/a.png", max_bytes=1000)
    assert server.connections == []


def test_an_unresolvable_name_says_only_that(net):
    net["install"](dns={})
    with pytest.raises(FetchError) as exc:
        fetch_image("http://no-such-host.example/a.png", max_bytes=1000)
    assert exc.value.message == "That hostname does not resolve."


def test_a_name_with_no_records_says_the_same(net):
    net["install"](dns={"empty.example": []})
    with pytest.raises(FetchError, match="does not resolve"):
        fetch_image("http://empty.example/a.png", max_bytes=1000)


def test_ipv4_is_preferred_when_both_families_are_offered(net):
    """One vetted value to pin is a smaller surface than two."""
    host = "dual.example"
    server = net["install"](dns={host: [V6, PUBLIC]}, routes={f"{host}/a.png": _ok()})
    _img(f"http://{host}/a.png")
    assert server.connections[0]["ip"] == PUBLIC


def test_an_ipv6_only_name_is_still_connectable(net):
    host = "v6.example"
    server = net["install"](dns={host: [V6]}, routes={f"{host}/a.png": _ok()})
    _img(f"http://{host}/a.png")
    assert server.connections[0]["ip"] == V6


# ----------------------------------------------------------------------- redirects


def test_a_public_redirect_is_followed(net):
    routes = {"start.example/a.png": (302, {"Location": "https://end.example/a.png"}, b""),
              "end.example/a.png": _ok()}
    server = net["install"](dns={"start.example": [PUBLIC], "end.example": ["8.8.8.8"]},
                            routes=routes)
    assert _img("http://start.example/a.png").data == PNG
    assert [c["host"] for c in server.connections] == ["start.example", "end.example"]


def test_a_redirect_inward_is_refused_even_from_a_public_first_hop(net):
    """The classic: a URL that looks fine and a 302 that does not.

    Each hop restarts the vetting from the scheme, so the second address is checked on its own
    merits rather than inherited from the first.
    """
    routes = {"open.example/a.png":
              (302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b"")}
    server = net["install"](dns={"open.example": [PUBLIC]}, routes=routes)
    with pytest.raises(FetchError, match="private or reserved"):
        _img("http://open.example/a.png")
    assert [c["host"] for c in server.connections] == ["open.example"], \
        "the metadata address was never dialled"


def test_a_scheme_downgrade_on_redirect_is_vetted_like_any_other_hop(net):
    routes = {"secure.example/x.png": (302, {"Location": "http://10.0.0.9/x.png"}, b"")}
    server = net["install"](dns={"secure.example": [PUBLIC]}, routes=routes)
    with pytest.raises(FetchError, match="private or reserved"):
        fetch_image("https://secure.example/x.png", max_bytes=1000)
    assert len(server.connections) == 1


def test_a_relative_redirect_is_resolved_against_the_current_url(net):
    routes = {"a.example/deep/path/a.png": (301, {"Location": "/b.png"}, b""),
              "a.example/b.png": _ok()}
    server = net["install"](dns={"a.example": [PUBLIC]}, routes=routes)
    assert _img("http://a.example/deep/path/a.png").data == PNG
    assert [c["ip"] for c in server.connections] == [PUBLIC, PUBLIC]


def test_a_query_string_survives_the_redirect_join(net):
    routes = {"a.example/x": (302, {"Location": "http://a.example/b.png?size=large"}, b""),
              "a.example/b.png": _ok()}
    server = net["install"](dns={"a.example": [PUBLIC]}, routes=routes)
    assert _img("http://a.example/x").data == PNG
    assert server.paths == ["/x", "/b.png?size=large"], "a dropped query is a different image"


def test_a_redirect_loop_is_cut_off(net):
    routes = {f"h{i}.example/x": (302, {"Location": f"http://h{i + 1}.example/x"}, b"")
              for i in range(8)}
    server = net["install"](dns={f"h{i}.example": [PUBLIC] for i in range(9)}, routes=routes)
    with pytest.raises(FetchError, match="too many times"):
        _img("http://h0.example/x")
    # The refusal happens when the *next* hop would be the fourth redirect, so four requests ran.
    assert len(server.connections) == fetch.MAX_REDIRECTS + 1


def test_a_redirect_with_no_location_is_not_a_free_pass(net):
    net["install"](dns={"a.example": [PUBLIC]},
                   routes={"a.example/x": (302, dict(IMAGE_HEADERS), b"")})
    with pytest.raises(FetchError, match="could not be downloaded"):
        _img("http://a.example/x")


# -------------------------------------------------------------------- size ceilings


def test_a_lying_content_length_is_refused_before_the_body_is_read(net):
    net["install"](dns={"a.example": [PUBLIC]}, routes={"a.example/x.png": _ok(
        b"x" * 10, **{"Content-Length": "1000000000"})})
    with pytest.raises(FetchError, match="larger than this service accepts"):
        fetch_image("http://a.example/x.png", max_bytes=1000)


def test_an_undeclared_body_over_the_read_ceiling_is_caught_while_reading(net):
    """Chunked has no length to trust, so the ceiling has to be the read loop."""
    big = b"x" * (5_000 + fetch.READ_CEILING_SLACK)
    net["install"](dns={"a.example": [PUBLIC]}, routes={"a.example/x.png": _ok(big)})
    with pytest.raises(FetchError, match="larger than this service accepts"):
        fetch_image("http://a.example/x.png", max_bytes=1000)


def test_the_slack_between_the_read_ceiling_and_the_api_cap_is_still_refused(net):
    """Two ceilings, and the smaller one still has the final say."""
    body = b"x" * (2_000 + fetch.READ_CEILING_SLACK)
    net["install"](dns={"a.example": [PUBLIC]}, routes={"a.example/x.png": _ok(body)})
    with pytest.raises(FetchError, match="larger than this service accepts"):
        fetch_image("http://a.example/x.png", max_bytes=2_000)


def test_a_body_exactly_at_the_cap_is_accepted(net):
    body = PNG + b"y" * (1_000 - len(PNG))
    net["install"](dns={"a.example": [PUBLIC]}, routes={"a.example/x.png": _ok(body)})
    assert len(fetch_image("http://a.example/x.png", max_bytes=1_000).data) == 1_000


def test_a_garbage_content_length_header_is_ignored_not_fatal(net):
    net["install"](dns={"a.example": [PUBLIC]}, routes={"a.example/x.png":
                                                        _ok(PNG, **{"Content-Length": "banana"})})
    assert len(fetch_image("http://a.example/x.png", max_bytes=1_000_000).data) == len(PNG)


# ------------------------------------------------------------------------- deadlines


def test_the_total_deadline_stops_a_slow_redirect_loop(net, monkeypatch):
    """A server that trickles one redirect per second must not become a job that never finishes."""
    server = net["install"](dns={"a.example": [PUBLIC]})
    base = time.monotonic()
    ticks = iter([base, base + fetch._TOTAL_DEADLINE_S + 1])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    with pytest.raises(FetchError, match="too long"):
        fetch_image("http://a.example/x.png", max_bytes=1000)
    assert server.connections == []


# ------------------------------------------------------------------- internal guards


def test_resolve_and_vet_returns_the_address_that_will_actually_be_pinned(net):
    """The value handed back is the one the socket connects to, and it is always a vetted answer."""
    net["install"](dns={"mixed.example": [V6, PUBLIC, "8.8.8.8"]})
    assert _resolve_and_vet("mixed.example", 8080) == (PUBLIC, 8080)


def test_resolve_and_vet_refuses_an_empty_host(net):
    net["install"]()
    with pytest.raises(FetchError, match="no hostname"):
        _resolve_and_vet("", 80)


def test_the_tls_connection_pins_sni_to_the_hostname():
    """The line that keeps IP pinning from becoming "accept any certificate on that box"."""
    import inspect

    assert "server_hostname=self.host" in inspect.getsource(fetch._PinnedHTTPS.connect)
