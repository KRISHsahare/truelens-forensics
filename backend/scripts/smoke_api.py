"""Exercise every HTTP route against a real app instance.

The orchestrator smoke runs the pipeline in-process and therefore proves the analysers work. This
proves the *transport* works: that a multipart upload becomes a job, that a poll returns a schema
the client can render, that an overlay URL survives the round trip through signing and back out of
the filesystem, and that every rejection returns the one error envelope rather than a framework
page. Both are run before a commit; only this one can catch a route wired to the wrong dependency.

The app is driven through ``TestClient``, which runs the real lifespan (pool start, model warm-up,
sweeper task) and therefore catches startup failures a bare import would miss.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.core.logconfig import configure_logging
from app.main import create_app

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "images"

failures = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global failures
    if ok:
        print(f"    PASS  {label}")
    else:
        failures += 1
        print(f"    FAIL  {label}  {detail}")


def _img(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _envelope_shape(body: Any, code: str) -> bool:
    return (
        isinstance(body, dict)
        and body.get("error") == code
        and isinstance(body.get("message"), str)
        and body["message"].strip() != ""
    )


def health(client: TestClient) -> None:
    print("  health")
    r = client.get("/api/health")
    check(r.status_code == 200, "GET /api/health 200", str(r.status_code))
    body = r.json()
    check(body.get("status") in ("ok", "degraded"), "status ok/degraded", str(body.get("status")))
    check(body.get("app") == "TrueLens", "app name", str(body.get("app")))
    check("cache" in body and "concurrency" in body, "cache + concurrency blocks present")
    print(f"        {json.dumps({k: body[k] for k in ('status', 'version', 'environment', 'perf_profile')}, default=str)}")
    print(f"        artifacts={json.dumps(body.get('artifacts', {}), default=str)}")


def metrics(client: TestClient) -> None:
    print("  metrics")
    r = client.get("/api/metrics")
    check(r.status_code == 200, "GET /api/metrics 200", str(r.status_code))
    body = r.json()
    check({"detectors", "fusion", "techniques", "evaluation"} <= set(body), "four top-level blocks")
    check(isinstance(body["techniques"], list) and len(body["techniques"]) > 0,
          "technique catalogue non-empty", str(len(body.get("techniques", []))))
    ev = body.get("evaluation") or {}
    if not ev.get("available", False):
        print(f"        evaluation reported unavailable: {ev.get('reason', '')[:70]}")


def _wait(client: TestClient, job_id: str, *, timeout_s: float = 60.0) -> dict[str, Any]:
    """Poll until terminal, returning the last status body.

    Quiet per poll on purpose: a job takes twenty polls and the log would say PASS forty times. What
    is worth asserting is the *transport* — every poll answered 200 and the body carried stages — and
    a single non-200 is enough to stop and report.
    """
    global failures
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    polls = 0
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        polls += 1
        if r.status_code != 200:
            failures += 1
            print(f"    FAIL  poll {job_id[:8]} returned {r.status_code}: {r.text[:120]}")
            return last
        last = r.json()
        if last["status"] in ("succeeded", "failed", "expired"):
            print(f"        {polls} poll(s) to terminal; stages={last.get('stages')}")
            return last
        time.sleep(0.1)
    failures += 1
    print(f"    FAIL  job {job_id[:8]} never reached a terminal state")
    return last


def analyze(client: TestClient) -> dict[str, Any]:
    print("  analyze (upload)")
    started = time.monotonic()
    r = client.post(
        "/api/analyze",
        files={"file": ("spliced.jpg", _img("spliced.jpg"), "image/jpeg")},
    )
    check(r.status_code == 200, "POST /api/analyze 200", f"{r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        return {}
    created = r.json()
    check(len(created.get("job_id", "")) >= 8, "job_id returned", str(created.get("job_id")))
    check(created.get("stages"), "stages shipped with the job id", str(created.get("stages")))
    status = _wait(client, created["job_id"])
    check(status.get("status") == "succeeded", "job succeeded", str(status.get("error")))
    check(status.get("progress") == 1.0, "finished job reports full progress",
          str(status.get("progress")))
    check(not [n for n, s in (status.get("stages") or {}).items()
               if s in ("pending", "running")],
          "no stage left spinning on a finished job", str(status.get("stages")))
    result = status.get("result") or {}
    wall = time.monotonic() - started
    check(wall < 30, f"end-to-end in {wall:.2f}s")
    for field in ("verdict", "ai_score", "edited_score", "overall", "reliability",
                  "techniques", "metadata", "c2pa", "sha256", "timings"):
        check(field in result, f"result.{field}", str(sorted(result)[:6]))
    check(result.get("band") in ("low", "moderate", "high"), "risk band", str(result.get("band")))
    check(len(result.get("reasons", [])) > 0, "reasons non-empty")
    check(len(result.get("limitations", [])) > 0, "limitations non-empty")
    # The central honesty invariant, asserted where a regression cannot slip past: a check that did
    # not measure must not carry a number, because the gauge colours a 0.0 green and a green means
    # "this test passed". ``measured`` is the engine's own admission that it has nothing to say.
    silent = [(name, v.get("score")) for name, v in result.get("techniques", {}).items()
              if not v.get("measured", True) and v.get("score") is not None]
    check(not silent, "no unmeasured technique carries a score", str(silent))
    over = float(result.get("overall") or 0.0)
    print(f"        verdict={result.get('verdict')} ai={result.get('ai_score'):.3f} "
          f"edited={result.get('edited_score'):.3f} overall={over:.3f} "
          f"reliability={result.get('reliability'):.3f} fusion={result.get('fusion', {}).get('source')}")
    for name, view in sorted(result.get("techniques", {}).items()):
        print(f"          {name:<18} score={view.get('score')} status={view.get('status')} "
              f"measured={view.get('measured')} weight={view.get('explanation', {}).get('weight')} "
              f"overlay={'y' if view.get('overlay_url') else '-'}")
    return result


def overlays(client: TestClient, result: dict[str, Any]) -> None:
    print("  overlays")
    urls = [v["overlay_url"] for v in result.get("techniques", {}).values() if v.get("overlay_url")]
    check(len(urls) > 0, f"{len(urls)} overlay URL(s)")
    if not urls:
        return
    url = urls[0]
    check(url.startswith("/api/overlays/"), "relative URL", url[:60])
    r = client.get(url)
    check(r.status_code == 200, "signed URL 200", str(r.status_code))
    check(r.headers.get("content-type") == "image/png", "content-type png", r.headers.get("content-type"))
    check(r.headers.get("x-content-type-options") == "nosniff", "nosniff header")
    check(len(r.content) > 100, "png bytes", str(len(r.content)))
    check("no-store" in r.headers.get("cache-control", ""), "no-store cache-control")

    # Tampered signature.
    bad = url.replace("sig=", "sig=00") if not url.endswith("=") else url + "00"
    r = client.get(bad)
    check(r.status_code == 404, "tampered signature rejected", str(r.status_code))
    check(_envelope_shape(r.json(), "not_found"), "tamper response is the envelope")

    # Right signature, different job: the last character of the nonce flipped so the id still passes
    # the allowlist and only the signature can be what rejects it.
    job, tail = url.split("/")[3], url.split("/", 3)[3]
    other_job = job[:-1] + ("0" if job[-1] != "0" else "1")
    r = client.get(f"/api/overlays/{other_job}/{tail}")
    check(r.status_code == 404, "signature does not transfer between jobs", str(r.status_code))

    # Unsigned.
    head = url.split("?")[0]
    r = client.get(head)
    check(r.status_code == 404, "unsigned request rejected", str(r.status_code))

    # Correctly signed, but for a window that has already closed. Minted with the deployment's own
    # key so the only thing failing is the expiry check.
    from app.config import get_settings
    from app.core.signing import OverlaySigner

    name = tail.split("?")[0].removesuffix(".png")
    stale = OverlaySigner(get_settings().overlay_signing_key).url(
        job, name, now=time.time() - 10_000)
    r = client.get(stale)
    check(r.status_code == 404, "expired signature rejected", str(r.status_code))

    # Traversal: a job id and a name that are not the pair we minted, addressed through the root.
    job = url.split("/")[3]
    for probe in (
        f"/api/overlays/{job}/..%2f..%2f..%2fconfig.py",
        f"/api/overlays/..%2f{job}/heartbeat.png",
        f"/api/overlays/{job}/ela.PNG",
        f"/api/overlays/{job}/ela.png.exe",
    ):
        r = client.get(probe)
        check(r.status_code in (404, 422), f"traversal/refusal probe {probe[13:40]!r}", str(r.status_code))


def cache_replay(client: TestClient) -> None:
    print("  cache replay over HTTP")
    data = _img("small.jpg")
    r1 = client.post("/api/analyze", files={"file": ("a.jpg", data, "image/jpeg")})
    check(r1.status_code == 200, "first submit", r1.text[:200])
    s1 = _wait(client, r1.json()["job_id"])
    check(not r1.json().get("cached"), "first submit not cached")
    r2 = client.post("/api/analyze", files={"file": ("a.jpg", data, "image/jpeg")})
    check(r2.status_code == 200, "second submit", r2.text[:200])
    created = r2.json()
    check(created.get("cached") is True, "second submit flagged cached", str(created.get("cached")))
    # Immediately poll: the record must already exist when the job id was returned.
    r = client.get(f"/api/jobs/{created['job_id']}")
    check(r.status_code == 200, "cache-hit job is pollable at once", str(r.status_code) + " " + r.text[:120])
    if r.status_code == 200:
        check(r.json()["status"] == "succeeded", "cache-hit job already succeeded", r.json()["status"])
        urls = [v.get("overlay_url") for v in (r.json().get("result") or {}).get("techniques", {}).values()
                if v.get("overlay_url")]
        ok = all(client.get(u).status_code == 200 for u in urls)
        check(ok, f"{len(urls)} re-signed overlays still fetch")


def url_import(client: TestClient) -> None:
    print("  url import")
    for target, why in [
        ("http://127.0.0.1:9/x.jpg", "loopback"),
        ("http://169.254.169.254/latest/meta-data/", "link-local metadata"),
        ("http://10.0.0.5/a.jpg", "private range"),
        ("file:///etc/passwd", "non-http scheme"),
        ("not-a-url", "unparseable"),
    ]:
        r = client.post("/api/analyze", data={"url": target})
        check(r.status_code in (400, 415, 422), f"SSRF refused ({why})", str(r.status_code))
        if r.status_code in (400, 415):
            check(isinstance(r.json().get("message"), str), f"refusal ({why}) uses envelope")
    r = client.post("/api/analyze", data={"url": "https://example.invalid/x.jpg"})
    check(r.status_code in (400, 415, 422, 502), "unresolvable host refused", str(r.status_code))
    r = client.post("/api/analyze", data={"url": "https://example.com/x.jpg", "file": ("x", "", "text/plain")})
    check(r.status_code in (200, 400, 415, 422), "url + empty file does not 500", str(r.status_code))


def rejections(client: TestClient) -> None:
    print("  rejections")
    r = client.post("/api/analyze", files={"file": ("x.jpg", b"%PDF-1.7\n" + b"x" * 200, "image/jpeg")})
    check(r.status_code == 415, "pdf disguised as jpg", str(r.status_code))
    check(_envelope_shape(r.json(), "unsupported_media"), "pdf rejection envelope", str(r.json())[:160])

    r = client.post("/api/analyze", files={"file": ("t.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    check(r.status_code == 415, "truncated png", str(r.status_code))

    r = client.post("/api/analyze", files={"file": ("e.jpg", b"", "image/jpeg")})
    check(r.status_code == 400, "empty body", str(r.status_code))
    check(_envelope_shape(r.json(), "invalid_image"), "empty-body envelope")

    r = client.post("/api/analyze", files={"file": ("n.txt", b"hello", "text/plain")})
    check(r.status_code == 415, "text/plain", str(r.status_code))

    r = client.post("/api/analyze", data={})
    check(r.status_code == 400, "no part at all", str(r.status_code))

    junk = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * (11 * 1024 * 1024))
    r = client.post("/api/analyze", files={"file": ("big.jpg", junk, "image/jpeg")})
    check(r.status_code in (413, 400), "oversize upload refused", str(r.status_code))
    if r.status_code != 413:
        print(f"        (oversize surfaced as {r.status_code}: {_plain(r.json())})")

    r = client.get("/api/jobs/nope")
    check(r.status_code == 404, "unknown job 404", str(r.status_code))
    check(_envelope_shape(r.json(), "not_found"), "unknown-job envelope")

    r = client.get("/api/jobs/" + ("../etc/passwd"))
    check(r.status_code in (404, 400), "malformed job id refused", str(r.status_code))

    r = client.get("/api/nope")
    check(r.status_code == 404, "unknown route 404")
    check(_envelope_shape(r.json(), "not_found"), "unknown-route envelope")

    r = client.post("/api/health")
    check(r.status_code == 405, "wrong method 405", str(r.status_code))


def _plain(body: Any) -> str:
    return json.dumps(body, default=str)[:120]


def batch(client: TestClient) -> None:
    print("  batch")
    files = [("files", ("a.jpg", _img("small.jpg"), "image/jpeg")),
             ("files", ("b.jpg", _img("authentic.jpg"), "image/jpeg")),
             ("files", ("bad.txt", b"nope", "text/plain"))]
    r = client.post("/api/batch", files=files)
    check(r.status_code == 200, "POST /api/batch 200", f"{r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        return
    body = r.json()
    check(len(body["items"]) == 3, "one row per input", str(len(body["items"])))
    check(body["accepted"] == 2 and body["rejected"] == 1,
          "accepted/rejected counts", f"{body['accepted']}/{body['rejected']}")
    check(body["limit"] == 20, "limit echoed", str(body["limit"]))
    bad = [i for i in body["items"] if i["status"] == "failed"]
    check(len(bad) == 1 and bad[0]["error"], "rejected row carries a reason", _plain(bad))
    for item in body["items"]:
        if item["job_id"]:
            s = _wait(client, item["job_id"], timeout_s=40)
            check(s["status"] == "succeeded", f"batch job {item['label'] or item['job_id'][:8]} succeeded",
                  str(s.get("error")))
    r = client.post("/api/batch", files=[("files", ("a.jpg", b"", "image/jpeg"))])
    check(r.status_code == 400, "batch with nothing usable 400", str(r.status_code))
    check(_envelope_shape(r.json(), "invalid_image"), "empty batch envelope")


def rate_limit(client: TestClient) -> None:
    print("  rate limit")
    hit_429 = None
    for i in range(40):
        r = client.post("/api/analyze", files={"file": ("e.jpg", b"", "image/jpeg")})
        if r.status_code == 429:
            hit_429 = (i, r)
            break
    check(hit_429 is not None, "analyze eventually returns 429", "40 requests went through")
    if hit_429:
        i, r = hit_429
        body = r.json()
        check(_envelope_shape(body, "rate_limited"), "429 envelope", _plain(body))
        check(isinstance(body.get("retry_after_seconds"), int) and body["retry_after_seconds"] > 0,
              "retry_after_seconds present", str(body.get("retry_after_seconds")))
        check(r.headers.get("retry-after"), "Retry-After header", str(r.headers.get("retry-after")))
        print(f"        limited after {i + 1} calls")
        check(client.get("/api/jobs/whatever").status_code in (404, 429), "other routes still answered")


def main() -> int:
    configure_logging("INFO")
    app = create_app()
    with TestClient(app) as client:
        health(client)
        metrics(client)
        result = analyze(client)
        overlays(client, result)
        cache_replay(client)
        url_import(client)
        rejections(client)
        batch(client)
        rate_limit(client)
    print()
    if failures:
        print(f"  {failures} FAILURES")
        return 1
    print("  api smoke: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
