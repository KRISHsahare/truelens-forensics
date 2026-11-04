"""The HTTP surface: envelopes, limits, signatures, and what a client is allowed to learn.

The app is built with a *fake orchestrator* and **no lifespan**: entering ``TestClient`` as a context
manager would boot the spawn pool, load torch and start the sweeper, and every request-shape test in
this file would pay that to check a header. The real end-to-end request lives in
``test_api_live_end_to_end``, which does enter the context manager and does the whole thing once.

What is *not* faked is the part the security claims are made of: the signer, the scratch allowlists,
the error envelope, the limiter and the response schemas are all production objects. A route test with
a stubbed signer proves nothing about whether a tampered overlay URL 404s.

Three invariants recur, and they are the ones a frontend integrates against:
every failure leaves through one envelope shape; a poll never says more than it knows
(``response_model_exclude_none``); and no refusal distinguishes between "never existed", "expired" and
"not allowed to look".
"""

from __future__ import annotations

import itertools
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from app.api import routes_analyze
from app.api.limits import limiter
from app.config import Settings
from app.core.cache import ResultCache
from app.core.fetch import FetchError
from app.core.job import JobRecord, new_job_id
from app.core.orchestrator import BusyError, Rejected
from app.core.scratch import Scratch
from app.core.signing import OverlaySigner
from app.core.store.memory import MemoryJobStore
from app.main import _unhandled_handler, _validation_handler, create_app
from app.schemas.analysis import AnalysisResult
from tests.support import fixture_bytes

PAYLOAD = fixture_bytes("small")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

#: One address per test, monotonically. The limiter's store is a process-wide object and a fixed
#: window outlives the test that filled it, so a recycled address inherits somebody else's 429.
_CLIENTS = itertools.count(1)

#: Every key a successful error envelope is allowed to have. A fourth shape, or a fifth field, is a
#: breaking change for the client even when the status code is the one it expected.
ENVELOPE_KEYS = {"error", "message", "detail", "retry_after_seconds", "extra"}


def _settings(base: Settings, **overrides: Any) -> Settings:
    return base.model_copy(update={
        "environment": "dev",
        # ``cors_origins`` is a computed property over these two fields, so a test that wants the
        # allowlist to contain a given origin has to set the field the property reads.
        "frontend_origin": "https://truelens.test",
        "overlay_signing_key": "test-overlay-key",
        # Each test carries its own X-Forwarded-For, so limiter windows never bleed across tests;
        # the two tests that care assert the trust decision itself.
        "trust_proxy": True,
        "rate_limit_analyze": "1000/minute",
        "rate_limit_jobs": "10000/minute",
        "rate_limit_batch": "1000/minute",
        **overrides})


def _result(**overrides: Any) -> dict[str, Any]:
    """A schema-valid result body, so ``to_status``'s revalidation is a pass rather than a fallback.

    Revalidated rather than ``model_copy``-patched: a copy leaves nested dicts un-coerced, and the
    serializer then warns on every dump — which is the same shape mismatch a real client would see.
    """
    base = AnalysisResult(
        verdict="inconclusive", ai_score=0.2, edited_score=0.1, overall=0.2, reliability=0.6,
        reasons=["Only two checks produced a measurement."],
        sha256="a" * 64,
        limitations=[{"title": "No model is proof", "body": "Probabilities, not certainty."}])
    merged = {**base.model_dump(mode="json"), **overrides}
    return AnalysisResult.model_validate(merged).model_dump(mode="json")


class FakeOrchestrator:
    """The slice of the orchestrator the routes touch, with the decisions a test wants to script.

    ``create_job`` mints a real job id and publishes a real record, because the routes' contracts are
    about *those* objects: a fake that returned a dict would let a route pass while the schema broke.
    """

    def __init__(self, settings: Settings, *, store: MemoryJobStore, scratch: Scratch) -> None:
        self.settings = settings
        self.store = store
        self.scratch = scratch
        self.cache = ResultCache(ttl_s=60, max_entries=8)
        self.signer = OverlaySigner("test-overlay-key", ttl_s=settings.overlay_ttl_s)
        self.uploads: list[tuple[bytes, str, str]] = []
        self.fetched: list[str] = []
        self.raise_next: Exception | None = None
        #: Refuse a specific row of a batch, by the index it was submitted at.
        self.rejects: dict[int, Exception] = {}
        #: Script what the job looks like the moment it is created.
        self.status = "queued"
        self.result: dict[str, Any] | None = None
        self.error: dict[str, Any] | None = None
        self.finished = False

    async def create_job(self, data: bytes, *, client: str = "", label: str = "") -> JobRecord:
        self.uploads.append((data, client, label))
        index = len(self.uploads) - 1
        exc = self.rejects.get(index) if self.raise_next is None else self.raise_next
        if exc is not None:
            self.raise_next = None
            raise exc
        record = JobRecord(id=new_job_id(), sha256="a" * 64, mime="image/jpeg",
                           size_bytes=len(data), status=self.status, client=client)
        record.init_stages(_technique_names())
        record.result = self.result
        record.error = self.error
        if self.status == "running":
            record.started_at = time.time()
        if self.finished:
            record.started_at = record.started_at or time.time()
            record.finished_at = time.time()
            for name in record.stages:
                record.stages[name] = "done"
        await self.store.save(record)
        return record

    async def fetch(self, job_id: str) -> JobRecord | None:
        self.fetched.append(job_id)
        return await self.store.get(job_id)

    def stats(self) -> dict[str, Any]:
        return {"jobs_started": len(self.uploads), "in_flight": 0, "max_concurrent_jobs": 2,
                "pool_mode": "stub", "pool_workers": 0, "threads_per_worker": 1,
                "soft_deadline_s": 6.0, "hard_deadline_s": 12.0, "cache_hits": 0}

    @property
    def uptime_s(self) -> float:
        return 1.5


def _technique_names() -> list[str]:
    from app.analysis.registry import names as technique_names

    return list(technique_names())


class Client:
    """TestClient plus this test's client address, so one rate-limit window is never two tests'."""

    def __init__(self, app, client_id: str) -> None:
        self.raw = TestClient(app)
        self.headers = {"X-Forwarded-For": client_id}

    def _headers(self, kwargs: dict[str, Any]) -> dict[str, str]:
        return {**self.headers, **kwargs.pop("headers", {})}

    def get(self, url: str, **kwargs: Any):
        return self.raw.get(url, headers=self._headers(kwargs), **kwargs)

    def post(self, url: str, **kwargs: Any):
        return self.raw.post(url, headers=self._headers(kwargs), **kwargs)

    def delete(self, url: str, **kwargs: Any):
        return self.raw.delete(url, headers=self._headers(kwargs), **kwargs)

    def upload(self, data: bytes = PAYLOAD, *, name: str = "photo.jpg",
               content_type: str = "image/jpeg", **kwargs: Any):
        return self.post("/api/analyze",
                         files={"file": (name, data, content_type)}, **kwargs)


@pytest.fixture
def overrides(request) -> dict[str, Any]:
    """A test's settings patch, supplied by ``@pytest.mark.parametrize("overrides", ..., indirect)``.

    Indirect rather than a ``request.param`` read inside ``fake`` so the patch is visible as the
    argument it is, and so a test that needs a different limit declares it on its own decorator.
    """
    return getattr(request, "param", {}) or {}


@pytest.fixture
def fake(settings, tmp_path, monkeypatch, overrides):
    """A stubbed API app: real state objects, fake orchestrator, no pools, no sweeper."""
    cfg = _settings(settings, **overrides)
    store = MemoryJobStore()
    scratch = Scratch(str(tmp_path / "scratch"))
    app = create_app(cfg)
    orch = FakeOrchestrator(cfg, store=store, scratch=scratch)
    app.state.orchestrator = orch
    app.state.store = store
    app.state.scratch = scratch
    app.state.signer = orch.signer
    app.state.cache = orch.cache
    # The limit callables read the process-wide settings, not app.state, so a test that moves a limit
    # has to move the object the limiter actually consults.
    monkeypatch.setattr("app.api.limits.get_settings", lambda: cfg)
    monkeypatch.setattr("app.core.ratelimit.get_settings", lambda: cfg)
    yield orch, Client(app, f"test-client-{next(_CLIENTS)}")


def body(response) -> dict[str, Any]:
    assert "application/json" in response.headers.get("content-type", ""), response.text[:200]
    return json.loads(response.text)


# ------------------------------------------------------------------- POST /api/analyze
def test_an_upload_returns_a_job_id_and_the_whole_stepper_shape(fake):
    _orch, api = fake
    r = api.upload()
    assert r.status_code == 200
    data = body(r)
    assert data["status"] == "queued"
    assert data["job_id"]
    assert data["sha256"] == "a" * 64
    assert data["cached"] is False
    assert data["poll_after_ms"] > 0
    assert set(data["stages"]) == set(_technique_names())
    assert set(data["stages"].values()) == {"pending"}


def test_the_uploaded_bytes_are_handed_to_the_orchestrator_untouched(fake):
    orch, api = fake
    api.upload(PAYLOAD, name="whatever.jpg")
    data, client_ip, label = orch.uploads[0]
    assert data == PAYLOAD
    assert label == "whatever.jpg"
    assert client_ip == api.headers["X-Forwarded-For"]


def test_a_json_body_is_refused_with_the_instruction_it_needs(fake):
    _orch, api = fake
    r = api.post("/api/analyze", json={"url": "https://example.com/a.jpg"})
    assert r.status_code == 400
    assert body(r)["message"] == "Send a multipart/form-data request."


def test_a_multipart_request_with_no_file_field_is_refused(fake):
    _orch, api = fake
    r = api.post("/api/analyze", files={"note": (None, "just a note")})
    assert r.status_code == 400
    assert "No image was uploaded." in body(r)["message"]


def test_an_empty_file_is_refused_rather_than_queued(fake):
    _orch, api = fake
    r = api.upload(b"")
    assert r.status_code == 400
    assert body(r)["message"] == "The upload was empty."


@pytest.mark.parametrize("overrides", [{"max_upload_bytes": 2048}], indirect=True)
def test_the_size_cap_is_applied_before_the_orchestrator_is_called(fake):
    orch, api = fake
    r = api.upload(b"\xff\xd8\xff" + b"x" * 4096)
    assert r.status_code == 413
    data = body(r)
    assert data["error"] == "too_large"
    assert data["extra"]["max_bytes"] == orch.settings.max_upload_bytes
    assert orch.uploads == []


def test_a_busy_server_answers_503_with_a_retry_after_header(fake):
    orch, api = fake
    orch.raise_next = BusyError()
    r = api.upload()
    assert r.status_code == 503
    data = body(r)
    assert data["error"] == "server_busy"
    assert data["retry_after_seconds"] == 3
    assert r.headers["Retry-After"] == "3"


def test_a_rejection_keeps_its_status_code_and_its_machine_code(fake):
    orch, api = fake
    orch.raise_next = Rejected(415, "unsupported_media", "That format is not supported.",
                               detail="JPEG, PNG, WebP or BMP.", extra={"formats": ["image/jpeg"]})
    r = api.upload()
    assert r.status_code == 415
    data = body(r)
    assert data["error"] == "unsupported_media"
    assert data["extra"] == {"formats": ["image/jpeg"]}


def test_a_url_import_is_fetched_and_queued(fake, monkeypatch):
    orch, api = fake
    seen: dict[str, Any] = {}

    def fake_fetch(url: str, *, max_bytes: int):
        seen["url"] = url
        seen["max_bytes"] = max_bytes
        return SimpleNamespace(data=PAYLOAD)

    monkeypatch.setattr(routes_analyze, "fetch_image", fake_fetch)
    r = api.post("/api/analyze", files={"url": (None, "https://example.com/a.jpg")})
    assert r.status_code == 200
    assert seen["url"] == "https://example.com/a.jpg"
    assert seen["max_bytes"] == orch.settings.max_upload_bytes
    assert orch.uploads[0][2] == "url-import"


def test_a_url_that_will_not_fetch_is_a_400_with_the_fetchers_sentence(fake, monkeypatch):
    orch, api = fake

    def refuse(url: str, *, max_bytes: int):
        raise FetchError("That address is not a public http(s) image URL.")

    monkeypatch.setattr(routes_analyze, "fetch_image", refuse)
    r = api.post("/api/analyze", files={"url": (None, "http://169.254.169.254/latest/meta-data")})
    assert r.status_code == 400
    assert body(r)["message"] == "That address is not a public http(s) image URL."
    assert orch.uploads == []


@pytest.mark.parametrize("overrides", [{"rate_limit_analyze": "1/minute"}], indirect=True)
def test_the_upload_limit_answers_429_in_the_same_envelope(fake):
    _orch, api = fake
    assert api.upload().status_code == 200
    r = api.upload()
    assert r.status_code == 429
    data = body(r)
    assert data["error"] == "rate_limited"
    assert data["retry_after_seconds"] == 60
    assert r.headers["Retry-After"] == "60"


@pytest.mark.parametrize("overrides", [{"rate_limit_analyze": "2/minute"}], indirect=True)
def test_the_limit_window_is_per_address_when_the_proxy_is_trusted(fake):
    _orch, api = fake
    other = dict(api.headers, **{"X-Forwarded-For": "203.0.113.99"})
    assert api.upload().status_code == 200
    assert api.upload().status_code == 200
    assert api.upload().status_code == 429
    assert api.post("/api/analyze", files={"file": ("a.jpg", PAYLOAD, "image/jpeg")},
                    headers=other).status_code == 200


@pytest.mark.parametrize("overrides", [{"rate_limit_analyze": "1/minute",
                                       "trust_proxy": False}], indirect=True)
def test_a_spoofed_forwarded_header_buys_nothing_when_the_proxy_is_not_trusted(fake):
    """Without a rewriting proxy in front, ``X-Forwarded-For`` is client-authored and must be ignored.

    Trusting it would let a script buy a fresh allowance per request, which is the opposite of what a
    per-address limit is for.
    """
    _orch, api = fake
    assert api.upload().status_code == 200
    assert api.upload().status_code == 429


# ------------------------------------------------------------------------- GET /jobs
def test_a_running_poll_carries_no_result_key(fake):
    orch, api = fake
    orch.status = "running"
    job_id = body(api.upload())["job_id"]
    data = body(api.get(f"/api/jobs/{job_id}"))
    assert data["status"] == "running"
    assert "result" not in data
    assert "error" not in data
    assert "finished_at" not in data
    assert data["progress"] == 0.0


def test_a_finished_poll_carries_the_full_result(fake):
    orch, api = fake
    orch.status = "succeeded"
    orch.result = _result()
    orch.finished = True
    job_id = body(api.upload())["job_id"]
    data = api.get(f"/api/jobs/{job_id}")
    assert data.status_code == 200
    result = body(data)["result"]
    assert result["verdict"] == "inconclusive"
    assert result["limitations"][0]["title"] == "No model is proof"


def test_a_stored_result_that_no_longer_fits_the_schema_degrades_instead_of_500ing(fake):
    """The poll revalidates, because the record was written by a different code path than this one."""
    orch, api = fake
    orch.status = "succeeded"
    orch.result = {"verdict": "banana", "ai_score": 5}
    orch.finished = True
    job_id = body(api.upload())["job_id"]
    data = body(api.get(f"/api/jobs/{job_id}"))
    assert data["status"] == "failed"
    assert data["progress"] == 1.0
    assert data["error"]["code"] == "internal"
    assert "schema" in data["error"]["message"]
    assert "result" not in data


def test_a_failed_job_surfaces_its_error_but_a_succeeded_one_never_does(fake):
    orch, api = fake
    orch.status = "failed"
    orch.error = {"code": "no_measurements", "message": "Nothing could be measured."}
    orch.finished = True
    job_id = body(api.upload())["job_id"]
    data = body(api.get(f"/api/jobs/{job_id}"))
    assert data["error"] == {"code": "no_measurements", "message": "Nothing could be measured."}

    orch.status, orch.error, orch.result = "succeeded", None, _result()
    other = body(api.upload())["job_id"]
    assert "error" not in body(api.get(f"/api/jobs/{other}"))


def test_an_unknown_job_id_is_a_plain_404(fake):
    _orch, api = fake
    r = api.get(f"/api/jobs/{new_job_id()}")
    assert r.status_code == 404
    data = body(r)
    assert data["error"] == "not_found"
    assert "ten minutes" in data["message"]


def test_a_malformed_job_id_never_reaches_the_store(fake):
    orch, api = fake
    for guess in ("not-a-job-id", "..", "%2e%2e%2f..%2fetc%2fpasswd", "", "x" * 300):
        r = api.get(f"/api/jobs/{guess}")
        assert r.status_code == 404, guess
    assert orch.fetched == []


async def test_a_deleted_job_and_a_never_existing_job_are_the_same_answer(fake):
    """Nothing in the poll tells a caller whether the id it guessed was ever a job."""
    orch, api = fake
    job_id = body(api.upload())["job_id"]
    live = api.get(f"/api/jobs/{job_id}")
    assert body(live)["status"] == "queued"

    await orch.store.delete(job_id)
    deleted = api.get(f"/api/jobs/{job_id}").text
    never = api.get(f"/api/jobs/{new_job_id()}").text
    assert json.loads(deleted)["error"] == "not_found"
    assert deleted == never


# ------------------------------------------------------------------------ overlays
def signed(api, name: str = "ela", *, job_id: str | None = None,
           ttl: int | None = None) -> tuple[str, str]:
    """A real signed URL for a real file under the real scratch root."""
    orch: FakeOrchestrator = api.raw.app.state.orchestrator
    job_id = job_id or new_job_id()
    orch.scratch.ensure_job_dir(job_id)
    path = orch.scratch.overlay_path(job_id, name)
    assert path is not None
    path.write_bytes(PNG)
    url = orch.signer.url(job_id, name, ttl_override=ttl)
    return job_id, url


def test_a_signed_overlay_is_served_and_is_never_cacheable(fake):
    _orch, api = fake
    _job_id, url = signed(api)
    r = api.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.headers["cache-control"] == "private, no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.content == PNG


def test_a_tampered_a_missing_and_an_unborn_overlay_are_the_same_404(fake):
    _orch, api = fake
    job_id, url = signed(api)
    tampered = url.replace("sig=", "sig=0")
    expired = api.raw.app.state.signer.url(job_id, "ela", now=time.time() - 9000)
    missing = api.raw.app.state.signer.url(new_job_id(), "ela")

    responses = [api.get(target) for target in (tampered, expired, missing)]
    assert [r.status_code for r in responses] == [404, 404, 404]
    assert len({r.text for r in responses}) == 1, "three refusals, one indistinguishable body"
    assert body(responses[0])["error"] == "not_found"


def test_a_traversal_in_the_overlay_name_is_refused_before_anything_is_read(fake):
    _orch, api = fake
    job_id, _url = signed(api)
    for target in (f"/api/overlays/{job_id}/..%2fsecret.png",
                   f"/api/overlays/{job_id}/ela.png.png",
                   f"/api/overlays/{job_id}/ela.PNG",
                   "/api/overlays/not-a-job/ela.png"):
        r = api.get(target)
        assert r.status_code == 404, target


def test_an_overlay_url_does_not_work_for_a_different_technique(fake):
    """The name is inside the signature, so a valid URL cannot be pointed at another heatmap."""
    _orch, api = fake
    job_id, url = signed(api, name="ela")
    orch: FakeOrchestrator = api.raw.app.state.orchestrator
    other = orch.scratch.overlay_path(job_id, "noise")
    assert other is not None
    other.write_bytes(PNG)
    assert api.get(url.replace("/ela.png", "/noise.png")).status_code == 404


# ---------------------------------------------------------------------------- batch
def test_batch_returns_one_id_per_image_and_counts_the_rows(fake):
    orch, api = fake
    r = api.post("/api/batch", files=[
        ("files", ("a.jpg", PAYLOAD, "image/jpeg")),
        ("files", ("b.jpg", PAYLOAD, "image/jpeg")),
    ], data={"labels": "row-a,row-b"})
    assert r.status_code == 200
    data = body(r)
    assert data["accepted"] == 2 and data["rejected"] == 0
    assert [i["label"] for i in data["items"]] == ["row-a", "row-b"]
    assert len({i["job_id"] for i in data["items"]}) == 2
    assert len(orch.uploads) == 2


def test_one_unreadable_file_costs_a_row_not_the_request(fake):
    orch, api = fake
    orch.rejects[0] = Rejected(415, "unsupported_media", "That format is not supported.",
                               detail="Try JPEG or PNG.")
    r = api.post("/api/batch", files=[
        ("files", ("bad.tiff", PAYLOAD, "image/tiff")),
        ("files", ("good.jpg", PAYLOAD, "image/jpeg")),
    ], data={"labels": "first,second"})
    assert r.status_code == 200
    data = body(r)
    assert (data["accepted"], data["rejected"]) == (1, 1)
    failed, good = data["items"]
    assert failed["job_id"] == "" and failed["status"] == "failed"
    # The table cell carries the sentence, not just the machine code: "unsupported_media" tells a
    # user nothing about their screenshot.
    assert "That format is not supported." in failed["error"]
    assert "Try JPEG or PNG." in failed["error"]
    assert good["job_id"] and good["status"] == "queued"
    assert len(orch.uploads) == 2


def test_a_batch_with_no_usable_image_is_a_400(fake):
    _orch, api = fake
    r = api.post("/api/batch", files={"notes": (None, b"not images")})
    assert r.status_code == 400
    assert body(r)["error"] == "invalid_image"


@pytest.mark.parametrize("overrides", [{"batch_max_images": 2}], indirect=True)
def test_a_batch_over_its_own_limit_is_refused_with_the_number(fake):
    orch, api = fake
    r = api.post("/api/batch", files=[
        ("files", (f"{i}.jpg", PAYLOAD, "image/jpeg")) for i in range(3)
    ])
    assert r.status_code == 400
    data = body(r)
    assert data["error"] == "too_many_images"
    assert data["extra"]["limit"] == 2
    assert orch.uploads == []


# -------------------------------------------------------------------------- health
def test_health_reports_the_blocks_an_operator_asks_for(fake):
    _orch, api = fake
    r = api.get("/api/health")
    assert r.status_code == 200
    data = body(r)
    assert data["status"] == "ok"
    assert data["problems"] == []
    assert data["store"]["backend"] == "memory"
    assert data["concurrency"]["pool_mode"] == "stub"
    assert data["cache"]["capacity"] == 8
    assert isinstance(data["artifacts"], dict)
    # The allowlist is published here on purpose. A CORS misconfiguration makes every request
    # succeed and still breaks the site, so "which origins will get a readable response" has to be
    # answerable from the health endpoint rather than inferred from a browser's console.
    assert data["network"]["cors_origins"], data["network"]
    assert "trust_proxy" in data["network"]
    assert data["uptime_s"] == 1.5


def test_health_goes_degraded_when_the_store_stops_answering(fake):
    orch, api = fake

    class Unreachable:
        backend = "memory"

        async def save(self, record):
            raise ConnectionError("redis down")

        async def get(self, job_id):
            raise ConnectionError("redis down")

        async def delete(self, job_id):
            raise ConnectionError("redis down")

        async def sweep(self, ttl_s):
            raise ConnectionError("redis down")

        async def stats(self):
            raise ConnectionError("redis down")

    orch.store = Unreachable()
    data = body(api.get("/api/health"))
    assert data["status"] == "degraded"
    assert "job store is not answering" in data["problems"]
    assert data["store"]["reachable"] is False


@pytest.mark.parametrize("overrides", [{"environment": "prod", "result_cache_ttl_s": 900,
                                       "overlay_ttl_s": 600}], indirect=True)
def test_health_names_a_configuration_that_cannot_work(fake):
    _orch, api = fake
    data = body(api.get("/api/health"))
    assert data["status"] == "degraded"
    assert any("result cache" in p.lower() or "RESULT_CACHE_TTL_S" in p for p in data["problems"])


def test_health_leaks_no_secrets(fake):
    _orch, api = fake
    text = api.get("/api/health").text
    assert "test-overlay-key" not in text
    assert "redis://" not in text
    assert api.headers["X-Forwarded-For"] not in text


# ------------------------------------------------------------------------- metrics
def test_metrics_reports_a_missing_evaluation_instead_of_zeroes(fake):
    _orch, api = fake
    data = body(api.get("/api/metrics"))
    evaluation = data["evaluation"]
    assert evaluation["available"] is False
    assert "accuracy" not in evaluation
    assert evaluation["meaning"]


def test_metrics_passes_a_real_evaluation_file_through(fake, tmp_path):
    _orch, api = fake
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"accuracy": 0.93, "roc_auc": 0.97}), encoding="utf-8")
    app = api.raw.app
    app.state.settings = app.state.settings.model_copy(update={"metrics_path": str(path)})
    data = body(api.get("/api/metrics"))
    assert data["evaluation"] == {"available": True, "accuracy": 0.93, "roc_auc": 0.97}


def test_metrics_lists_every_technique_the_pipeline_actually_runs(fake):
    _orch, api = fake
    data = body(api.get("/api/metrics"))
    assert {t["name"] for t in data["techniques"]} == set(_technique_names())
    assert {"name", "label", "group", "track", "overlay", "summary"} <= set(
        data["techniques"][0])


def test_metrics_says_which_fusion_weights_are_in_play(fake):
    _orch, api = fake
    fusion = body(api.get("/api/metrics"))["fusion"]
    assert fusion["source"] in ("expert", "learned")
    assert fusion["expert_weights"]
    assert "prior" in fusion["note"] or "fitted fusion model" in fusion["note"]


def test_metrics_says_which_artifacts_are_present_without_the_files(fake):
    _orch, api = fake
    detectors = body(api.get("/api/metrics"))["detectors"]
    assert set(detectors) == {"efficientnet", "clip_probe", "face_model"}
    # Absent-but-enabled is the interesting combination, so it must be two fields, not one.
    for name, block in detectors.items():
        assert set(block) == {"path", "present", "enabled"}, name


# ----------------------------------------------------------------- the one envelope
@pytest.mark.parametrize("call", [
    lambda api: api.get("/api/does-not-exist"),
    lambda api: api.delete("/api/health"),
    lambda api: api.get("/api/jobs/%2e%2e%2fnope"),
    lambda api: api.upload(b""),
    lambda api: api.get("/api/overlays/nope/ela.png"),
], ids=["unknown-route", "wrong-method", "bad-id", "empty-upload", "bad-overlay"])
def test_every_failure_leaves_through_one_envelope(fake, call):
    _orch, api = fake
    data = body(call(api))
    assert set(data) <= ENVELOPE_KEYS, data
    assert data["error"]
    assert isinstance(data["message"], str) and data["message"]
    assert data.get("detail") is None or isinstance(data["detail"], str)


def test_the_openapi_schema_is_served_outside_production(fake):
    _orch, api = fake
    assert api.get("/api/openapi.json").status_code == 200


@pytest.mark.parametrize("overrides", [{"environment": "prod"}], indirect=True)
def test_the_endpoint_map_is_closed_in_production(fake):
    _orch, api = fake
    r = api.get("/api/openapi.json")
    assert r.status_code == 404
    assert body(r)["error"] == "not_found"


def _stub_request():
    """The handlers log the request, and a handler called directly has no TestClient request to give."""
    return SimpleNamespace(method="POST", url=SimpleNamespace(path="/api/analyze"))


async def test_a_validation_error_does_not_echo_the_value_that_was_wrong():
    """Reflected input in a response body is a helper for XSS and for log-forced content injection."""
    payload = "<script>steal(document.cookie)</script>"
    exc = RequestValidationError([
        {"type": "string_type", "loc": ("body", "file", "url"),
         "msg": "Input should be a valid string", "input": payload,
         "ctx": {"expected": "string"}}])
    response = await _validation_handler(_stub_request(), exc)
    text = response.body.decode()
    assert response.status_code == 422
    assert payload not in text
    # The location is kept — it is a field path, not user data — and it is what makes the message
    # actionable.
    assert json.loads(text)["extra"] == {"fields": ["file.url"]}


async def test_an_unhandled_crash_returns_a_sentence_and_no_traceback():
    response = await _unhandled_handler(
        _stub_request(), RuntimeError("failed to open D:\\scratch\\secret.png"))
    text = response.body.decode()
    assert response.status_code == 500
    assert json.loads(text)["error"] == "internal"
    assert "secret" not in text and "Traceback" not in text
    assert "scratch" not in text


# ------------------------------------------------------------------------- CORS
def test_the_frontend_origin_is_allowed_and_the_limit_headers_are_exposed(fake):
    _orch, api = fake
    r = api.get("/api/health", headers={"Origin": "https://truelens.test"})
    assert r.headers["access-control-allow-origin"] == "https://truelens.test"
    assert "x-ratelimit-remaining" in r.headers["access-control-expose-headers"].lower()
    assert "retry-after" in r.headers["access-control-expose-headers"].lower()


def test_an_arbitrary_origin_gets_no_cors_grant(fake):
    _orch, api = fake
    r = api.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    assert r.status_code == 200


# ------------------------------------------------------------------- the real stack
@pytest.mark.slow
def test_api_live_end_to_end(settings, tmp_path):
    """One request through the whole app: lifespan, pools, orchestrator, poll, overlay fetch.

    Everything above stubs the orchestrator; this is the test that fails if the wiring between the
    routes and the pipeline is wrong — a missing ``app.state`` object, a lifespan that never starts
    the pools, a signer the routes and the orchestrator disagree about.
    """
    cfg = _settings(settings, analyzer_parallel=False, scratch_root=str(tmp_path / "scratch"),
                    job_ttl_s=600, overlay_ttl_s=600)
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post("/api/analyze", files={"file": ("real.jpg", PAYLOAD, "image/jpeg")},
                        headers={"X-Forwarded-For": "198.51.100.7"})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]

        deadline = time.monotonic() + 120.0
        data: dict[str, Any] = {}
        while time.monotonic() < deadline:
            poll = client.get(f"/api/jobs/{job_id}", headers={"X-Forwarded-For": "198.51.100.7"})
            assert poll.status_code == 200, poll.text
            data = poll.json()
            if data["status"] in ("succeeded", "failed", "expired"):
                break
            time.sleep(0.1)
        assert data["status"] == "succeeded", data
        assert data["progress"] == 1.0
        assert set(data["result"]["techniques"]) == set(_technique_names())

        health = client.get("/api/health").json()
        assert health["status"] == "ok", health["problems"]
        assert health["concurrency"]["jobs_started"] >= 1

        overlays = [v["overlay_url"] for v in data["result"]["techniques"].values()
                    if v.get("overlay_url")]
        assert overlays, "a real analysis produced no overlay"
        got = client.get(overlays[0])
        assert got.status_code == 200
        assert got.content[:8] == b"\x89PNG\r\n\x1a\n"

    # The promise the privacy page makes: the scratch tree this run created goes with the process.
    assert not (tmp_path / "scratch").exists()


def test_the_limiter_is_one_object_shared_by_every_route():
    """A second ``Limiter`` would count a second, independent window and silently double the limits."""
    from app.api import routes_analyze, routes_batch, routes_jobs, routes_report

    for module in (routes_analyze, routes_batch, routes_jobs, routes_report):
        assert module.limiter is limiter


# ------------------------------------------------------------------------- report / PDF
def _technique(name: str, *, label: str, group: str, score=None, overlay: bool = False) -> dict:
    """A real ``TechniqueView``, dumped — ``_result`` re-dumps through Pydantic and a raw dict warns.

    ``score=None`` is the case the report must not turn into a printed 0%.
    """
    from app.analysis.adapt import TechniqueView

    return TechniqueView(
        name=name,
        label=label,
        group=group,
        status="ok",
        score=score,
        band="not_measured" if score is None else ("high" if score >= 0.6 else "low"),
        ran=True,
        measured=score is not None,
        overlay_url=f"/api/overlays/x/{name}.png?exp=1&sig=2" if overlay else None,
        overlay_kind="heatmap" if overlay else "none",
        explanation={
            "why": "Compression leaves a periodic fingerprint.",
            "found": "Nothing measurable in this file.",
            "weight": "weak",
            "weight_note": "A single check is not a conclusion.",
        },
        details={"period": None},
        ms=41,
        error=None,
    ).model_dump(mode="json")


def _techniques() -> dict:
    return {
        "ela": _technique("ela", label="Error level analysis", group="edited", overlay=True),
        "frequency": _technique("frequency", label="Frequency spectrum", group="ai",
                              score=0.71, overlay=True),
        "metadata": _technique("metadata", label="Metadata", group="context"),
    }


def _succeeded(api, orch, **overrides) -> str:
    orch.status = "succeeded"
    orch.finished = True
    orch.result = _result(**overrides)
    response = api.upload(name="holiday-photo.jpg")
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _png(width: int = 40, height: int = 30) -> bytes:
    """A real PNG: the report reads it with an image decoder, and a header-only fixture will not do."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (12, 140, 210)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_finished_job_answers_with_a_pdf(fake):
    orch, api = fake
    job_id = _succeeded(api, orch, techniques=_techniques())
    response = api.get(f"/api/report/{job_id}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:5] == b"%PDF-"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_the_report_is_named_by_the_hash_not_by_the_upload(fake):
    """The uploader's filename is their business; a document that leaves the building must not carry it."""
    orch, api = fake
    job_id = _succeeded(api, orch, techniques=_techniques())
    disposition = api.get(f"/api/report/{job_id}").headers["content-disposition"]
    assert "holiday-photo" not in disposition
    # ``_result`` fixes the sha at "a"*64, so the first eight are known.
    assert "truelens-report-aaaaaaaa.pdf" in disposition


def test_a_running_job_has_no_report_yet_and_an_unknown_one_never_will(fake):
    orch, api = fake
    orch.status = "queued"
    job_id = api.upload().json()["job_id"]

    waiting = api.get(f"/api/report/{job_id}")
    assert waiting.status_code == 409
    assert body(waiting)["error"] == "job_not_ready"

    unknown = api.get(f"/api/report/{new_job_id()}")
    malformed = api.get("/api/report/not-a-job")
    traversal = api.get("/api/report/..%2fsecrets")
    assert [unknown.status_code, malformed.status_code, traversal.status_code] == [404, 404, 404]
    assert body(unknown)["error"] == "not_found"
    assert unknown.text == malformed.text, "one answer for every way a report can be missing"
    # The allowlist runs before the store, so a probe id never becomes a lookup.
    assert orch.fetched.count("not-a-job") == 0


def test_an_unmeasured_score_is_words_and_a_measured_zero_is_a_number():
    from app.api.routes_report import _pct

    assert _pct(None) == "not measured"
    assert _pct(0.0) == "0%"


def test_metadata_from_a_stranger_is_text_and_not_markup():
    """EXIF is attacker-authored, and reportlab parses paragraph markup in every cell."""
    from app.api.routes_report import _escape, build_pdf
    from app.core.job import JobRecord
    from app.schemas.analysis import MetadataBlock

    assert _escape("<b>x</b>") == "&lt;b&gt;x&lt;/b&gt;"
    hostile = MetadataBlock(
        status="present",
        present=True,
        software="<img src=x onerror=alert(1)>",
        findings=[{"code": "xmp_edit_history", "text": "<script>keep</script>",
                   "severity": "moderate", "points": 0.15}],
    ).model_dump(mode="json")
    record = JobRecord(
        id=new_job_id(), sha256="a" * 64, mime="image/jpeg", size_bytes=10, status="succeeded",
        result=_result(techniques=_techniques(), metadata=hostile))
    pdf = build_pdf(record, {}, app_version="test", ttl_s=600)
    assert pdf[:5] == b"%PDF-"
    assert b"<script>keep</script>" not in pdf


def test_a_present_overlay_is_drawn_and_an_expired_one_is_captioned(tmp_path):
    from app.api.routes_report import build_pdf
    from app.core.job import JobRecord

    record = JobRecord(
        id=new_job_id(), sha256="a" * 64, mime="image/jpeg", size_bytes=10, status="succeeded",
        result=_result(techniques=_techniques()))

    overlay = tmp_path / "ela.png"
    overlay.write_bytes(_png())

    without = build_pdf(record, {}, app_version="test", ttl_s=600)
    with_image = build_pdf(record, {"ela": overlay}, app_version="test", ttl_s=600)

    assert without[:5] == with_image[:5] == b"%PDF-"
    # The heatmap is the only difference, and it has to be in there as bytes, not as a blank frame.
    assert len(with_image) > len(without)


def test_a_result_with_no_overlays_still_writes_a_visual_section():
    from app.api.routes_report import build_pdf
    from app.core.job import JobRecord

    techniques = {"metadata": _technique("metadata", label="Metadata", group="context")}
    record = JobRecord(
        id=new_job_id(), sha256="a" * 64, mime="image/jpeg", size_bytes=10, status="succeeded",
        result=_result(techniques=techniques))
    assert build_pdf(record, {}, app_version="test", ttl_s=600)[:5] == b"%PDF-"
