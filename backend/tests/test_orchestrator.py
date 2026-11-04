"""The orchestrator: from uploaded bytes to a finished, honest job record.

Everything here replaces ``worker.run_group`` — the function that crosses the process boundary — with
a programmable stub, so a test can say "this group takes four seconds" or "this technique errored"
without waiting for real image analysis. That substitution is the whole point: the orchestrator's
responsibilities are admission, capacity, deadlines, progress truth and cache safety, and every one of
them is only observable when the workers misbehave. A test that ran the real analysers could check the
happy path and nothing else, and the happy path is where no bugs live.

Three tests do run the real thing. One drives the in-process groups over a fixture image and asserts
the full result envelope; one seeds the cache and replays it; the ``pool`` test boots a real spawn
``ProcessPoolExecutor``, because the queue reader, the pickling and the child-side configuration have
no stand-in.

The recurring assertion is the one the product rests on: a semaphore slot, a runs-table entry and a
scratch directory must be returned on *every* exit path, including the paths where the server itself
is at fault. Several of these tests failed the first time they were written, and what they found was
not inside an analyser.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest

from app.analysis import worker
from app.analysis.registry import names as technique_names
from app.analysis.types import TaskOutcome
from app.config import Settings
from app.core.cache import ResultCache
from app.core.job import JobRecord
from app.core.orchestrator import _INPROC_GROUPS, BusyError, Orchestrator, Rejected
from app.core.pipeline import sha256_bytes
from app.core.scratch import Scratch
from app.core.signing import OverlaySigner
from tests.support import CLEAN_DETAILS, fixture_bytes, outcome

REGISTRY = list(technique_names())

#: A real, decodable JPEG. Magic bytes are not enough: ``probe_header`` opens the file with PIL, and a
#: header that will not parse is refused before admission, which is the behaviour the rejection ladder
#: tests and the wrong substrate for everything else. ``tag()`` appends trailing bytes for the tests
#: that need a distinct SHA-256 — PIL stops at the JPEG's EOI marker and ignores them, so the same
#: image is served to the pipeline while the cache sees a different file.
PAYLOAD = fixture_bytes("small")


def tag(suffix: bytes = b"") -> bytes:
    """The default image with a distinct SHA-256, and still the same decodable picture."""
    return PAYLOAD + suffix


# --------------------------------------------------------------------------- doubles
class StubStore:
    """A ``JobStore`` that is a dict, plus a log of every publish and a refusal switch.

    It saves ``to_dict()`` snapshots and hands back reconstructed records, so a test cannot pass by
    reading the orchestrator's live object through a shared reference — which is exactly how "the
    record was published before it was finished" would hide.
    """

    def __init__(self) -> None:
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.publishes: list[tuple[str, str]] = []
        self.refuse_saves = False

    async def save(self, record: JobRecord) -> None:
        if self.refuse_saves:
            raise RuntimeError("redis is down")
        self.publishes.append((record.id, record.status))
        self.snapshots[record.id] = record.to_dict()

    async def get(self, job_id: str) -> JobRecord | None:
        snap = self.snapshots.get(job_id)
        return None if snap is None else JobRecord.from_dict(dict(snap))

    async def delete(self, job_id: str) -> None:
        self.snapshots.pop(job_id, None)

    async def sweep(self, ttl_s: float) -> int:
        stale = [jid for jid, snap in self.snapshots.items()
                 if time.time() - snap["created_at"] > ttl_s]
        for jid in stale:
            self.snapshots.pop(jid, None)
        return len(stale)

    def stats(self) -> dict[str, Any]:
        return {"backend": "stub", "jobs_held": len(self.snapshots)}

    async def close(self) -> None:
        pass


class StubRuntime:
    """Stand-in for the torch runtime so a warm-up does not import torch forty times.

    The warm-up *call* is still made, and asserted, because "load the model once, before the first
    request" is a requirement rather than a detail.
    """

    def __init__(self, weights_path: str | None = None, *, torch_threads: int = 4) -> None:
        self.weights_path = weights_path
        self.torch_threads = torch_threads

    def load(self) -> bool:
        return False

    @property
    def load_error(self) -> str | None:
        return "no weights in this test environment"

    def warmup(self) -> str:
        return "not loaded"


class Harness:
    """An orchestrator wired to stubs, with the knobs a test needs.

    The stub worker reads :attr:`answers` (per-technique outcome), :attr:`delays` (per-group sleep),
    :attr:`events` (stage transitions to emit, optionally without returning the outcome) and
    :attr:`gate` (block until the test releases it). Those four cover every scheduling question worth
    asking of a job runner.
    """

    def __init__(self, settings: Settings, tmp_path: Path, monkeypatch: Any,
                 *, stub_group: bool = True, stub_runtime: bool = True) -> None:
        self.settings = settings
        self.store = StubStore()
        self.signer = OverlaySigner("test-overlay-key", ttl_s=settings.overlay_ttl_s)
        self.cache = ResultCache(ttl_s=settings.result_cache_ttl_s, max_entries=32)
        self.scratch = Scratch(str(tmp_path / f"scratch-{time.time_ns()}"))
        self.calls: list[worker.GroupRequest] = []
        self.answers: dict[str, TaskOutcome] = {}
        self.delays: dict[str, float] = {}
        self.events: list[tuple[str, str, bool]] = []
        self.gate: threading.Event | None = None
        self.on_group: Callable[[worker.GroupRequest], None] | None = None
        self.orch = Orchestrator(settings, store=self.store, cache=self.cache,
                                 signer=self.signer, scratch=self.scratch)
        if stub_group:
            monkeypatch.setattr(worker, "run_group", self._run_group)
        if stub_runtime:
            # The real warm-up imports torch before it discovers that there are no weights, which
            # costs more than most of the tests it belongs to. The two that run the pipeline for real
            # opt out and pay it.
            monkeypatch.setattr("app.ml.cnn_runtime.get_runtime", StubRuntime)

    async def start(self) -> Harness:
        await self.orch.startup()
        return self

    # -- the stubbed worker ---------------------------------------------------------------
    def _run_group(self, request: worker.GroupRequest) -> worker.GroupResult:
        self.calls.append(request)
        if self.on_group is not None:
            self.on_group(request)
        delay = self.delays.get(_group_of(request.names), 0.0)
        if delay:
            time.sleep(delay)
        if self.gate is not None:
            assert self.gate.wait(10.0), "test forgot to release the group"

        returned: list[TaskOutcome] = []
        for name in request.names:
            answer = self.answers.get(name) or self._default_answer(name)
            returned.append(answer)
            for ev_name, status, with_outcome in self.events:
                if ev_name != name:
                    continue
                # ``worker._emit`` is the child's own emission path and ``startup()`` installed the
                # parent's callback, so this is the queue without the queue. ``with_outcome=False``
                # reproduces the failure mode where the stage settles but the payload is lost.
                worker._emit(request.job_id, ev_name, status, 3, answer if with_outcome else None)
                if not with_outcome:
                    returned.pop()  # this name is the last one appended
        return worker.GroupResult(job_id=request.job_id, outcomes=returned, wall_s=0.01)

    def _default_answer(self, name: str) -> TaskOutcome:
        return outcome(name, 0.0, CLEAN_DETAILS.get(name))

    # -- driving --------------------------------------------------------------------------
    async def submit(self, data: bytes = PAYLOAD, **kwargs: Any) -> JobRecord:
        return await self.orch.create_job(data, client="test", **kwargs)

    async def wait(self, job_id: str, *, timeout_s: float = 10.0,
                   predicate: Callable[[JobRecord], bool] | None = None) -> JobRecord:
        """Poll the store the way a client does, so publishing is part of the test."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            record = await self.orch.fetch(job_id)
            if record is not None and (predicate(record) if predicate else record.is_terminal()):
                return record
            await asyncio.sleep(0.01)
        raise AssertionError(f"job {job_id} never satisfied the wait ({timeout_s}s)")

    async def settle(self, job_id: str, *, timeout_s: float = 10.0) -> JobRecord:
        return await self.wait(job_id, timeout_s=timeout_s)

    def overlay_dir(self, job_id: str) -> Path:
        return Path(self.scratch.root) / job_id

    def age_out(self, job_id: str, *, seconds: float = 3600.0) -> None:
        """Make a finished job and its overlays look old, which is the only way to test the sweeper.

        A TTL of zero would depend on the filesystem's mtime resolution being finer than the gap
        between the write and the sweep, and on Windows it is not. Ageing the record and the directory
        is the scenario, stated as the scenario.
        """
        snap = self.store.snapshots.get(job_id)
        if snap is not None:
            snap["created_at"] = time.time() - seconds
        directory = self.overlay_dir(job_id)
        if directory.exists():
            old = time.time() - seconds
            os.utime(directory, (old, old))


def _group_of(names: tuple[str, ...]) -> str:
    return worker.group_for(names[0]) if names else ""


@pytest.fixture
async def harness(tmp_path, settings, monkeypatch):
    """In-process, fully stubbed orchestrator. Each test overrides the settings it cares about."""
    opened: list[Harness] = []

    async def build(**overrides: Any) -> Harness:
        # In-process groups unless a test asks otherwise: with the pool on, ``run_group`` is pickled
        # into a child, and a stub is a bound method that cannot cross that boundary. Those tests would
        # measure the IPC instead of the scheduling, and report a "failed group" for every one of them.
        cfg = {"analyzer_parallel": False, **overrides}
        h = await Harness(settings.model_copy(update=cfg), tmp_path, monkeypatch).start()
        opened.append(h)
        return h

    yield build

    for h in opened:
        await h.orch.shutdown()


# ------------------------------------------------------------------ rejection ladder
async def test_an_empty_upload_is_refused_before_a_job_exists(harness):
    h = await harness()
    with pytest.raises(Rejected) as exc:
        await h.submit(b"")
    assert (exc.value.status, exc.value.code) == (400, "invalid_image")
    assert h.store.snapshots == {}
    assert h.orch.stats()["jobs_started"] == 0


async def test_an_oversize_upload_reports_the_limit_it_hit(harness):
    cap = 1024
    h = await harness(max_upload_bytes=cap)
    with pytest.raises(Rejected) as exc:
        await h.submit(b"\xff\xd8\xff" + b"x" * (cap + 1))
    assert (exc.value.status, exc.value.code) == (413, "too_large")
    assert exc.value.extra == {"max_bytes": cap}
    assert h.store.snapshots == {}


async def test_a_non_image_keeps_the_security_error_its_own_status(harness):
    """The orchestrator relays ``SecurityError.status_code`` rather than flattening to 400.

    A 415 and a 400 are different client behaviours — the first says this format is never going to
    work, the second invites a retry with the same file — and the flattening would have been invisible
    in the routes, which just echo whatever the exception carries.
    """
    h = await harness()
    with pytest.raises(Rejected) as exc:
        await h.submit(b"%PDF-1.4\n" + b"%" * 600)
    assert (exc.value.status, exc.value.code) == (415, "unsupported_media")


async def test_every_kind_of_bad_upload_leaves_nothing_behind(harness):
    h = await harness()
    for payload in (b"", b"not an image at all", b"%PDF-1.4\n" + b"%" * 600,
                    b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, b"\xff\xd8\xff" + b"\x00" * 64):
        with pytest.raises(Rejected):
            await h.submit(payload)
    assert len(h.store.snapshots) == 0
    root = Path(h.scratch.root)
    assert not root.exists() or list(root.iterdir()) == []
    assert h.orch.stats()["in_flight"] == 0

    # Capacity survived the whole ladder: a refused upload must not be billed for a slot.
    record = await h.settle((await h.submit()).id)
    assert record.status == "succeeded"


# ------------------------------------------------------------------------ cache
def cached_payload(*, statuses: dict[str, str] | None = None,
                   overlay_urls: dict[str, str] | None = None) -> dict[str, Any]:
    """A minimal finished-result body for a hand-seeded cache entry."""
    views = {}
    for name in REGISTRY:
        views[name] = {"name": name, "status": (statuses or {}).get(name, "ok"),
                       "overlay_url": (overlay_urls or {}).get(name)}
    return {
        "verdict": "inconclusive", "ai_score": 0.1, "edited_score": 0.1, "overall": 0.1,
        "reliability": 0.5, "reasons": ["seeded"], "techniques": views,
        "metadata": {}, "c2pa": {}, "sha256": "a" * 64,
        "timings": {"total_ms": 1, "per_technique_ms": {"ela": 12}},
        "image": {"mime": "image/jpeg", "size_bytes": 512, "width": 64, "height": 64},
        "notes": [], "from_cache": False,
    }


async def test_the_cache_is_consulted_before_the_file_is_decoded(harness):
    """An undecodable body with a cached verdict is served, not rejected.

    The order matters for more than speed: the lookup is the only reason a repeat upload of a file the
    decoder is currently unhappy with does not turn into a 400 for something already analysed.
    """
    h = await harness()
    data = b"%PDF-1.4\n" + b"%" * 600
    h.cache.put(sha256_bytes(data), cached_payload(), {}, overlay_deadline=time.time() + 300)
    record = await h.submit(data)
    assert record.from_cache is True
    assert record.status == "succeeded"
    assert h.calls == []


async def test_a_cache_replay_is_fetchable_the_moment_its_id_is_returned(harness):
    """This save is awaited, unlike the progress publishes: the caller polls immediately."""
    h = await harness()
    data = tag(b"replay")
    h.cache.put(sha256_bytes(data), cached_payload(), {}, overlay_deadline=time.time() + 300)
    record = await h.submit(data)
    fetched = await h.orch.fetch(record.id)
    assert fetched is not None and fetched.status == "succeeded"
    assert fetched.result["from_cache"] is True


async def test_a_replay_mints_a_new_job_and_reuses_the_analysis(harness):
    """The job is new even when the analysis is not: two ids, one verdict, one body."""
    h = await harness()
    data = tag(b"twice")
    h.cache.put(sha256_bytes(data), cached_payload(), {}, overlay_deadline=time.time() + 300)
    first = await h.submit(data)
    second = await h.submit(data)
    assert first.id != second.id
    assert first.sha256 == second.sha256 == sha256_bytes(data)
    assert first.result["overall"] == second.result["overall"]
    assert h.orch.stats()["cache_hits"] == 2


async def test_replay_stages_describe_the_run_that_was_cached(harness):
    """A cached technique that errored must not come back as a green "done" in the stepper."""
    h = await harness()
    data = tag(b"stages")
    h.cache.put(sha256_bytes(data),
                cached_payload(statuses={"ela": "error", "noise": "skipped"}),
                {}, overlay_deadline=time.time() + 300)
    record = await h.submit(data)
    assert record.stages["ela"] == "failed"
    assert record.stages["noise"] == "skipped"
    assert record.stages["metadata"] == "done"
    assert record.progress == 1.0
    assert record.stage_ms == {"ela": 12}


async def test_expired_overlays_degrade_to_scores_plus_a_note(harness):
    """A stale cache entry answers "here is the verdict, the pictures expired" — never a 410."""
    h = await harness()
    data = tag(b"expired")
    payload = cached_payload(overlay_urls={"ela": "/api/overlays/oldjob/ela"})
    h.cache.put(sha256_bytes(data), payload, {"ela": "oldjob/ela"},
                overlay_deadline=time.time() - 1)
    record = await h.submit(data)
    assert record.result["techniques"]["ela"]["overlay_url"] is None
    notes = " ".join(record.result["notes"])
    assert "have expired" in notes
    assert "the original analysis, unchanged" in notes


async def test_a_partial_run_is_not_replayed_from_cache(harness):
    """``skipped`` is a function of load, so caching it pins a thin analysis to the hash."""
    h = await harness()
    data = tag(b"skipped")
    h.answers["ela"] = outcome("ela", None, status="skipped")
    await h.settle((await h.submit(data)).id)
    assert h.cache.get(sha256_bytes(data), h.signer) is None
    assert not (await h.submit(data)).from_cache


async def test_a_technique_that_errored_on_the_file_is_still_cacheable(harness):
    """An error caused by what is in the image repeats identically, so the next upload earns it."""
    h = await harness()
    data = tag(b"errored")
    h.answers["ela"] = outcome("ela", None, status="error")
    await h.settle((await h.submit(data)).id)
    hit = h.cache.get(sha256_bytes(data), h.signer)
    assert hit is not None
    assert hit.result["techniques"]["ela"]["status"] == "error"


async def test_a_run_that_measured_nothing_is_not_cached(harness):
    """Every technique failed: there is no analysis to reuse, and a replay would report success.

    ``_from_cache`` mints a ``succeeded`` record because a cached body always carries a verdict. When
    nothing was measured the original record said ``failed`` and carried an error explaining that the
    body is an accounting of what could not run — so a cache entry here would silently upgrade a
    failure into a success for every later upload of the same bytes.
    """
    h = await harness()
    data = tag(b"nothing")
    for name in REGISTRY:
        h.answers[name] = outcome(name, None, status="error")
    record = await h.settle((await h.submit(data)).id)
    assert record.status == "failed"
    assert record.error["code"] == "no_measurements"
    assert h.cache.get(sha256_bytes(data), h.signer) is None
    assert h.orch.stats()["cache_hits"] == 0


async def test_the_second_identical_upload_never_reaches_a_worker(harness):
    h = await harness()
    data = fixture_bytes("authentic")
    await h.settle((await h.submit(data)).id)
    calls = len(h.calls)
    assert (await h.submit(data)).from_cache is True
    assert len(h.calls) == calls


# --------------------------------------------------------------------- capacity
async def test_the_concurrency_cap_answers_busy_and_gives_the_slot_back(harness):
    h = await harness(max_concurrent_jobs=1)
    h.gate = threading.Event()
    first = await h.submit()

    with pytest.raises(BusyError) as exc:
        await h.submit(tag(b"second"))
    assert (exc.value.status, exc.value.code) == (503, "server_busy")
    assert exc.value.retry_after == 3

    # Released, not merely un-checked: the third job is the proof.
    h.gate.set()
    await h.settle(first.id)
    h.gate = None
    await h.settle((await h.submit(tag(b"third"))).id)
    assert h.orch.stats()["in_flight"] == 0


async def test_a_store_refusal_at_admission_returns_the_slot_and_the_run_entry(harness):
    """A server fault after the id was minted must not cost a unit of capacity forever."""
    h = await harness(max_concurrent_jobs=1)
    h.store.refuse_saves = True
    with pytest.raises(Rejected) as exc:
        await h.submit()
    assert (exc.value.status, exc.value.code) == (500, "internal")
    assert h.orch.stats()["in_flight"] == 0

    h.store.refuse_saves = False
    record = await h.settle((await h.submit(tag(b"again"))).id)
    assert record.status == "succeeded"


async def test_a_job_driven_before_startup_fails_without_leaking_capacity(settings, tmp_path,
                                                                          monkeypatch):
    """The precondition guard raises *inside* the driven task, so the ``finally`` still runs.

    A guard written before the ``try`` costs a permanent unit of capacity per mistimed request; this
    asserts the cheap-looking version is the one that actually holds the slot.
    """
    h = Harness(settings.model_copy(update={"max_concurrent_jobs": 1}), tmp_path, monkeypatch)
    record = await h.submit()
    settled = await h.wait(record.id)
    assert settled.status == "failed"
    assert settled.error["code"] == "RuntimeError"
    assert set(settled.stages.values()) == {"failed"}
    assert h.orch.stats()["in_flight"] == 0

    again = await h.submit(tag(b"second"))
    assert again.id != record.id


# -------------------------------------------------------------------- deadlines
async def test_the_hard_deadline_floor_still_produces_an_answer(harness):
    """``max(0.1, hard_s)``: a zero-second budget gives an honest partial job, not a crash."""
    h = await harness(job_hard_deadline_s=0.0)
    h.delays = {g: 0.6 for g in ("container", "native", "analysis", "learned")}
    record = await h.settle((await h.submit()).id)

    assert record.status == "failed"
    assert record.error["code"] == "no_measurements"
    assert record.result is not None
    assert "did not complete within the job deadline" in " ".join(record.result["notes"])
    for view in record.result["techniques"].values():
        assert view["score"] is None
        assert view["band"] == "not_measured"
    assert record.progress == 1.0


async def test_results_that_arrived_before_the_deadline_survive_it(harness):
    h = await harness(job_hard_deadline_s=0.4)
    h.delays = {"native": 2.0, "analysis": 2.0}
    record = await h.settle((await h.submit()).id)

    views = record.result["techniques"]
    assert record.status == "succeeded"
    assert views["metadata"]["measured"] is True
    assert views["ela"]["measured"] is False
    assert views["noise"]["measured"] is False
    assert record.result["reliability_detail"]["coverage"]["edited"] == 0
    assert "did not complete within the job deadline" in " ".join(record.result["notes"])


async def test_a_stage_that_settled_without_its_outcome_is_reported_as_lost(harness):
    """An event with no payload and a group return without the member is a gap, not a score.

    ``_finish`` could have invented 0.0 here, which reads as "this check ran and found nothing".
    """
    h = await harness()
    h.events = [("metadata", "done", False)]
    record = await h.settle((await h.submit()).id)

    assert record.stages["metadata"] == "done"
    view = record.result["techniques"]["metadata"]
    assert view["measured"] is False
    assert view["score"] is None
    assert record.result["metadata"]["status"] == "error"
    assert record.status == "succeeded"


# ---------------------------------------------------------------- progress truth
async def test_live_progress_reaches_the_store_while_the_job_runs(harness):
    h = await harness()
    h.gate = threading.Event()

    def emit_first(request: worker.GroupRequest) -> None:
        name = request.names[0]
        worker._emit(request.job_id, name, "running", 0, None)
        worker._emit(request.job_id, name, "done", 5, h._default_answer(name))

    h.on_group = emit_first
    record = await h.submit()

    mid = await h.wait(record.id, predicate=lambda r: any(s == "done" for s in r.stages.values()))
    assert mid.status == "running"
    assert 0.0 < mid.progress < 1.0
    assert "pending" in mid.stages.values()

    h.gate.set()
    final = await h.settle(record.id)
    assert final.progress == 1.0
    assert set(final.stages.values()) <= {"done", "failed", "skipped", "blocked"}


async def test_progress_publishes_are_debounced(harness):
    """Ten stage changes in one instant is one snapshot, not ten writes."""
    h = await harness()
    h.gate = threading.Event()

    def emit_running(request: worker.GroupRequest) -> None:
        for name in request.names:
            worker._emit(request.job_id, name, "running", 0, None)

    h.on_group = emit_running
    record = await h.submit()
    await asyncio.sleep(0.05)
    h.gate.set()
    await h.settle(record.id)

    writes = [p for p in h.store.publishes if p[0] == record.id]
    assert len(writes) <= 8, writes
    assert len(writes) < len(REGISTRY) + 3


async def test_an_event_for_a_finished_job_is_dropped(harness):
    """A late result must not resurrect a record nobody is waiting on."""
    h = await harness()
    record = await h.settle((await h.submit()).id)
    writes = len(h.store.publishes)

    h.orch._apply_event(worker.ProgressEvent(job_id=record.id, name="ela", status="running", ms=0))
    await asyncio.sleep(0.02)

    assert len(h.store.publishes) == writes
    assert (await h.orch.fetch(record.id)).stages["ela"] == "done"


# ------------------------------------------------------------------- plumbing
async def test_stats_describe_the_pools_that_are_actually_running(harness):
    """``pool_workers`` is shown on /api/health; naming the configured value instead of the real
    ``ThreadPoolExecutor`` size would tell operators about workers that do not exist."""
    h = await harness()
    stats = h.orch.stats()
    assert stats["pool_mode"] == "in-process threads"
    assert stats["pool_workers"] == _INPROC_GROUPS
    assert stats["jobs_started"] == 0


class _DeadPool:
    """A process pool whose children are gone. ``submit`` raises, and keeps raising: that is what
    ``BrokenProcessPool`` means, and it is why a service that does not rebuild becomes a machine for
    turning every later upload into a failure."""

    def __init__(self) -> None:
        self.submits = 0

    def submit(self, fn, *args):  # noqa: ANN001 - mirrors the executor signature, not a typed API
        self.submits += 1
        raise BrokenProcessPool("A child process terminated abruptly")

    def shutdown(self, **kwargs):  # noqa: ANN003
        pass


async def test_a_dead_pool_is_replaced_and_the_job_that_found_it_still_finishes(harness, monkeypatch):
    """A child that dies must cost one rebuild, not the deployment.

    ``BrokenProcessPool`` is sticky: every later ``submit`` on that executor raises without touching a
    worker, so without this path the first native segfault in a decoder turns the service into a
    machine that fails every upload until somebody restarts it.
    """
    h = await harness()
    live = h.orch._groups
    h.orch.settings = h.orch.settings.model_copy(update={"analyzer_parallel": True})
    h.orch._pool, h.orch._groups = _DeadPool(), None
    rebuilds: list[dict[str, Any]] = []

    def fake_new_pool(config):  # noqa: ANN001 - the test installs the in-process executor as "healthy"
        rebuilds.append(config)
        h.orch._pool = live

    monkeypatch.setattr(h.orch, "_new_pool", fake_new_pool)
    finished = await h.settle((await h.submit()).id)

    assert finished.status == "succeeded", finished.error
    assert len(rebuilds) == 1
    assert h.orch.stats()["pool_rebuilds"] == 1
    # The retry submits only what the first attempt did not: a job that ran the learned group twice
    # would pay for two inferences and write one overlay from whichever finished last.
    requested = [name for call in h.calls for name in call.names]
    assert sorted(requested) == sorted(REGISTRY)


async def test_the_rebuild_is_cooldown_limited_so_a_poisoned_file_cannot_fork_storm(harness, monkeypatch):
    h = await harness()
    live = h.orch._groups
    h.orch.settings = h.orch.settings.model_copy(update={"analyzer_parallel": True})
    monkeypatch.setattr(h.orch, "_new_pool", lambda config: setattr(h.orch, "_pool", live))

    h.orch._pool, h.orch._groups = _DeadPool(), None
    first = await h.settle((await h.submit()).id)
    assert first.status == "succeeded", first.error

    h.orch._pool, h.orch._groups = _DeadPool(), None
    # A different SHA-256: the identical bytes would be answered from the result cache and never
    # reach the pool, which would prove nothing about the cooldown.
    second = await h.settle((await h.submit(tag(b"second"))).id)
    assert second.status == "failed"
    assert second.error["code"] == "BrokenProcessPool"
    assert h.orch.stats()["pool_rebuilds"] == 1


async def test_every_registered_technique_is_asked_for_exactly_once(harness):
    h = await harness()
    await h.settle((await h.submit()).id)
    requested = [name for call in h.calls for name in call.names]
    assert set(requested) == set(REGISTRY)
    assert len(requested) == len(REGISTRY)
    for call in h.calls:
        assert call.sink.root == str(h.scratch.root)
        assert call.sink.job_id == call.job_id
        assert call.arrays is not None


async def test_sweep_removes_expired_records_and_their_scratch_trees(harness):
    h = await harness(job_ttl_s=60, overlay_ttl_s=60)
    record = await h.settle((await h.submit()).id)
    # Stand in for a finished analysis that wrote overlays.
    h.scratch.ensure_job_dir(record.id)
    (h.overlay_dir(record.id) / "ela.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    h.age_out(record.id)

    records, dirs = await h.orch.sweep()
    assert records == 1
    assert dirs == 1
    assert await h.orch.fetch(record.id) is None
    assert not h.overlay_dir(record.id).exists()


async def test_sweep_leaves_a_live_job_alone(harness):
    h = await harness(job_ttl_s=600, overlay_ttl_s=600)
    record = await h.settle((await h.submit()).id)
    h.scratch.ensure_job_dir(record.id)
    assert await h.orch.sweep() == (0, 0)
    assert (await h.orch.fetch(record.id)).status == "succeeded"
    assert h.overlay_dir(record.id).exists()


# ------------------------------------------------------------------ the real thing
@pytest.mark.slow
async def test_one_real_run_end_to_end(tmp_path, settings, monkeypatch):
    """No stubbed worker: the in-process groups, the real analysers, the real assembly.

    Deliberately excludes the spawn pool (that is the ``pool`` test) so this can assert the envelope
    without a second interpreter, and deliberately runs the whole registry so "every registered
    technique is accounted for" is tested against the registry itself rather than a hand-written list.
    """
    h = Harness(settings.model_copy(update={"analyzer_parallel": False}),
                tmp_path, monkeypatch, stub_group=False, stub_runtime=False)
    await h.start()
    try:
        data = fixture_bytes("authentic")
        finished = await h.wait((await h.orch.create_job(data, client="test")).id, timeout_s=120.0)
        assert finished.status == "succeeded", finished.error
        result = finished.result

        assert set(result["techniques"]) == set(REGISTRY)
        assert finished.progress == 1.0
        assert result["sha256"] == finished.sha256
        assert 0.0 <= result["overall"] <= 1.0
        assert 0.0 <= result["reliability"] <= 1.0
        assert result["reasons"], "a verdict with no explanation is not a verdict"
        assert result["limitations"], "the honesty block must never be empty"
        assert result["from_cache"] is False

        for view in result["techniques"].values():
            if view["measured"]:
                assert view["score"] is not None and 0.0 <= view["score"] <= 1.0
            else:
                assert view["score"] is None
                assert view["band"] == "not_measured"
            if view["overlay_url"]:
                assert view["overlay_url"].startswith("/api/overlays/")
                assert h.signer.check_url(view["overlay_url"])

        assert result["fusion"]["source"] in ("expert", "learned")
        assert result["provenance"]["app"] == settings.app_name
        assert set(result["timings"]["per_technique_ms"]) == set(REGISTRY)
        assert (await h.orch.create_job(data, client="test")).from_cache is True
    finally:
        await h.orch.shutdown()


@pytest.mark.pool
async def test_the_spawn_pool_delivers_results_and_overlay_files(tmp_path, settings, monkeypatch):
    """The real ``ProcessPoolExecutor``: spawn, ``init_child``, pickling and the queue reader.

    Nothing else in the suite touches a child process, so every defect in the IPC contract — an
    unpicklable request, a child that never received the config, a queue the reader thread abandoned —
    is invisible until this test runs.
    """
    h = Harness(settings.model_copy(update={"analyzer_parallel": True, "pool_max_workers": 2}),
                tmp_path, monkeypatch, stub_group=False, stub_runtime=False)
    await h.start()
    try:
        record = await h.orch.create_job(fixture_bytes("spliced"), client="test")
        finished = await h.wait(record.id, timeout_s=180.0)
        assert finished.status == "succeeded", finished.error

        views = finished.result["techniques"]
        ran = [n for n, v in views.items() if v["status"] == "ok"]
        assert len(ran) >= len(REGISTRY) - 2, views
        # The learned detector belongs to the parent's torch thread; a misrouted one reports the
        # routing bug rather than a score, and that is the failure this test exists to catch.
        assert views["cnn"].get("error_code") != "misrouted"

        overlays = [v["overlay_url"] for v in views.values() if v["overlay_url"]]
        assert overlays, "a real run must produce at least one overlay"
        for url in overlays:
            assert h.signer.check_url(url)
        assert list(h.overlay_dir(record.id).glob("*.png"))
    finally:
        await h.orch.shutdown()
