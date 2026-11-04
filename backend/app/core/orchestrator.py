"""The orchestrator: turns uploaded bytes into a finished job record.

**One writer, and it is the event loop.** Pool children cannot touch a ``JobRecord`` — they are
another process, and a record is not shared memory. Progress and results reach the parent through a
``multiprocessing.Queue`` (children) or a direct callback (the parent-side torch thread), and both are
marshalled onto the event loop before they are applied. So every mutation of a record happens on one
thread, in the order the loop dequeues it, and no lock is needed anywhere in this module. The
terminal-sticky rule in ``JobRecord.set_stage`` is still load-bearing: single-threaded does not mean
single-authored, and a technique's "done" and the watchdog's "deadline" really do arrive as two
independent posts about the same stage.

**Publishing is a stream of snapshots, not a callback per write.** See ``core/store/base.py``. A
debounced publish bounds redis traffic during a fast job, while a forced publish on every terminal
transition guarantees the client never waits longer than one poll to learn a check finished.

**Deadlines bound the wait, not the work.** ``JOB_SOFT_DEADLINE_S`` is handed to the workers, who
check it before each technique and skip what they cannot start — that genuinely saves CPU.
``JOB_HARD_DEADLINE_S`` only stops the *parent* from waiting: a CPU-bound analyser inside a
``ProcessPoolExecutor`` cannot be preempted, and cancelling a running future does nothing. What the
hard deadline buys is a partial answer delivered on time, because the outcomes already posted through
the queue survive the timeout. "We cannot stop the work, only the waiting" is the accurate version of
the claim, and it is why the soft deadline is the one that protects the box.

**A cache hit never enters a worker.** The whole analysis is a deterministic function of the file's
bytes, so a repeat upload gets the stored result and freshly signed overlay URLs, and the job is
``succeeded`` before the client's first poll. See ``core/cache.py`` for why that is safe with
transient overlays.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any

from app.analysis import worker
from app.analysis.adapt import build_technique_view
from app.analysis.explainers import KNOWN_LIMITATIONS, band, explain
from app.analysis.fusion import fuse
from app.analysis.registry import TECHNIQUES
from app.analysis.registry import names as technique_names
from app.analysis.reliability import ReliabilityContext, compute_reliability
from app.analysis.types import JobArrays, JobContext, SinkConfig, TaskOutcome
from app.analysis.verdict import decide
from app.config import Settings, get_settings
from app.core.cache import ResultCache
from app.core.job import JobRecord, new_job_id
from app.core.pipeline import build_arrays, sha256_bytes
from app.core.scratch import Scratch
from app.core.security import SecurityError, probe_header, sanitise_error
from app.core.signing import OverlaySigner
from app.core.store import JobStore
from app.schemas.analysis import (
    AnalysisResult,
    C2paBlock,
    EvidenceFlagView,
    MetadataBlock,
    ReliabilityBlock,
    Timings,
)

__all__ = ["BusyError", "Orchestrator", "Rejected", "worker_config"]

log = logging.getLogger("truelens.orchestrator")

#: Snapshots are published at most this often while a job runs; terminal transitions bypass it.
PUBLISH_INTERVAL_S = 0.12
#: How long the reader thread waits for an event before checking the shutdown flag.
_QUEUE_POLL_S = 0.1

#: Stage states that mean "nothing more will happen to this technique".
_SETTLED = ("done", "failed", "skipped", "blocked")

#: How many groups run at once when the process pool is switched off. A named constant because
#: ``stats()`` reports it to the health endpoint, and a second copy of the literal would be a number
#: the operator is shown that nobody maintains.
_INPROC_GROUPS = 3

#: Minimum wall seconds between two rebuilds of a dead process pool. A child that dies on its own
#: (an OpenCV or Pillow segfault on one pathological file) must cost one rebuild, not a fork storm.
_POOL_REBUILD_COOLDOWN_S = 10.0


def worker_config(settings: Settings) -> dict[str, Any]:
    """The ``FromConfig`` namespace the registry resolves against.

    Names here are the contract with ``registry.py``; a rename on one side makes a technique fail
    with "not configured", which is loud enough to catch in a test.

    Module-level rather than a private method because the in-process smokes and the tests bind the
    same techniques without an orchestrator. Duplicating the dict there let one of them drift: a
    hand-written config missing ``copy_move_features`` made ``copy_move`` construct its SIFT
    detector with ``None`` keypoints, raise, and report a *failed check* on a fixture built to
    exercise it.
    """
    return {
        "redact_pii": settings.redact_pii,
        "face_model_path": settings.yunet_model_path,
        "copy_move_features": settings.copy_move_features,
        "cnn_tta": settings.cnn_tta,
        "cnn_weights_path": settings.cnn_weights_path,
        "clip_probe_path": settings.clip_probe_path,
        "gradcam_min_ai_score": settings.gradcam_min_ai_score,
        "clip_probe_enabled": settings.enable_clip_probe,
    }


class Rejected(Exception):
    """The request cannot be served. Carries the HTTP shape so routes stay thin."""

    def __init__(self, status: int, code: str, message: str, *,
                 detail: str | None = None, retry_after: int | None = None,
                 extra: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail
        self.retry_after = retry_after
        self.extra = extra or {}


class BusyError(Rejected):
    """Every analysis slot is taken. A distinct type because the route's answer differs: retry."""

    def __init__(self, *, retry_after: int = 3) -> None:
        super().__init__(
            503, "server_busy",
            "The server is at its concurrent-analysis limit right now.",
            detail="Try again in a few seconds, or use a smaller image.",
            retry_after=retry_after)


@dataclass(slots=True)
class _Run:
    """In-flight job state, living only in the process that owns the job."""

    record: JobRecord
    arrays: JobArrays | None
    context: JobContext | None
    outcomes: dict[str, TaskOutcome] = field(default_factory=dict)
    submitted_at: float = 0.0
    decode_ms: int = 0
    queue_ms: int = 0
    last_publish: float = 0.0

    @property
    def job_id(self) -> str:
        return self.record.id


class Orchestrator:
    """Owns the pools, the per-job state, and the path from bytes to verdict."""

    def __init__(self, settings: Settings | None, *, store: JobStore, cache: ResultCache,
                 signer: OverlaySigner, scratch: Scratch) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.cache = cache
        self.signer = signer
        self.scratch = scratch
        self._runs: dict[str, _Run] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._groups: ThreadPoolExecutor | None = None
        self._torch: ThreadPoolExecutor | None = None
        self._queue: Any = None
        self._reader: threading.Thread | None = None
        self._stopping = threading.Event()
        self._sem = asyncio.Semaphore(max(1, self.settings.max_concurrent_jobs))
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._started_at = time.time()
        self._jobs_started = 0
        self._cache_hits = 0
        self._pool_rebuilds = 0
        self._pool_rebuilt_at = 0.0

    # ------------------------------------------------------------------ lifecycle
    async def startup(self) -> None:
        """Create the pools and load the torch runtime *before* the first request.

        Warming torch here rather than lazily is not an optimisation: the first job would otherwise
        pay a two-second model load inside its own deadline budget and report a timeout that is
        really a cold start.
        """
        s = self.settings
        self._loop = asyncio.get_running_loop()
        config = self._worker_config()
        # The parent needs the same configuration the children get, and its own event route, whether
        # or not a pool exists: the ``learned`` group always runs here, on the torch thread, and an
        # unconfigured parent has neither the ``FromConfig`` values the CNN binds against nor a way to
        # report progress. Without this the CNN's stage stays pending forever and ``_finish`` writes
        # a deadline timeout over a technique that actually ran in 7ms.
        worker.configure_parent(config, max(1, s.torch_threads), self._post_event)

        if s.analyzer_parallel:
            self._new_pool(config)
            log.info("analysis pool: spawn x%d", s.pool_max_workers)
        else:
            # No children: groups run on threads in this process. Saves the spawn cost and the IPC on
            # a 1-vCPU box, at the price of sharing one GIL with the event loop.
            self._groups = ThreadPoolExecutor(max_workers=_INPROC_GROUPS,
                                              thread_name_prefix="truelens-grp")
            log.info("analysis pool: disabled, groups run in-process")

        # One thread, forever: the torch runtime is a module-level singleton and wants a single
        # owning thread as much as it wants a single load.
        self._torch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="truelens-torch")
        self._warm_models()

    def _new_pool(self, config: dict[str, Any]) -> None:
        """One process pool plus the queue and reader it reports progress through."""
        s = self.settings
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._pool = ProcessPoolExecutor(
            max_workers=max(1, s.pool_max_workers), mp_context=ctx,
            initializer=worker.init_child,
            initargs=(self._queue, config, max(1, s.torch_threads)))
        self._reader = threading.Thread(
            target=self._drain_queue, name="truelens-events", daemon=True)
        self._reader.start()

    def _rebuild_pool(self) -> bool:
        """Replace a pool whose children died, so one crash does not end the deployment.

        ``BrokenProcessPool`` is permanent for an executor: every later ``submit`` raises the same
        error without touching a child, so a single native segfault in a decoder leaves a service
        that answers requests with failures until someone restarts it. The cooldown is the other
        half of the answer — a file that kills every child on the way in gets one rebuild per
        window rather than an unbounded respawn of spawn-costly interpreters.
        """
        if not self.settings.analyzer_parallel or self._stopping.is_set():
            return False
        now = time.time()
        left = _POOL_REBUILD_COOLDOWN_S - (now - self._pool_rebuilt_at)
        if left > 0:
            log.warning("analysis pool is unusable and still cooling down (%.0fs left)", left)
            return False
        old_pool, old_queue = self._pool, self._queue
        self._pool = None
        if old_pool is not None:
            try:
                old_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # pragma: no cover - a broken pool can raise from its own teardown
                log.debug("broken pool teardown raised", exc_info=True)
        if old_queue is not None:
            # Without this the old reader blocks on a queue nobody will ever write to again; closing
            # it makes the reader's own ``get`` raise and the thread return.
            try:
                old_queue.cancel_join_thread()
                old_queue.close()
            except Exception:  # pragma: no cover - best-effort teardown of a dying resource
                log.debug("queue close raised", exc_info=True)
        try:
            self._new_pool(self._worker_config())
        except Exception:
            log.exception("could not rebuild the analysis pool")
            self._pool = None
            return False
        self._pool_rebuilt_at = now
        self._pool_rebuilds += 1
        log.warning("analysis pool rebuilt after a child died (rebuild #%d)", self._pool_rebuilds)
        return True

    async def shutdown(self) -> None:
        self._stopping.set()
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        for pool in (self._pool, self._groups, self._torch):
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        if self._queue is not None:
            try:
                self._queue.cancel_join_thread()
                self._queue.close()
            except Exception:
                log.debug("queue close raised", exc_info=True)
        self._runs.clear()

    def _worker_config(self) -> dict[str, Any]:
        return worker_config(self.settings)

    def _warm_models(self) -> None:
        """Load the learned detector once, on its own thread, before serving traffic.

        ``load()`` is what this is really for: it is the first thing in the process that imports
        torch and torchvision, and on a cold container that import costs several seconds. Doing it
        here rather than inside the first job keeps that job's deadline budget intact, and doing it
        on the torch thread means the import lands on the one thread that will ever run the model.
        """
        if self._torch is None:
            return

        def warm() -> None:
            try:
                from app.ml.cnn_runtime import get_runtime

                runtime = get_runtime(self.settings.cnn_weights_path,
                                      torch_threads=max(1, self.settings.torch_threads))
                available = runtime.load()
                log.info("cnn runtime available=%s%s", available,
                         "" if available else f" ({runtime.load_error})")
                if available:
                    log.info("cnn warm-up: %s", runtime.warmup())
            except Exception as exc:
                log.warning("cnn warm-up failed: %s", sanitise_error(exc))

        self._torch.submit(warm)

    # ------------------------------------------------------------------ submission
    async def create_job(self, data: bytes, *, client: str = "", label: str = "") -> JobRecord:
        """Validate, decode, and either serve from cache or start a job.

        Raises ``Rejected`` for anything the client can fix, and ``BusyError`` for capacity. Both
        happen *before* a job id exists, so a rejected upload leaves no record and no files — except
        the store refusing the new record, which is a server fault reported after the id was minted.
        """
        s = self.settings
        if not data:
            raise Rejected(400, "invalid_image", "The upload was empty.")
        if len(data) > s.max_upload_bytes:
            raise Rejected(
                413, "too_large", "That image is larger than the server accepts.",
                detail=f"{len(data):,} bytes uploaded; the limit is {s.max_upload_bytes:,} bytes.",
                extra={"max_bytes": s.max_upload_bytes})

        sha = sha256_bytes(data)
        hit = self.cache.get(sha, self.signer)
        if hit is not None:
            return await self._from_cache(sha, hit, client=client)

        t0 = time.perf_counter()
        try:
            probe = probe_header(data, max_pixels=s.max_analysis_pixels)
            arrays = await asyncio.to_thread(
                build_arrays, data, probe,
                analysis_max_dim=s.analysis_max_dim, native_max_dim=s.native_max_dim)
        except SecurityError as exc:
            # These messages are written for display: they name the limit or the format, never a path.
            raise Rejected(exc.status_code, exc.code, exc.message) from exc
        except Exception as exc:
            log.info("decode failed for %s: %s", sha[:12], sanitise_error(exc))
            raise Rejected(400, "decode_failed", "This file could not be decoded as an image.",
                           detail=sanitise_error(exc)) from exc
        decode_ms = int((time.perf_counter() - t0) * 1000)

        if self._sem.locked():
            raise BusyError()
        t1 = time.perf_counter()
        await self._sem.acquire()
        queue_ms = int((time.perf_counter() - t1) * 1000)

        record = JobRecord(id=new_job_id(), sha256=sha, mime=arrays.mime,
                           size_bytes=arrays.file_size, status="queued", client=client)
        record.init_stages(list(technique_names()))
        run = _Run(record=record, arrays=arrays, context=JobContext.from_arrays(arrays),
                   decode_ms=decode_ms, queue_ms=queue_ms, submitted_at=time.monotonic())
        self._runs[record.id] = run
        try:
            await self.store.save(record)
        except Exception as exc:
            # Admission holds two things that outlive the job's own task: a semaphore slot and an entry
            # in the runs table. If the store refuses the record they must be returned now, or every
            # later request inherits the leak and the box slowly runs out of capacity.
            log.error("store refused the queued record for job %s: %s",
                      record.id, sanitise_error(exc))
            self._runs.pop(record.id, None)
            self._sem.release()
            raise Rejected(500, "internal", "The server could not record this job.",
                           detail="Try again; if it persists, the job store is unreachable.") from exc
        self._jobs_started += 1
        self._spawn(self._drive(run))
        log.info("job %s queued (%s, %d bytes, decode %dms, label=%r)",
                 record.id, arrays.mime, arrays.file_size, decode_ms, _safe_label(label))
        return record

    async def _from_cache(self, sha: str, hit: Any, *, client: str) -> JobRecord:
        """A finished record for a cached image, without touching a worker.

        The record is minted fresh — new id, new timestamps — because the *job* is new even when the
        analysis is not. Only ``result`` is reused, and the cache already re-signed its overlay URLs
        against the current clock.

        The save is awaited, unlike the progress publishes elsewhere: this method returns the job id
        the client is about to poll, so a record that exists only in a pending task is a 404 for a
        job we just told the caller about.
        """
        payload = dict(hit.result)
        payload["from_cache"] = True
        if hit.expired_overlays:
            notes = list(payload.get("notes") or [])
            notes.append(
                "Overlay images for "
                + ", ".join(sorted(hit.expired_overlays))
                + " have expired and are no longer downloadable; the scores and explanations are "
                "the original analysis, unchanged.")
            payload["notes"] = notes
        image = payload.get("image") or {}
        record = JobRecord(
            id=new_job_id(), sha256=sha, mime=str(image.get("mime") or "application/octet-stream"),
            size_bytes=int(image.get("size_bytes") or 0), status="succeeded", client=client,
            from_cache=True)
        record.init_stages(list(technique_names()))
        # A replay's stages describe the analysis that was cached, not a fresh run of every check. An
        # errored technique is cacheable on purpose (it fails for what is in the file), and a record
        # that calls it "done" while its own result body says "error" is the contradiction the progress
        # stepper would show the user.
        views = payload.get("techniques") or {}
        for name in record.stages:
            status = str((views.get(name) or {}).get("status") or "ok")
            record.stages[name] = ("done" if status == "ok"
                                   else "skipped" if status == "skipped" else "failed")
        now = time.time()
        record.started_at = now
        record.finished_at = now
        record.result = payload
        record.stage_ms = dict(payload.get("timings", {}).get("per_technique_ms") or {})
        self._cache_hits += 1
        await self.store.save(record)
        log.info("job %s served from cache (%s)", record.id, sha[:12])
        return record

    # ------------------------------------------------------------------------ jobs
    async def fetch(self, job_id: str) -> JobRecord | None:
        return await self.store.get(job_id)

    async def sweep(self) -> tuple[int, int]:
        """Drop expired jobs and their overlays. Returns ``(records, directories)``.

        Both halves are needed: a record outliving its scratch directory hands out overlay URLs that
        404, and a directory outliving its record is disk nobody can account for. The TTLs are ordered
        so neither happens in the normal case — this is the belt, ``validate_prod`` is the braces.
        """
        removed_records = await self.store.sweep(self.settings.job_ttl_s)
        removed_dirs = self.scratch.sweep_expired(self.settings.overlay_ttl_s)
        if removed_dirs:
            log.info("removed %d scratch directory(s)", removed_dirs)
        return removed_records, removed_dirs

    @property
    def uptime_s(self) -> float:
        """Wall seconds since the pools were created. Public because /api/health reports it and a
        route has no business reaching into ``_started_at``."""
        return round(time.time() - self._started_at, 1)

    def stats(self) -> dict[str, Any]:
        return {
            "jobs_started": self._jobs_started,
            "cache_hits": self._cache_hits,
            "in_flight": len(self._runs),
            "max_concurrent_jobs": self.settings.max_concurrent_jobs,
            "pool_mode": "spawn" if self._pool is not None else "in-process threads",
            "pool_rebuilds": self._pool_rebuilds,
            "pool_workers": (self.settings.pool_max_workers if self._pool is not None
                             else (_INPROC_GROUPS if self._groups else 0)),
            "threads_per_worker": max(1, self.settings.torch_threads),
            "soft_deadline_s": self.settings.job_soft_deadline_s,
            "hard_deadline_s": self.settings.job_hard_deadline_s,
        }

    # ------------------------------------------------------------------ the drive
    def _spawn(self, coro: Any, *, what: str = "job") -> None:
        """Start a background task and keep a strong reference to it.

        ``asyncio`` holds only weak references to running tasks; without this set a job's driving
        coroutine can be garbage-collected mid-flight, which presents as an analysis that silently
        never finishes.
        """
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda t: self._note_task_failure(t, what))

    def _note_task_failure(self, task: asyncio.Task[Any], what: str) -> None:
        """Log a fire-and-forget task's failure instead of leaving it unobserved.

        ``_drive`` handles its own errors, but a progress publish does not — and a store write that
        raised inside a spawned task would otherwise surface only as asyncio's
        "Task exception was never retrieved", which is not something a log shipper can act on.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("%s failed: %s", what, sanitise_error(exc))

    async def _drive(self, run: _Run) -> None:
        """Submit the groups, wait out the deadline, assemble the result. Never raises."""
        record = run.record
        s = self.settings
        try:
            # Inside the try, not before it: a guard that raises from here skips the ``finally``, and
            # the job's semaphore slot and runs-table entry are never given back. "Used before
            # startup()" then costs a permanent unit of capacity instead of failing one job.
            if (self._pool or self._groups) is None or self._torch is None:
                raise RuntimeError("orchestrator used before startup()")
            record.status = "running"
            record.started_at = time.time()
            await self.store.save(record)

            sink = SinkConfig(
                root=str(self.scratch.root), job_id=record.id, max_dim=s.overlay_max_dim,
                expires_at=time.time() + s.overlay_ttl_s)
            soft_at = run.submitted_at + s.job_soft_deadline_s

            futures: dict[asyncio.Future[worker.GroupResult], str] = {}
            try:
                self._submit_groups(run, sink, soft_at, futures)
            except BrokenProcessPool:
                # Exactly one attempt at recovery, then the job fails like any other failed job and
                # the next one finds a live pool.
                if not self._rebuild_pool():
                    raise
                self._submit_groups(run, sink, soft_at, futures)

            # Release the pixels: they are now either pickled into a child or referenced by the
            # request the torch thread holds. Everything the verdict needs is in ``run.context``.
            run.arrays = None
            await self._await_groups(run, futures, s.job_hard_deadline_s)
            await self._finish(run, sink)
        except Exception as exc:
            log.exception("job %s crashed", record.id)
            record.status = "failed"
            record.error = {"code": type(exc).__name__, "message": sanitise_error(exc)}
            for name in list(record.stages):
                record.set_stage(name, "failed")
            try:
                await self.store.save(record)
            except Exception:
                log.exception("could not publish the failure for job %s", record.id)
        finally:
            self._runs.pop(record.id, None)
            self._sem.release()

    def _submit_groups(
        self,
        run: _Run,
        sink: SinkConfig,
        soft_at: float,
        futures: dict[asyncio.Future[Any], str],
    ) -> None:
        """Submit every group not already in ``futures``, mutating that dict in place.

        In place, rather than returning a fresh mapping, because the caller may retry after a rebuild
        and a retry that re-submits the whole set would run the learned group twice for one job.
        """
        pool = self._pool or self._groups
        if pool is None or self._torch is None:
            raise RuntimeError("orchestrator used before startup()")
        record = run.record
        for gname, members in worker.groups_for(list(technique_names())).items():
            if gname in futures:
                continue
            request = worker.GroupRequest(
                job_id=record.id, names=tuple(members),
                arrays=worker.trim_arrays(run.arrays, tuple(members)),
                sink=sink, deadline_at=soft_at)
            # The learned group runs here rather than in the pool: torch is loaded once, in the
            # parent, on a single thread. Every other group goes to a child process.
            executor = self._torch if gname == "learned" else pool
            futures[asyncio.wrap_future(executor.submit(worker.run_group, request))] = gname

    async def _await_groups(
        self, run: _Run, futures: dict[asyncio.Future[Any], str], hard_s: float
    ) -> None:
        """Wait for groups until the hard deadline, then keep whatever already arrived.

        ``return_when=ALL_COMPLETED`` is the shape that answers the question actually being asked,
        which is "has every group finished", not "has any". On timeout the unfinished futures are
        cancelled: that stops *queued* work and abandons *running* work, whose results are dropped.
        See the module docstring for what a deadline can and cannot bound.
        """
        pending = set(futures)
        deadline = time.monotonic() + max(0.1, hard_s)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(pending, timeout=remaining,
                                               return_when=asyncio.ALL_COMPLETED)
            for fut in done:
                self._absorb(run, fut, futures[fut])
        if pending:
            log.warning("job %s hit its %.1fs deadline with %d group(s) unfinished",
                        run.job_id, hard_s, len(pending))
        for fut in pending:
            fut.cancel()

    def _absorb(self, run: _Run, fut: asyncio.Future[Any], group: str) -> None:
        """Take a finished group's outcomes into the run, preferring what the queue already delivered.

        Outcomes arrive twice when all goes well — once on the queue, once in the return value. The
        queue copy wins because it is what the client already saw; the return value exists so that a
        dropped queue event (a full queue, ``put_nowait`` failing) does not silently cost a
        technique's result.
        """
        if fut.cancelled():
            return
        try:
            # No ``timeout`` argument: these are ``asyncio`` futures wrapped from the executor's own,
            # and the C implementation of ``asyncio.Future.result`` accepts none. Passing one raised
            # ``TypeError`` for every group, which this except swallowed — the queue copy of the
            # outcome kept the job looking healthy while the group-return safety net was dead.
            result = fut.result()
        except Exception as exc:
            log.warning("group %s failed for job %s: %s", group, run.job_id, sanitise_error(exc))
            return
        for outcome in result.outcomes:
            run.outcomes.setdefault(outcome.name, outcome)

    async def _finish(self, run: _Run, sink: SinkConfig) -> None:
        """Account for every technique, assemble the result, and store it."""
        record = run.record
        for name in technique_names():
            if name in run.outcomes:
                # A technique's stage normally settles from its queue event, but the event and the
                # outcome are two separate deliveries and the group can return first. When it does,
                # this run's last stage would otherwise be published as ``running`` on a job that is
                # already ``succeeded`` — a stepper spinner that never stops, and a progress figure
                # that never reaches 1.0. Settle it from the outcome that proves it finished.
                outcome = run.outcomes[name]
                if record.stages.get(name) not in _SETTLED:
                    settled = ("done" if outcome.status == "ok"
                               else "skipped" if outcome.status == "skipped" else "failed")
                    record.set_stage(name, settled, ms=int(outcome.ms))
                continue
            stage = record.stages.get(name, "pending")
            if stage in ("pending", "running", "blocked"):
                # Nothing arrived: the group never started, or was still running when the hard
                # deadline passed. The client is told "not measured", never "clean".
                outcome = TaskOutcome(
                    name=name, status="timeout", score=None,
                    details={"note": "This check did not finish before the job's deadline."},
                    error_code="deadline",
                    error_msg="the job deadline passed while this check ran")
                record.set_stage(name, "failed")
            elif stage == "skipped":
                outcome = TaskOutcome(
                    name=name, status="skipped", score=None,
                    details={"note": "Skipped: the deadline passed before this check started."},
                    error_code="deadline")
            else:
                # Its stage settled but no outcome came with it — a queue event without a payload and
                # a group return that did not include it. Report the gap rather than inventing a score.
                outcome = TaskOutcome(
                    name=name, status="error", score=None,
                    details={"note": "This check reported a result that did not reach the server."},
                    error_code="lost_result",
                    error_msg="the technique finished but its outcome was not delivered")
            run.outcomes[name] = outcome

        t0 = time.perf_counter()
        try:
            result = self._assemble(run, sink)
        except Exception as exc:
            log.exception("assembly failed for job %s", record.id)
            record.status = "failed"
            record.error = {"code": "assembly_failed", "message": sanitise_error(exc)}
            # Settled like every other terminal path: a record that says "failed" with no finish time
            # is one the sweeper cannot age and the UI renders as if it were still running.
            record.finished_at = time.time()
            await self.store.save(record)
            return
        assembly_ms = int((time.perf_counter() - t0) * 1000)
        # ``model_copy`` is how a frozen model is amended. The alternative — declaring Timings mutable
        # to save two lines — would let any code path rewrite a published result after validation.
        timings = result.timings.model_copy(update={"assembly_ms": assembly_ms})
        result = result.model_copy(update={"timings": timings})
        payload = result.model_dump(mode="json")

        measured = sum(1 for o in run.outcomes.values() if o.status == "ok")
        record.result = payload
        record.status = "succeeded" if measured else "failed"
        if not measured:
            record.error = {
                "code": "no_measurements",
                "message": "No technique produced a usable measurement for this file, so the "
                           "result below is an accounting of what could not run."}
        record.finished_at = time.time()
        await self.store.save(record)

        if measured and self._cacheable(run.outcomes):
            # ``measured`` is the gate the cache cannot see from the inside. A run where every
            # technique failed is deterministic — it will fail identically — but its record says
            # ``failed`` and carries an error explaining that the body is an accounting of what could
            # not run, and ``_from_cache`` mints a ``succeeded`` record with no error. Caching it
            # would silently upgrade a failure into a success for every later upload of these bytes.
            self.cache.put(
                record.sha256, payload,
                {name: f"{record.id}/{name}" for name, o in run.outcomes.items() if o.overlay},
                overlay_deadline=sink.expires_at)
        log.info("job %s %s verdict=%s overall=%.3f reliability=%.3f (%dms total, %dms assembly)",
                 record.id, record.status, result.verdict, result.overall, result.reliability,
                 result.timings.total_ms, assembly_ms)

    def _cacheable(self, outcomes: dict[str, TaskOutcome]) -> bool:
        """Only a complete, deterministic run is worth caching.

        A *skipped* technique is a function of load, not of the file: caching its result would pin a
        thin analysis to a hash that could support a full one, and the second user to upload the file
        would inherit the first user's busy minute. An *errored* technique is different — it failed
        because of what is in the file, so it will fail identically and belongs in the cache.
        """
        return all(o.status not in ("skipped", "timeout") for o in outcomes.values())

    # -------------------------------------------------------------------- assembly
    def _assemble(self, run: _Run, sink: SinkConfig) -> AnalysisResult:
        """Turn outcomes into a validated result: explanations, fusion, reliability, verdict."""
        outcomes = run.outcomes
        ctx = run.context
        explanations = {name: explain(name, o.score, o.details, status=o.status)
                        for name, o in outcomes.items()}
        fusion, _features = fuse(outcomes, ctx, explanations,
                                 model_path=self.settings.fusion_path or None)
        reliability = compute_reliability(ReliabilityContext(
            arrays=ctx, outcomes=outcomes, explanations=explanations,
            fusion_available=fusion.calibrated))
        verdict = decide(fusion, reliability, outcomes)

        views = {}
        for name, outcome in outcomes.items():
            url = self.signer.url(run.job_id, name) if outcome.overlay is not None else None
            views[name] = build_technique_view(outcome, explanations[name], overlay_url=url)

        total_ms = int((time.monotonic() - run.submitted_at) * 1000)
        timings = Timings(
            total_ms=total_ms, decode_ms=run.decode_ms, queue_ms=run.queue_ms,
            analysis_ms=max(0, total_ms - run.decode_ms - run.queue_ms),
            assembly_ms=0,
            per_technique_ms={n: int(o.ms) for n, o in outcomes.items()})

        md = _details_of(outcomes, "metadata")
        c2 = _details_of(outcomes, "c2pa")
        counts = {g: sum(1 for spec in TECHNIQUES if spec.group == g) for g in ("ai", "edited")}

        notes = list(fusion.notes)
        if any(o.status in ("timeout", "skipped") for o in outcomes.values()):
            notes.append(
                "One or more checks did not complete within the job deadline, so this verdict rests "
                "on fewer measurements than a full run would provide.")

        return AnalysisResult(
            verdict=verdict.verdict,
            ai_score=round(verdict.ai_score, 4),
            edited_score=round(verdict.edited_score, 4),
            overall=round(verdict.overall, 4),
            reliability=round(reliability.score, 4),
            reasons=list(verdict.reasons),
            techniques=views,
            metadata=MetadataBlock(
                status=_status_of(outcomes, "metadata"),
                present=bool(md.get("has_exif") or md.get("png_text_chunks")),
                exif=_dict_of(md, "exif"),
                software=md.get("software"),
                camera_make=md.get("camera_make"),
                camera_model=md.get("camera_model"),
                findings=list(md.get("findings") or []),
                generator_hits=list(md.get("generator_hits") or []),
                pii_redacted=bool(md.get("exif_pii_redacted", True)),
                deterministic_flag=md.get("deterministic_flag"),
                raw=md),
            c2pa=C2paBlock(
                status=_status_of(outcomes, "c2pa"),
                present=bool(c2.get("present")),
                credential=str(c2.get("status") or "unknown"),
                issuer=c2.get("issuer") or _dict_of(c2, "signing_info").get("issuer"),
                claim_generator=c2.get("claim_generator"),
                assertions=list(c2.get("assertions") or []),
                note=str(c2.get("note") or ""),
                raw=c2),
            sha256=run.record.sha256,
            timings=timings,
            band=band(verdict.overall),
            headline=verdict.headline,
            detail=verdict.detail,
            flags=[EvidenceFlagView.from_flag(f) for f in verdict.flags],
            reliability_detail=ReliabilityBlock(
                score=round(reliability.score, 4),
                reasons=list(reliability.reasons),
                coverage=dict(reliability.coverage),
                note=(
                    f"{reliability.coverage.get('ai', 0)} of {counts['ai']} AI checks and "
                    f"{reliability.coverage.get('edited', 0)} of {counts['edited']} manipulation "
                    "checks produced a measurement for this file."),
            ),
            fusion={
                "source": fusion.source,
                "calibrated": fusion.calibrated,
                "support": fusion.support,
                "adjusted": verdict.adjusted,
                "contributions": {k: round(float(v), 4) for k, v in fusion.contributions.items()
                                  if v > 0},
            },
            provenance=self._provenance(),
            limitations=list(KNOWN_LIMITATIONS),
            notes=notes,
            from_cache=False,
            image=self._image_block(run),
        )

    def _image_block(self, run: _Run) -> dict[str, Any]:
        """Facts about the file. Never the filename: that is user input of arbitrary content."""
        ctx = run.context
        if ctx is None:
            return {"size_bytes": run.record.size_bytes, "mime": run.record.mime}
        return {
            "width": ctx.orig_w,
            "height": ctx.orig_h,
            "mime": ctx.mime,
            "size_bytes": ctx.file_size,
            "est_jpeg_quality": ctx.est_jpeg_quality,
            "bpp": round(ctx.bpp, 4),
            "analysis_scale": round(ctx.analysis_scale, 4),
            "native_box": list(ctx.native_box),
            "round_dimensions": bool(ctx.round_dimensions),
        }

    def _provenance(self) -> dict[str, Any]:
        """Which artifacts actually took part, because "the model said so" is not auditable."""
        s = self.settings

        def present(path: str) -> str | None:
            return path if path and os.path.exists(path) else None

        return {
            "app": s.app_name,
            "environment": s.environment,
            "perf_profile": s.perf_profile,
            "fusion_artifact": present(s.fusion_path),
            "cnn_weights": present(s.cnn_weights_path),
            "clip_probe": present(s.clip_probe_path) if s.enable_clip_probe else None,
            "overlay_ttl_s": s.overlay_ttl_s,
        }

    # ------------------------------------------------------------------- event ingress
    def _post_event(self, event: worker.ProgressEvent) -> None:
        """Entry point for in-process producers (the torch thread)."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._apply_event, event)

    def _drain_queue(self) -> None:
        """Reader thread: hand child events to the loop, in order.

        A dedicated thread rather than a coroutine because ``mp.Queue.get`` blocks and there is no
        awaitable variant; polling ``get_nowait`` would either burn a core or add latency to every
        progress tick. The thread does nothing but move objects, so that every mutation it triggers
        happens back on the loop and the single-writer rule holds.
        """
        source = self._queue
        if source is None:  # pragma: no cover - the reader is only started when a queue exists
            return
        while not self._stopping.is_set():
            try:
                event = source.get(timeout=_QUEUE_POLL_S)
            except queue.Empty:
                continue
            except (OSError, EOFError, ValueError):
                # Queue closed or an unpicklable object. The reader dying must not kill the API.
                log.info("event queue reader stopping")
                return
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            try:
                loop.call_soon_threadsafe(self._apply_event, event)
            except RuntimeError:  # pragma: no cover - loop closed between the check and the post
                return

    def _apply_event(self, event: worker.ProgressEvent) -> None:
        """Apply one progress event. Runs on the loop, so ``_runs`` needs no lock."""
        run = self._runs.get(event.job_id)
        if run is None:
            # The job finished, failed or was swept while this was in flight. Dropping it is right:
            # a late result for a job nobody is waiting on must not resurrect its record.
            return
        if event.outcome is not None:
            run.outcomes.setdefault(event.outcome.name, event.outcome)
        if run.record.set_stage(event.name, event.status, ms=event.ms or None):
            self._maybe_publish(run)

    def _maybe_publish(self, run: _Run) -> None:
        """Debounce snapshots, but never when the job has just settled."""
        now = time.monotonic()
        settled = all(s in _SETTLED for s in run.record.stages.values())
        if not settled and now - run.last_publish < PUBLISH_INTERVAL_S:
            return
        run.last_publish = now
        self._spawn(self.store.save(run.record), what=f"progress publish for job {run.job_id}")


def _details_of(outcomes: dict[str, TaskOutcome], name: str) -> dict[str, Any]:
    outcome = outcomes.get(name)
    return dict(outcome.details or {}) if outcome else {}


def _status_of(outcomes: dict[str, TaskOutcome], name: str) -> str:
    outcome = outcomes.get(name)
    return outcome.status if outcome else "missing"


def _dict_of(details: dict[str, Any], key: str) -> dict[str, Any]:
    value = details.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _safe_label(label: str) -> str:
    """A client-supplied filename, made safe enough to log.

    Never echoed into a result, never used as a path, never persisted: it exists so a batch of
    uploads can be correlated in the log with the rows the user sees.
    """
    cleaned = "".join(c for c in str(label) if c.isprintable())[:64]
    return cleaned.replace("\n", " ").replace("\r", " ")
