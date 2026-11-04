"""Runs techniques: in a spawn child for the classical analysers, in the parent for torch.

**Why groups instead of one task per technique.** ``ProcessPoolExecutor`` pickles its arguments per
``submit``. A ``JobArrays`` is two resampled tracks plus the original container — tens of megabytes
on a large photograph. Submitting ten techniques would serialise that payload ten times over, and
the IPC would cost more than the analysis it feeds. So work is grouped by the resolution track its
members read: *container*, *native*, *analysis*. Each group crosses the pipe once, with the array
fields it does not use replaced by ``None`` so they do not travel at all. The grouping is derived
from the registry's ``track`` and ``heavy`` declarations, so adding a technique cannot desynchronise
it.

**Why progress is still per-technique.** A group reports each member's start and finish on a shared
``multiprocessing.Queue`` created in the parent and handed to workers through the pool's
``initargs`` — the one supported way to give a spawn child a live channel. The orchestrator drains
that queue, so the stepper advances while a group is mid-flight rather than in a burst at the end.

**Why the CNN is not in the pool.** Torch wants roughly 300 MB and a couple of seconds to import,
and it must be loaded *once*. A pool of N spawn children would either pay that N times or need
N copies resident for the process lifetime. The learned detector therefore runs in the parent, on a
one-thread pool of its own, while the classical analysers run in the children. ``run_group`` enforces
the split: a heavy technique that arrives in a child is refused rather than silently importing
torch there.

**Nothing here raises.** A technique that throws produces an ``Outcome`` with ``status="error"`` and a
sanitised one-line message, and the group carries on with the next member. The full traceback goes to
the log, which the operator sees; the client sees a class name and one line, because a traceback is a
listing of the machine's filesystem.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from app.analysis.registry import BY_NAME, Arg, FromArrays, FromConfig, TechniqueSpec
from app.analysis.types import JobArrays, OverlayRef, SinkConfig, TaskOutcome
from app.core.job import StageStatus

__all__ = [
    "GroupRequest",
    "GroupResult",
    "ProgressEvent",
    "configure_parent",
    "group_for",
    "groups_for",
    "init_child",
    "needed_fields",
    "run_group",
    "trim_arrays",
]

log = logging.getLogger("truelens.worker")

#: Populated by ``init_child`` in the spawn child. ``None`` in the parent, where progress travels by
#: direct callback instead of through the queue.
_PROGRESS_QUEUE: Any = None
#: Parent-side sink for the same events. Set by ``configure_parent``.
_PROGRESS_CALLBACK: Callable[[ProgressEvent], None] | None = None
#: Worker-side config, resolved from ``FromConfig`` bindings. Plain dict so the registry needs no
#: pydantic import and a child cannot drag ``pydantic-settings`` (and a .env read) into its closure.
_CONFIG: dict[str, Any] = {}
#: True only inside a pool child. Used to refuse heavy techniques there.
_IN_CHILD = False
#: cv2 spawns its own thread pool per process; the parent bounds the total by setting the env var
#: before import. Recorded here so ``/api/health`` can report what the workers actually got.
_THREADS = 0


# --------------------------------------------------------------------- payloads
@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One technique's stage change, emitted live from the worker.

    Terminal events carry the ``TaskOutcome`` itself. That makes the queue both the progress channel
    and the result channel, which is what lets a job that hits its hard deadline still report the
    techniques that had already finished: their outcomes reached the parent before the deadline did,
    whereas the pool future that would have returned them never completes. The ``GroupResult`` return
    value duplicates them as a fallback for when the queue is full and an event is dropped.
    """

    job_id: str
    name: str
    status: StageStatus
    ms: int = 0
    outcome: TaskOutcome | None = None


@dataclass(slots=True)
class GroupRequest:
    """Everything a group of techniques needs, in one picklable object."""

    job_id: str
    names: tuple[str, ...]
    arrays: JobArrays
    sink: SinkConfig
    #: ``time.monotonic()`` value after which remaining members are skipped. Comes from the
    #: orchestrator's soft deadline; monotonic so a clock adjustment cannot extend a job.
    deadline_at: float
    #: Overlay files are written under ``sink.root``; False when the caller wants scores only.
    write_overlays: bool = True

    def with_arrays(self, arrays: JobArrays) -> GroupRequest:
        return replace(self, arrays=arrays)


@dataclass(slots=True)
class GroupResult:
    job_id: str
    outcomes: list[TaskOutcome] = field(default_factory=list)
    #: Seconds the group spent inside the child, for the timings block.
    wall_s: float = 0.0
    #: ``wall_s`` of the pickling round trip, filled in by the orchestrator (it knows both ends).
    ipc_s: float = 0.0


# ------------------------------------------------------------------ initialisation
def init_child(progress_queue: Any, config: dict[str, Any], threads: int) -> None:
    """Pool ``initializer``: install the progress channel and the worker config.

    Runs once per child, before any task. ``threads`` is re-asserted here rather than trusted from
    the environment because a child inherits the parent's env only as it was at spawn time, and
    ``torch.set_num_threads`` has no effect on cv2 anyway — this is about OpenCV and NumPy.
    """
    global _PROGRESS_QUEUE, _CONFIG, _IN_CHILD, _THREADS
    _PROGRESS_QUEUE = progress_queue
    _CONFIG = dict(config)
    _IN_CHILD = True
    _THREADS = max(1, int(threads))
    try:
        import cv2

        cv2.setNumThreads(_THREADS)
    except Exception:
        pass
    log.debug("worker child ready (pid=%d, threads=%d)", os.getpid(), _THREADS)


def configure_parent(
    config: dict[str, Any],
    threads: int,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> None:
    """Parent-side equivalent of ``init_child`` for in-parent groups (torch).

    The parent takes a callback rather than a queue because it *is* the queue's reader: routing a
    torch technique's progress through a pipe that loops back into the same process would add a
    serialisation step and a race for no benefit.
    """
    global _CONFIG, _THREADS, _IN_CHILD, _PROGRESS_CALLBACK
    _CONFIG = dict(config)
    _THREADS = max(1, int(threads))
    _IN_CHILD = False
    _PROGRESS_CALLBACK = on_event


# ------------------------------------------------------------------------ grouping
def needed_fields(spec: TechniqueSpec) -> frozenset[str]:
    """``JobArrays`` field names one technique reads, from its own bindings."""

    def of(arg: Arg) -> set[str]:
        return {arg.field} if isinstance(arg, FromArrays) else set()

    fields: set[str] = set()
    for arg in spec.args:
        fields |= of(arg)
    fields |= {name for arg in spec.kwargs.values() for name in of(arg)}
    return frozenset(fields)


def group_for(name: str) -> str:
    """Execution group for a technique: its track, or 'learned' when it needs torch."""
    spec = BY_NAME[name]
    return "learned" if spec.heavy else spec.track


def groups_for(names: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    """Group technique names by execution group, preserving registry order.

    ``container`` first because it is the cheapest and produces the first movement in the stepper.
    """
    order = {"container": 0, "native": 1, "analysis": 2, "learned": 3}
    buckets: dict[str, list[str]] = {}
    for name in sorted(names, key=lambda n: (order.get(group_for(n), 9), BY_NAME[n].order)):
        buckets.setdefault(group_for(name), []).append(name)
    return buckets


#: The only fields ``trim_arrays`` empties. Each is either a pixel plane or the whole container,
#: so ``None`` in one of them unambiguously means "this group was not given it".
#:
#: Scalars are deliberately never trimmed. They are a handful of ints, free to pickle, and one of
#: them — ``est_jpeg_quality`` — has ``None`` as a *result*: a PNG genuinely has no JPEG quality.
#: Treating a null scalar as a missing one made every non-JPEG file report ``ela`` and ``noise`` as
#: errored techniques, when the honest answer is that they ran and had nothing to read.
TRIMMABLE_FIELDS: frozenset[str] = frozenset(
    {"rgb_u8", "gray_u8", "native_rgb_u8", "native_gray_u8", "orig_bytes"})


def trim_arrays(arrays: JobArrays, names: tuple[str, ...]) -> JobArrays:
    """A copy of ``arrays`` carrying only the pixel payloads the named techniques read.

    Pickling ``None`` is free; pickling an unused 2560x2560x3 track is not. The copy is shallow for
    the scalars and keeps array identity for the fields that survive, so nothing is duplicated in
    the parent either.

    ``sha256`` and ``mime`` always stay: ``mime`` because several techniques branch on it and the
    registry bindings do not all name it explicitly, ``sha256`` because it belongs to the job rather
    than to a measurement and an outcome that cannot say which file it describes is unloggable.
    """
    keep: set[str] = {"sha256", "mime"}
    for name in names:
        keep |= needed_fields(BY_NAME[name])
    trimmed = replace(arrays)
    for f in TRIMMABLE_FIELDS - keep:
        setattr(trimmed, f, None)
    return trimmed


# ----------------------------------------------------------------------- execution
def run_group(request: GroupRequest) -> GroupResult:
    """Run every technique in ``request.names``, serially, and return their outcomes.

    This is the function that crosses the process boundary, so it takes one picklable argument and
    returns one picklable result, and it must never raise: a group that dies takes its already-computed
    members with it.
    """
    started = time.monotonic()
    result = GroupResult(job_id=request.job_id)
    for name in request.names:
        # Checked before each member, not once for the group: the first technique's cost is the
        # last one's deadline risk, and skipping is what keeps a job's total inside its budget.
        if time.monotonic() > request.deadline_at:
            outcome = _skipped(name, "deadline")
            result.outcomes.append(outcome)
            _emit(request.job_id, name, "skipped", outcome.ms, outcome)
            continue
        _emit(request.job_id, name, "running")
        outcome, stage = _run_one(name, request)
        result.outcomes.append(outcome)
        _emit(request.job_id, name, stage, outcome.ms, outcome)
    result.wall_s = time.monotonic() - started
    return result


def _run_one(name: str, request: GroupRequest) -> tuple[TaskOutcome, StageStatus]:
    """One technique, fully guarded. Returns the outcome and the stage it implies."""
    spec = BY_NAME.get(name)
    if spec is None:
        return _failed(name, "unknown_technique", f"{name!r} is not in the registry"), "failed"
    if _IN_CHILD and spec.heavy:
        return _failed(
            name, "misrouted",
            "learned detector cannot run in a spawn child; it belongs to the parent torch thread",
        ), "failed"

    t0 = time.perf_counter()
    try:
        args, kwargs = _bind(spec, request.arrays)
        module = importlib.import_module(f"app.analysis.techniques.{spec.module}")
        score, visual, details = module.run(*args, **kwargs)
    except Exception as exc:
        log.warning("technique %s failed: %s\n%s", name, exc, traceback.format_exc())
        return _failed(name, type(exc).__name__, _sanitise(exc)), "failed"

    ms = int((time.perf_counter() - t0) * 1000)
    overlay = None
    if visual is not None and request.write_overlays:
        overlay = _write_overlay(spec, visual, request.sink)
    # A NaN score means "I could not measure", and it is a valid *ok* outcome rather than an error.
    # It is preserved as NaN deliberately: ``adapt.build_technique_view`` turns it into ``score=None``
    # and the reliability engine penalises the missing coverage. Coercing it to 0.0 here would make an
    # unreadable image report as a measured-clean one.
    return TaskOutcome(
        name=name, status="ok", score=float(score), details=details,
        overlay=overlay, ms=ms,
    ), "done"


def _bind(spec: TechniqueSpec, arrays: JobArrays) -> tuple[list[Any], dict[str, Any]]:
    """Resolve the registry's binding descriptors into positional and keyword arguments."""

    def resolve(arg: Arg) -> Any:
        if isinstance(arg, FromArrays):
            value = getattr(arrays, arg.field, None)
            if value is None and arg.field in TRIMMABLE_FIELDS:
                raise ValueError(
                    f"{spec.name} needs JobArrays.{arg.field}, which was not sent to this group")
            return value
        if isinstance(arg, FromConfig):
            if arg.field not in _CONFIG:
                raise ValueError(f"{spec.name} needs config.{arg.field}, which is not configured")
            return _CONFIG[arg.field]
        return arg.value

    args = [resolve(a) for a in spec.args]
    kwargs = {k: resolve(v) for k, v in spec.kwargs.items()}
    return args, kwargs


def _write_overlay(spec: TechniqueSpec, visual: np.ndarray, sink: SinkConfig) -> OverlayRef | None:
    """Persist a technique's visual. Failures degrade to "no overlay", never to a failed technique."""
    from app.core.scratch import Scratch, write_overlay

    if not sink.enabled:
        return None
    try:
        scratch = Scratch(sink.root)
        target = scratch.overlay_path(sink.job_id, spec.name)
        if target is None:
            log.warning("refused overlay path for %s/%s", sink.job_id, spec.name)
            return None
        scratch.ensure_job_dir(sink.job_id)
        written = write_overlay(np.asarray(visual), target, max_dim=sink.max_dim)
        return OverlayRef(
            name=spec.name, rel_path=written.rel_path, width=written.width,
            height=written.height, media_type="image/png", expires_at=sink.expires_at,
        )
    except Exception as exc:
        log.warning("overlay write failed for %s: %s", spec.name, exc)
        return None


# -------------------------------------------------------------------------- events
def _emit(
    job_id: str,
    name: str,
    status: StageStatus,
    ms: int = 0,
    outcome: TaskOutcome | None = None,
) -> None:
    """Push a progress event. Never blocks, never raises.

    ``put_nowait`` because the queue is fed from inside the timed region: a full queue (a client that
    stopped polling, an orchestrator that crashed) must not convert a live analysis into a stalled
    one. Losing progress events costs the user a smooth stepper; blocking a worker costs the job.

    ``running`` events carry no outcome — there is nothing to report yet, and shipping a half-built
    ``TaskOutcome`` would suggest the consumer could read it.
    """
    event = ProgressEvent(job_id=job_id, name=name, status=status, ms=ms, outcome=outcome)
    if _PROGRESS_QUEUE is not None:
        try:
            _PROGRESS_QUEUE.put_nowait(event)
        except Exception:
            log.debug("progress event dropped for %s/%s", job_id, name, exc_info=True)
        return
    if _PROGRESS_CALLBACK is not None:
        try:
            _PROGRESS_CALLBACK(event)
        except Exception:
            log.debug("progress callback raised for %s/%s", job_id, name, exc_info=True)


# --------------------------------------------------------------------- outcome help
def _skipped(name: str, code: str) -> TaskOutcome:
    return TaskOutcome(
        name=name, status="skipped", score=None,
        details={"note": "Not started: the job reached its deadline before this technique ran."},
        error_code=code, error_msg="deadline reached before this check started",
    )


def _failed(name: str, code: str, msg: str) -> TaskOutcome:
    return TaskOutcome(
        name=name, status="error", score=None,
        details={"note": "This check could not be completed."},
        error_code=code, error_msg=msg[:200],
    )


#: Windows and POSIX paths, drive letters and UNC prefixes. Removed from client-visible messages so
#: a one-line error does not enumerate the filesystem layout.
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s'\"|]*|[\\/][^\s'\"|]+|\\\\[^\s'\"|]+)")


def _sanitise(exc: BaseException) -> str:
    """One line, at most 200 characters, no paths.

    The message is deliberately lossy. Everything needed to debug is in the log at WARNING with its
    traceback; what the client and the job record keep is enough to say which check failed and roughly
    why, without turning the API into a filesystem browser.
    """
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    text = _PATH_RE.sub("<path>", text)
    text = re.sub(r"\s+", " ", text)
    # An exception with no message still needs to say *something*; prefixing the class name twice
    # ("ValueError: ValueError") is the kind of output that makes a client doubt the rest of it.
    return (f"{type(exc).__name__}: {text}" if text else type(exc).__name__)[:200]
