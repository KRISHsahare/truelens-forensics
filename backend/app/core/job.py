"""The job record: states, per-technique progress, and the one rule that keeps them honest.

``POST /api/analyze`` returns a job id immediately and the client polls, so the job record is the
only shared state between the request that created it and every poll afterwards. It is a plain
dataclass with explicit ``to_dict`` / ``from_dict`` rather than a Pydantic model because it is
written from inside a process pool and read by possibly-several web workers; it has to survive a
``redis`` round trip and it must not be able to fail validation while an analysis is mid-flight.

**Terminal-sticky stages.** The progress stepper is driven by callbacks from pool workers, and two
of those callbacks can land out of order: a technique's "done" and the watchdog's "timed out" for
the same technique are racing across a thread boundary and the network is not the arbiter. Without
a rule, the display flickers backwards from Done to Running, or a timeout overwrites a result that
actually arrived. So transitions are one-way: ``pending -> running -> terminal``, where terminal is
``done | failed | skipped``, and a write to an already-terminal stage is ignored rather than
applied. The first writer to reach a terminal state wins, which is the only self-consistent choice
when two sources disagree about whether work finished.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["TERMINAL_STATUSES", "JobRecord", "JobStatus", "StageStatus", "new_job_id"]

JobStatus = Literal["queued", "running", "succeeded", "failed", "expired"]

#: Per-technique progress state. ``blocked`` means the orchestrator will not run it at all, which
#: is different from ``skipped`` (chose not to run for this profile) and from ``failed``.
StageStatus = Literal["pending", "running", "done", "failed", "skipped", "blocked"]

_TERMINAL: frozenset[str] = frozenset({"done", "failed", "skipped", "blocked"})
_RANK: dict[str, int] = {"pending": 0, "running": 1, "skipped": 2, "blocked": 2,
                         "failed": 3, "done": 3}

#: How long a job stays readable after finishing, and how long its overlays stay fetchable.
#: Declared here so the two cannot drift apart from what the sweeper enforces.
TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "expired"})


def new_job_id() -> str:
    import secrets

    # Sortable prefix: time-ordered ids keep the store's eviction and the logs readable without a
    # secondary index. The 128-bit nonce is the whole of the unguessability — nothing else in the
    # id needs to be cryptographic, so nothing here pretends it is.
    return f"{format(int(time.time() * 1000), 'x')}-{secrets.token_hex(16)}"


@dataclass(slots=True)
class JobRecord:
    id: str
    sha256: str
    mime: str
    size_bytes: int
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    #: technique name -> stage state, seeded from the registry so progress has a denominator.
    stages: dict[str, str] = field(default_factory=dict)
    #: technique name -> ms, for the timings block the spec asks for.
    stage_ms: dict[str, int] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    #: Set when the result was served from the SHA-256 cache rather than recomputed.
    from_cache: bool = False
    client: str = ""

    # ------------------------------------------------------------------ stages
    def init_stages(self, names: list[str]) -> None:
        self.stages = {n: "pending" for n in names}

    def set_stage(self, name: str, status: StageStatus, *, ms: int | None = None) -> bool:
        """Advance one technique's stage. Returns False if the write was refused.

        Refusing a regression is the point of this method; see the module docstring.
        """
        current = self.stages.get(name)
        if current is None:
            return False
        if current in _TERMINAL and _RANK[status] <= _RANK[current]:
            return False
        if status == "pending" and current == "running":
            return False
        self.stages[name] = status
        if ms is not None:
            self.stage_ms[name] = int(ms)
        return True

    @property
    def progress(self) -> float:
        """Fraction of stages that have reached a terminal state.

        ``done``, ``failed``, ``skipped`` and ``blocked`` all count: the question the UI is
        answering is "is there anything left to wait for", not "did everything succeed". A job
        whose techniques all failed must still reach 1.0, or the client polls forever.
        """
        if not self.stages:
            return 0.0 if self.status in ("queued", "running") else 1.0
        return sum(1 for s in self.stages.values() if s in _TERMINAL) / len(self.stages)

    # ----------------------------------------------------------------- serialise
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "sha256": self.sha256, "mime": self.mime,
            "size_bytes": self.size_bytes, "status": self.status,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "stages": dict(self.stages),
            "stage_ms": dict(self.stage_ms), "result": self.result, "error": self.error,
            "from_cache": self.from_cache, "client": self.client,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobRecord:
        return cls(
            id=d["id"], sha256=d["sha256"], mime=d.get("mime", ""),
            size_bytes=int(d.get("size_bytes", 0)), status=d.get("status", "queued"),
            created_at=float(d.get("created_at", time.time())),
            started_at=d.get("started_at"), finished_at=d.get("finished_at"),
            stages=dict(d.get("stages") or {}), stage_ms=dict(d.get("stage_ms") or {}),
            result=d.get("result"), error=d.get("error"),
            from_cache=bool(d.get("from_cache", False)), client=d.get("client", ""),
        )

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def age(self, now: float | None = None) -> float:
        return (now or time.time()) - self.created_at
