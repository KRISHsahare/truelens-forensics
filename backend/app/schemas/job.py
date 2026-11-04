"""Job-lifecycle schemas: what a poll returns, and what batch mode accepts.

``JobStatus`` is the endpoint the client hits perhaps twenty times per analysis, so its shape is a
latency decision as much as a data one: ``result`` is ``None`` until the job finishes and is the full
``AnalysisResult`` afterwards, which keeps a single endpoint doing both jobs instead of making the
client issue a second request it can only guess at the timing of.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analysis import AnalysisResult

__all__ = ["BatchItem", "BatchResult", "JobCreated", "JobStatus"]

JobState = Literal["queued", "running", "succeeded", "failed", "expired"]
StageState = Literal["pending", "running", "done", "failed", "skipped", "blocked"]


class JobCreated(BaseModel):
    """The immediate response to an upload: enough to start polling, and nothing else.

    Returning the job id only would leave the client unable to tell the user anything while the
    analysis runs, which is why ``stages`` ships here too — the stepper can render its full shape
    before the first poll lands.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: JobState
    sha256: str
    created_at: float
    poll_after_ms: int = Field(150, description="Suggested first poll delay; the server is not obliged.")
    stages: dict[str, StageState] = Field(default_factory=dict)
    #: Present when the SHA-256 result cache already held this exact image. The client can skip
    #: polling entirely and fetch the result on the first call.
    cached: bool = False


class JobStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: JobState
    #: 0..1 over techniques that reached a *terminal* state, so a job whose checks all failed still
    #: completes rather than spinning at 0.9 forever.
    progress: float = Field(ge=0, le=1)
    stages: dict[str, StageState] = Field(default_factory=dict)
    stage_ms: dict[str, int] = Field(default_factory=dict)
    result: AnalysisResult | None = None
    #: Only set when ``status`` is ``failed``. Sanitised: one line, no paths, no traceback.
    error: dict[str, Any] | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    from_cache: bool = False


class BatchItem(BaseModel):
    """One entry in a batch response.

    ``label`` is the client's own filename echoed back purely as a correlation key — batch mode is
    twenty uploads at once and the job ids alone are unreadable in a table. It is never used for
    anything else and never written to disk, because a filename is user input that can contain
    anything at all.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str
    label: str = ""
    status: JobState
    sha256: str = ""
    error: str | None = None


class BatchResult(BaseModel):
    """Job ids for a batch, submitted together so the client polls each one.

    Per-image failures are reported in ``items[].error`` rather than failing the request: losing one
    screenshot out of twenty to a decode error should not discard the nineteen good analyses, and the
    table the UI renders has a row per input whether or not it has a verdict.
    """

    model_config = ConfigDict(frozen=True)

    items: list[BatchItem] = Field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    limit: int = 20
