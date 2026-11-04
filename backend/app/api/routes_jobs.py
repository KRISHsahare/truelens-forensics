"""``GET /api/jobs/{id}`` — the poll.

The whole contract is here: queued, running with per-technique stages, or terminal with the result
attached. One endpoint rather than a status endpoint plus a separate result endpoint because the
client would have to guess when to switch, and a two-request handshake either polls twice as often
or shows a finished job for one poll too long.

``response_model_exclude_none`` keeps a running job's body to what it actually knows. ``result:
null`` in every one of twenty polls invites client code that treats a null result as "the analysis
found nothing", which is precisely the confusion the ``measured`` flag exists to prevent.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.api.deps import error_response, get_orchestrator
from app.api.limits import limit_jobs, limiter
from app.core.job import JobRecord
from app.core.scratch import job_id_safe
from app.schemas.analysis import AnalysisResult
from app.schemas.job import JobStatus

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["jobs"])

_GONE = (
    "Not found. Jobs are deleted within ten minutes of upload, so this may simply have expired; "
    "nothing is stored longer than that."
)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    response_model_exclude_none=True,
    summary="Poll a job for progress and, when finished, its result",
    responses={404: {"description": "Unknown, expired, or already deleted job id"}},
)
@limiter.limit(limit_jobs)
async def get_job(request: Request, job_id: str, response: Response):
    # The same allowlist the overlay route uses, applied before the store is touched. A job id is
    # hex-and-dash by construction, so anything else was typed or probed rather than issued by us,
    # and the answer to both is the same 404 that an expired job gets. Starlette percent-decodes
    # path parameters, so ``..%2f`` reaches this function as a literal ``../``.
    if not job_id_safe(job_id):
        return error_response(404, "not_found", _GONE)

    record = await get_orchestrator(request).fetch(job_id)
    if record is None:
        return error_response(404, "not_found", _GONE)
    return to_status(record)


def to_status(record: JobRecord) -> JobStatus:
    """The public shape of a job.

    ``result`` is revalidated rather than passed through as the stored dict. The record is written by
    the orchestrator and read here, and the revalidation is what guarantees the API cannot serve a
    schema change that happened between them — an expensive guarantee for a poll, which is why the
    rest of the response is built without copying the payload twice.
    """
    result: AnalysisResult | None = None
    if record.result is not None and record.status == "succeeded":
        try:
            result = AnalysisResult.model_validate(record.result)
        except Exception:
            return JobStatus(
                job_id=record.id, status="failed", progress=1.0,
                stages=dict(record.stages), stage_ms=dict(record.stage_ms),
                created_at=record.created_at, started_at=record.started_at,
                finished_at=record.finished_at, from_cache=record.from_cache,
                error={"code": "internal", "message":
                       "The stored result no longer matches the response schema."})
    return JobStatus(
        job_id=record.id,
        status=record.status,
        # ``progress`` counts terminal stages, so a job where everything failed still reaches 1.0 and
        # the client stops polling. A job with no stages at all is either queued or already over.
        progress=record.progress if record.stages else (1.0 if record.is_terminal() else 0.0),
        stages=dict(record.stages),
        stage_ms=dict(record.stage_ms),
        result=result,
        error=record.error if record.status == "failed" else None,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        from_cache=record.from_cache,
    )
