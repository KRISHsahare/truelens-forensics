"""``POST /api/analyze`` — the only route that starts work.

Two ways to supply the image, one response shape: a multipart upload, or a ``url`` field the server
fetches on your behalf. Both end as ``bytes`` handed to the orchestrator, which is where every real
validation lives — so this route deliberately does not pre-check format, decoded dimensions or
content-type. Duplicating those rules here would create a second, weaker copy of them and a path a
client could take to reach the softer one.

The response is a job id, not a result. That is the spec's async contract, and it is also what makes
the concurrency cap work: the request that queues the job returns immediately, so a busy server
refuses at admission instead of holding twenty connections open while one CPU does one analysis.

**Why the form is read by hand.** ``UploadFile | None`` is not a parameter FastAPI can declare: its
dependency analyser treats an optional file field as a body model, and the model cannot be built
because an uploaded file is not a Pydantic type. Since one of the two fields is always optional here,
``await request.form()`` is the honest way to express "either, not both", and the OpenAPI body schema
is declared alongside it so the generated docs still describe the contract.

Returning a ``JSONResponse`` from a route that declares ``response_model`` is not a contradiction:
Starlette passes a ``Response`` instance through untouched, which is how the error envelope keeps the
same shape here as everywhere else.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.api.deps import get_orchestrator, get_settings, rate_key, rejected_response
from app.api.limits import limit_analyze, limiter
from app.core.fetch import FetchError, fetch_image
from app.core.orchestrator import Rejected
from app.schemas.job import JobCreated

__all__ = ["router"]

log = logging.getLogger("truelens.api.analyze")

router = APIRouter(prefix="/api", tags=["analyze"])

#: Multipart bodies are read in this many bytes at a time so the size cap is enforced while streaming
#: instead of after buffering.
_CHUNK = 256 * 1024

#: The documented body shape, declared by hand because the endpoint parses the form itself.
_BODY_SCHEMA = {
    "multipart/form-data": {
        "schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "format": "binary",
                         "description": "The image to analyse."},
                "url": {"type": "string", "description": "An http(s) image URL to fetch instead."},
            },
        }
    },
}


@router.post(
    "/analyze",
    response_model=JobCreated,
    summary="Start an analysis, from an upload or a URL",
    openapi_extra={"requestBody": {"required": True, "content": _BODY_SCHEMA}},
    responses={
        400: {"description": "Empty body, no image supplied, or a URL that would not fetch"},
        413: {"description": "Above the upload size cap"},
        415: {"description": "Not a JPEG, PNG, WebP or BMP"},
        429: {"description": "Per-address rate limit"},
        503: {"description": "Every analysis slot is busy"},
    },
)
@limiter.limit(limit_analyze)
async def analyze(request: Request, response: Response):
    settings = get_settings(request)
    try:
        data, label = await _payload(request, max_bytes=settings.max_upload_bytes)
    except Rejected as exc:
        return rejected_response(exc)

    orch = get_orchestrator(request)
    try:
        record = await orch.create_job(data, client=rate_key(request), label=label)
    except Rejected as exc:
        return rejected_response(exc)
    return JobCreated(
        job_id=record.id,
        status=record.status,
        sha256=record.sha256,
        created_at=record.created_at,
        stages=dict(record.stages),
        cached=record.from_cache,
    )


async def _payload(request: Request, *, max_bytes: int) -> tuple[bytes, str]:
    """Resolve the request into image bytes plus a log-only label.

    Raises ``Rejected`` with the status the client should see. The label is the client's own filename:
    it is never persisted, never rendered, and exists so a batch of uploads can be correlated with
    the rows the user is looking at.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise Rejected(
            400, "invalid_image", "Send a multipart/form-data request.",
            detail="This endpoint takes an uploaded 'file' field or a 'url' field, not JSON.")

    try:
        form = await request.form()
    except Exception as exc:
        log.info("multipart parse failed: %s", type(exc).__name__)
        raise Rejected(400, "invalid_image", "The upload could not be read.") from exc

    target = str(form.get("url") or "").strip()
    upload = form.get("file")
    if target:
        try:
            # Blocking DNS and socket work; the fetcher is not written to be cancelled politely.
            fetched = await run_in_threadpool(fetch_image, target, max_bytes=max_bytes)
        except FetchError as exc:
            raise Rejected(400, "invalid_image", exc.message) from exc
        return fetched.data, "url-import"

    if not isinstance(upload, UploadFile):
        raise Rejected(
            400, "invalid_image", "No image was uploaded.",
            detail="Send a multipart 'file' field, or a 'url' field pointing at an image.")

    buffer = bytearray()
    while True:
        chunk = await upload.read(_CHUNK)
        if not chunk:
            break
        buffer += chunk
        if len(buffer) > max_bytes:
            await upload.close()
            raise Rejected(
                413, "too_large", "That image is larger than the server accepts.",
                detail=f"The limit is {max_bytes:,} bytes.", extra={"max_bytes": max_bytes})
    await upload.close()
    if not buffer:
        raise Rejected(400, "invalid_image", "The upload was empty.")
    return bytes(buffer), str(upload.filename or "")
