"""``POST /api/batch`` — up to ``BATCH_MAX_IMAGES`` analyses in one request.

Batch mode returns *job ids*, not results. Twenty images at once would otherwise be a single request
that has to stay open for twenty analyses, and the only way to make that work is to bound it — at
which point the client needs a resumable partial-result protocol, which is a job queue with extra
steps. Returning ids lets the batch page reuse the exact poller the single-image page already has,
which is why this route is thin.

Per-image failures are reported in the item rather than failing the request. Losing one screenshot out
of twenty to a truncated PNG should not discard the nineteen good analyses; but a request with *no*
usable image is a request-shaped problem, and that is a 4xx on the whole call.

Files are read and submitted one at a time on purpose: peak memory is then one image plus the queue,
rather than twenty images held while the twentieth is analysed.

The form is parsed by hand for the same reason ``/api/analyze`` parses it by hand — see that module's
docstring — and because it lets the per-file size cap be applied while streaming.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from starlette.datastructures import UploadFile

from app.api.deps import get_orchestrator, get_settings, rate_key, rejected_response
from app.api.limits import limit_batch, limiter
from app.core.orchestrator import Rejected
from app.schemas.job import BatchItem, BatchResult

__all__ = ["router"]

log = logging.getLogger("truelens.api.batch")

router = APIRouter(prefix="/api", tags=["batch"])

_CHUNK = 256 * 1024


@router.post(
    "/batch",
    response_model=BatchResult,
    summary="Queue several images at once",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {"type": "array", "items": {"type": "string",
                                                                 "format": "binary"}},
                            "labels": {"type": "string",
                                       "description": "Comma-separated row keys, one per file."},
                        },
                        "required": ["files"],
                    }
                }
            },
        }
    },
    responses={
        400: {"description": "No files, or more files than the batch limit"},
        413: {"description": "One or more files above the upload size cap"},
        429: {"description": "Per-address batch rate limit"},
    },
)
@limiter.limit(limit_batch)
async def batch(request: Request, response: Response):
    settings = get_settings(request)
    limit = max(1, int(settings.batch_max_images))

    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        return rejected_response(Rejected(
            400, "invalid_image", "Send the images as multipart/form-data fields."))
    try:
        form = await request.form()
    except Exception as exc:
        log.info("batch multipart parse failed: %s", type(exc).__name__)
        return rejected_response(Rejected(400, "invalid_image", "The upload could not be read."))

    uploads = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
    if not uploads:
        return rejected_response(Rejected(
            400, "invalid_image", "No files were uploaded.",
            detail="Each image must arrive in a multipart field named 'files'."))
    if len(uploads) > limit:
        return rejected_response(Rejected(
            400, "too_many_images", f"Batch mode accepts at most {limit} images per request.",
            detail=f"{len(uploads)} were sent.", extra={"limit": limit}))

    # A label is a display key the client chose for its own table. The list can disagree with the
    # number of files, so it is indexed defensively and never trusted to be the same length.
    keys = [p.strip() for p in str(form.get("labels") or "").split(",")]

    orch = get_orchestrator(request)
    client = rate_key(request)
    items: list[BatchItem] = []
    accepted = rejected = 0

    for index, upload in enumerate(uploads):
        label = keys[index] if index < len(keys) else str(upload.filename or "")
        try:
            data = await _read(upload, max_bytes=settings.max_upload_bytes)
            record = await orch.create_job(data, client=client, label=label)
        except Rejected as exc:
            # The item keeps the human sentence: the batch table has a per-row error cell, and a row
            # that says "unsupported_media" tells a user nothing about their screenshot.
            items.append(BatchItem(job_id="", label=label, status="failed",
                                   error=_plain(exc)))
            rejected += 1
            continue
        except Exception as exc:
            log.warning("batch item %d failed before submission: %s", index, type(exc).__name__)
            items.append(BatchItem(job_id="", label=label, status="failed",
                                   error="This file could not be processed."))
            rejected += 1
            continue
        finally:
            await upload.close()
        items.append(BatchItem(job_id=record.id, label=label, status=record.status,
                               sha256=record.sha256))
        accepted += 1

    if not accepted:
        return rejected_response(Rejected(
            400, "invalid_image", "None of the uploaded files could be analysed.",
            detail=items[0].error if items else None))

    return BatchResult(items=items, accepted=accepted, rejected=rejected, limit=limit)


async def _read(upload: UploadFile, *, max_bytes: int) -> bytes:
    """Stream one upload into memory with the cap applied while reading, not after."""
    buffer = bytearray()
    while True:
        chunk = await upload.read(_CHUNK)
        if not chunk:
            break
        buffer += chunk
        if len(buffer) > max_bytes:
            raise Rejected(
                413, "too_large", "That image is larger than the server accepts.",
                detail=f"The limit is {max_bytes:,} bytes.", extra={"max_bytes": max_bytes})
    if not buffer:
        raise Rejected(400, "invalid_image", "One of the uploaded files was empty.")
    return bytes(buffer)


def _plain(exc: Rejected) -> str:
    """The client-safe sentence for one table cell."""
    return f"{exc.message} {exc.detail or ''}".strip()
