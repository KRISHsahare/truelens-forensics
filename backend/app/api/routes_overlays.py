"""``GET /api/overlays/{job_id}/{name}.png`` — the only route that serves a stored file.

Two independent gates, and the order matters:

1. **The signature.** Without it, job ids are a catalogue: they are minted from a millisecond
   timestamp and a random tail, so they are unguessable but *time-sortable*, and an unsigned overlay
   URL handed to anyone who can read a referer header lets them enumerate who uploaded what.
2. **The path.** The job id and technique name come from the URL, so they are checked against the
   same allowlist the writer uses and then resolved with symlink-collapsing containment. A request
   for ``..`` is refused by rule 2 even if it somehow carried a valid signature for a name that
   should never have been signed.

Both refusals produce the same 404 with the same body. "Signature invalid" and "file gone" are
different facts, and distinguishing them tells a caller whether a job id is live — which is the one
thing the unguessable id was supposed to keep secret.

``no-store`` is not paranoia here: an overlay is a heatmap of a stranger's uploaded photograph, and a
shared or cached copy of it is a leak the application cannot revoke.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.api.deps import error_response, get_orchestrator, get_signer
from app.core.scratch import path_safe

__all__ = ["router"]

log = logging.getLogger("truelens.api.overlays")

router = APIRouter(prefix="/api/overlays", tags=["overlays"])

#: Same wording as the job 404. Every refusal on this route is indistinguishable from every other.
_NOT_FOUND = "Not found, or the link has expired."


@router.get("/{job_id}/{filename}", summary="Fetch a signed overlay PNG")
async def get_overlay(request: Request, job_id: str, filename: str):
    signer = get_signer(request)
    name = filename.removesuffix(".png")
    # ``path_safe`` is the writer's rule, so a URL that could address a file we never wrote is
    # refused before any signature work happens. The ".png" spelling is part of the identity we
    # verify, so a name carrying a second extension fails here rather than at the digest.
    if not path_safe(job_id, name):
        return error_response(404, "not_found", _NOT_FOUND)

    params = request.query_params
    if not signer.check(job_id, name, params.get("exp", ""), params.get("sig", "")):
        return error_response(404, "not_found", _NOT_FOUND)

    path = get_orchestrator(request).scratch.overlay_path(job_id, name)
    if path is None or not path.is_file():
        # A valid signature on a swept file is the normal expiry race, not an attack: the URL lives
        # as long as the TTL, and the sweeper runs on its own interval.
        return error_response(404, "not_found", _NOT_FOUND)

    return FileResponse(
        path,
        media_type="image/png",
        # Private + must-revalidate: a proxy that caches this can serve it to the next visitor.
        headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer",
                 "X-Content-Type-Options": "nosniff"},
        filename=None,
    )
