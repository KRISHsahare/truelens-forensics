"""Scratch files: where overlays live, and how they disappear.

**The upload is never written to disk.** The spec asks for no server-side image storage, and the
cheap way to satisfy that while still being able to say you did it is to write the upload to a temp
file, delete it at the end of the job, and note that a crash leaves it behind. So the decoded bytes
stay in memory and are handed to the pool as a buffer, and the only thing that ever reaches the
filesystem here is the PNG overlay the user explicitly asked to look at. Deleting those is a
correctness matter (disk fills up), not a privacy one.

What the sweeper enforces is a *ceiling*, not a promise: "overlays are removed within
``ttl + interval`` seconds". A process killed with ``SIGKILL`` leaves files, which is why the
directory is documented as disposable and why nothing is stored in it that cannot be recomputed.

Path handling is the security-relevant part of this module. ``GET /overlays/{job_id}/{name}.png``
takes both segments from the URL, so a naive ``os.path.join(root, job_id, name)`` with
``..%2F`` in either segment reads any PNG on the machine the process can reach. Both are validated
against a strict character class *and* the resolved path is checked to still be under the scratch
root, so a future change to either layer still fails closed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["OverlayWrite", "Scratch", "job_id_safe", "path_safe", "write_overlay"]

log = logging.getLogger("truelens.scratch")

#: Job ids are minted by ``job.new_job_id``: lowercase hex timestamp, dash, hex nonce.
_JOB_ID_RE = re.compile(r"^[0-9a-f]{1,24}-[0-9a-f]{8,64}$")
#: Overlay names are technique names, which are registry-declared ASCII identifiers.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


@dataclass(frozen=True, slots=True)
class OverlayWrite:
    """What ``write_overlay`` produced, in the shape ``OverlayRef`` expects."""

    #: Relative to the scratch root, ``"{job_id}/{name}.png"``. Carries the job directory rather
    #: than just the filename because ``abs_overlay_url_path`` resolves against the root, and a
    #: bare filename would resolve to a sibling of every other technique's overlay.
    rel_path: str
    width: int
    height: int
    bytes_written: int


def job_id_safe(job_id: str) -> bool:
    """True when a URL path segment can only name a job this process minted.

    Deliberately an allowlist rather than a blocklist: the attack is not ``..`` specifically, it is
    any spelling of it that the platform's path layer happens to accept.
    """
    return bool(_JOB_ID_RE.fullmatch(job_id))


def path_safe(job_id: str, name: str) -> bool:
    """True when both overlay URL segments can only address a file we minted ourselves."""
    return job_id_safe(job_id) and bool(_NAME_RE.fullmatch(name))


def _under(root: Path, candidate: Path) -> bool:
    """Second check, after ``path_safe``: the resolved target is still inside the root.

    ``Path.resolve`` collapses symlinks and ``..`` segments, so this catches a symlink planted
    inside the scratch tree pointing outward — something no character allowlist can see.
    """
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


class Scratch:
    """The scratch tree for one process.

    Created once at startup and shared, because the pool workers each need to resolve the same
    root and a per-request ``mkdtemp`` would scatter the tree beyond what the sweeper can find.
    """

    def __init__(self, root: str | None = None) -> None:
        if root:
            self.root = Path(root).expanduser().resolve()
            self.root.mkdir(parents=True, exist_ok=True)
            self._owns = True
        else:
            self.root = Path(tempfile.mkdtemp(prefix="truelens-scratch-"))
            self._owns = True

    # ------------------------------------------------------------------- layout
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def overlay_path(self, job_id: str, name: str) -> Path:
        """Absolute path for an overlay, or ``None`` when the segments are not addressable.

        Callers must treat ``None`` as a 404, never as an error that leaks why.
        """
        if not path_safe(job_id, name):
            return None
        p = self.job_dir(job_id) / f"{name}.png"
        return p if _under(self.root, p) else None

    def abs_overlay_url_path(self, job_id: str, rel_path: str) -> Path | None:
        """Resolve a stored ``rel_path`` back to disk, guarding against traversal."""
        candidate = (self.root / rel_path)
        return candidate if _under(self.root, candidate) else None

    # ------------------------------------------------------------------ lifecycle
    def ensure_job_dir(self, job_id: str) -> Path:
        if not _JOB_ID_RE.fullmatch(job_id):
            # Fail loud rather than creating a directory outside the tree. Job ids are minted by
            # ``new_job_id``, so reaching this line means one was fabricated somewhere upstream.
            raise ValueError("malformed job id")
        d = self.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def remove_job(self, job_id: str) -> None:
        if not _JOB_ID_RE.fullmatch(job_id):
            log.warning("refusing to remove scratch dir for malformed job id")
            return
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def sweep_expired(self, ttl_s: float, *, now: float | None = None) -> int:
        """Delete any job directory older than ``ttl_s``. Returns how many went.

        Age comes from the directory mtime rather than the job record, because the job may have
        already left the store — a directory whose job expired is precisely the leak this catches.
        """
        now = now or time.time()
        removed = 0
        if not self.root.is_dir():
            return 0
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if now - mtime > ttl_s:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed

    def discard(self) -> None:
        """Best-effort cleanup of the whole tree at shutdown."""
        if self._owns:
            shutil.rmtree(self.root, ignore_errors=True)


def write_overlay(
    rgb: np.ndarray, out_path: Path, *, max_dim: int
) -> OverlayWrite:
    """Downscale to ``max_dim`` and write a PNG. Returns what was written where.

    Overlays are resized *here*, at the sink, rather than in the analyser: an analyser must report
    on the resolution it measured, and silently shrinking the array it works on would change its
    block statistics. Only the picture the human looks at can be resized freely.

    PNG, not JPEG — a heatmap with sharp colour boundaries is exactly what JPEG ringing ruins, and
    the files are transient so their size costs nothing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if max_dim and longest > max_dim:
        f = max_dim / longest
        rgb = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))),
                         interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # Contract says techniques hand back RGB; cv2.imwrite expects BGR. The single conversion lives
    # here so a technique that forgets shows up as blue face boxes, not as twenty inconsistent ones.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    if not ok:
        raise OSError(f"overlay write failed for {out_path.name}")
    return OverlayWrite(rel_path=f"{out_path.parent.name}/{out_path.name}",
                        width=int(w), height=int(h), bytes_written=os.path.getsize(out_path))
