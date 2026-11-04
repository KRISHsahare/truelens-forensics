"""Data structures that cross the process boundary.

Deliberately plain ``dataclass`` rather than ``BaseModel``: ``worker.py`` is re-imported by
every spawn child, and dragging ``pydantic-core`` into N workers costs both memory and
startup time. Conversion to Pydantic happens once, at the API boundary (``analysis/adapt.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "JobArrays",
    "JobContext",
    "OverlayRef",
    "SinkConfig",
    "TaskOutcome",
    "TaskStatus",
]

TaskStatus = Literal["ok", "error", "timeout", "skipped"]


@dataclass(slots=True)
class JobArrays:
    """Pre-decoded arrays, built once in the parent and sliced per technique.

    Two resolution tracks are mandatory, not an optimisation:

    * ``analysis`` track (``rgb_u8``/``gray_u8``, longest side <= 2048) for techniques that
      are scale-invariant: noise, copy-move, face, CNN.
    * ``native`` track (``native_rgb_u8``/``native_gray_u8``) for techniques that read the
      JPEG DCT grid or the high-frequency tail: ELA, jpeg-ghost, block-grid, frequency.
      Resampling across grid boundaries destroys the grid phase, and the low-pass that
      downscaling performs fabricates the very spectral decay ``frequency`` tests for, so
      these must run on a **centre crop at original pixel coordinates** that was never
      resampled. Feeding them a resized image makes ELA measure our own interpolation filter
      instead of the camera's compression history.

    ``orig_bytes`` is the untouched container. EXIF, XMP, PNG text chunks, C2PA manifests and
    JPEG quantisation tables all need it pristine — anything re-encoded has already lost the
    C2PA signature.
    """

    sha256: str
    mime: str
    orig_w: int
    orig_h: int
    file_size: int
    #: analysis dims / orig dims, <= 1.0. Techniques with pixel-unit thresholds must scale by this.
    analysis_scale: float
    #: crop box in ORIGINAL pixel coordinates
    native_box: tuple[int, int, int, int]
    rgb_u8: np.ndarray
    gray_u8: np.ndarray
    native_rgb_u8: np.ndarray
    native_gray_u8: np.ndarray
    orig_bytes: bytes
    #: estimated from the JPEG quantisation tables in the parent (free); None for non-JPEG
    est_jpeg_quality: int | None
    #: bits per pixel: file_size * 8 / (w * h). Low bpp => messenger/social recompression.
    bpp: float
    #: True when width or height is a round number typical of screenshots (1920, 1080, ...)
    round_dimensions: bool

    @property
    def min_side(self) -> int:
        return min(self.orig_w, self.orig_h)

    @property
    def pixels(self) -> int:
        return self.orig_w * self.orig_h


@dataclass(slots=True, frozen=True)
class JobContext:
    """The scalars of a ``JobArrays``, after the pixels have been released.

    Fusion, reliability and the feature vector read only these — dimensions, mime, bits-per-pixel,
    estimated quality, the analysis scale factor. Nothing downstream of the analysers looks at an
    array. Holding the full ``JobArrays`` for the life of a job would therefore pin tens of
    megabytes per concurrent analysis to serve no purpose, and "per concurrent analysis" is exactly
    the multiplier that turns a working 2 CPU container into an out-of-memory one.

    Field names deliberately match ``JobArrays`` so consumers accept either without branching.
    """

    sha256: str
    mime: str
    orig_w: int
    orig_h: int
    file_size: int
    analysis_scale: float
    native_box: tuple[int, int, int, int]
    est_jpeg_quality: int | None
    bpp: float
    round_dimensions: bool

    @property
    def min_side(self) -> int:
        return min(self.orig_w, self.orig_h)

    @property
    def pixels(self) -> int:
        return self.orig_w * self.orig_h

    @classmethod
    def from_arrays(cls, a: JobArrays) -> JobContext:
        return cls(
            sha256=a.sha256, mime=a.mime, orig_w=a.orig_w, orig_h=a.orig_h,
            file_size=a.file_size, analysis_scale=a.analysis_scale, native_box=a.native_box,
            est_jpeg_quality=a.est_jpeg_quality, bpp=a.bpp, round_dimensions=a.round_dimensions,
        )


@dataclass(slots=True, frozen=True)
class OverlayRef:
    """A written overlay. The array itself never crosses the pipe back — only this."""

    name: str
    rel_path: str
    width: int
    height: int
    media_type: str
    expires_at: float


@dataclass(slots=True, frozen=True)
class SinkConfig:
    """Where and how a worker should write its overlay.

    Passed by value so workers never touch shared state or the job store.
    """

    root: str
    job_id: str
    max_dim: int
    expires_at: float
    enabled: bool = True


@dataclass(slots=True)
class TaskOutcome:
    """One technique's result. Tiny by design: a float, a JSON-safe dict, a filename."""

    name: str
    status: TaskStatus = "ok"
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    overlay: OverlayRef | None = None
    ms: int = 0
    error_code: str | None = None
    #: Sanitised, <= 200 chars. Must never contain filesystem paths or tracebacks.
    error_msg: str | None = None

    @property
    def degraded(self) -> bool:
        return self.status != "ok"
