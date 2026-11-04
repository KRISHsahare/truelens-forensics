"""Decode an upload into the arrays every technique needs.

The important design decision here is the **two-resolution split**. Resizing an image
resamples across JPEG's 8x8 DCT block boundaries, which destroys exactly the signal that
ELA, jpeg-ghost, block-grid alignment and the FFT spectrum read. Those four therefore run on
a *native* centre crop that has never been resampled; everything else runs on a downscaled
copy so a 60 MP upload still fits the latency budget.

The native crop origin is snapped to a multiple of 8. If it were not, the crop would shift
the DCT grid phase and ``block_grid`` would report misalignment in a perfectly clean image.
Frequency needs the native track for the same underlying reason but in a subtler form: a
downscale is a low-pass filter, so it re-imposes a steeply decaying tail on whatever it
touches, which is the very signature ``frequency`` tests for. Its cost is that on a very
large upload only the centre of the frame is spectrally examined — acceptable, because
spectral decay is a global property and a centre crop estimates it well.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
from PIL import Image

from app.analysis.types import JobArrays
from app.core.security import (
    ImageProbe,
    decode_verified,
    estimate_jpeg_quality,
    is_screenshot_shape,
)

__all__ = ["build_arrays", "sha256_bytes"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_rgb(im: Image.Image) -> Image.Image:
    """Flatten to RGB. Alpha is composited onto mid-grey, not white: compositing onto white
    invents a bright halo that ELA then reports as an edited region."""
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return im.convert("RGB")


def _native_crop_box(w: int, h: int, max_dim: int) -> tuple[int, int, int, int]:
    """Largest centre crop fitting ``max_dim``, 8-aligned, aspect-preserving.

    Returns ``(left, top, right, bottom)`` in original pixel coordinates. Full frame when
    the image already fits (the common case).
    """
    longest = max(w, h)
    if longest <= max_dim:
        return (0, 0, w, h)
    f = max_dim / longest
    cw = max(8, int(w * f) // 8 * 8)
    ch = max(8, int(h * f) // 8 * 8)
    # Snap the origin down to a multiple of 8 so the DCT grid phase is preserved.
    left = max(0, ((w - cw) // 2) // 8 * 8)
    top = max(0, ((h - ch) // 2) // 8 * 8)
    return (left, top, min(w, left + cw), min(h, top + ch))


def _resized(rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is <= ``max_dim``. Returns the array and the scale factor."""
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return rgb, 1.0
    scale = max_dim / longest
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    # INTER_AREA is the correct filter for downscaling: it box-averages rather than
    # interpolating, so it does not alias high-frequency content into the band that ``noise``
    # and ``copy_move`` then measure on this track. INTER_LINEAR/CUBIC would fabricate
    # texture and invent keypoints that were never in the file.
    return cv2.resize(rgb, new, interpolation=cv2.INTER_AREA), scale


def build_arrays(
    data: bytes, probe: ImageProbe, *, analysis_max_dim: int, native_max_dim: int
) -> JobArrays:
    """Decode once, derive every array. Blocking C work; call from a thread, not the loop."""
    im = decode_verified(data, probe)
    rgb_im = _to_rgb(im)

    native_box = _native_crop_box(probe.width, probe.height, native_max_dim)
    native_rgb = np.asarray(rgb_im.crop(native_box), dtype=np.uint8)

    analysis_rgb, analysis_scale = _resized(np.asarray(rgb_im, dtype=np.uint8), analysis_max_dim)
    analysis_gray = cv2.cvtColor(analysis_rgb, cv2.COLOR_RGB2GRAY)
    native_gray = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2GRAY)

    pixels = max(1, probe.width * probe.height)
    out = JobArrays(
        sha256=sha256_bytes(data),
        mime=probe.mime,
        orig_w=probe.width,
        orig_h=probe.height,
        file_size=len(data),
        analysis_scale=analysis_scale,
        native_box=native_box,
        rgb_u8=analysis_rgb,
        gray_u8=analysis_gray,
        native_rgb_u8=native_rgb,
        native_gray_u8=native_gray,
        orig_bytes=data,
        est_jpeg_quality=estimate_jpeg_quality(probe.quantization),
        bpp=round(len(data) * 8 / pixels, 4),
        round_dimensions=is_screenshot_shape(probe.width, probe.height),
    )
    del im, rgb_im
    return out
