"""JPEG double-compression ("JPEG ghost") detection.

Every JPEG save quantises each 8x8 DCT coefficient to an integer multiple of a step ``S(q, u, v)``
derived from the quality. So the histogram of coefficients at a given frequency is *striated*:
values pile up on multiples of ``S`` and the gaps between them are nearly empty. Re-saving at a
finer quality does **not** erase that striation — it survives as a secondary periodicity. If one
part of the frame carries striation at a step the rest of the frame does not, that part was
compressed separately before being composited.

Why not the obvious approach
----------------------------

The intuitive method is to re-encode each block at a ladder of qualities and take the argmin of
the reconstruction error, reading that as the block's original quality. **It does not work**, and
we measured it rather than assuming it:

* A q55 region inside a q90 file reproduces best at q90, identically to the surrounding plate.
  The region-level quality map comes out uniformly at the *final* quality everywhere, on every
  fixture, tampered or not.
* Worse, as ``q -> 100`` re-encoding approaches the identity, so reconstruction error falls for
  *any* input. The argmin is biased toward the top of the ladder regardless of content.

Encoding is a projection onto the final lattice, and the projection is determined by the pixels,
not by their history. What survives is not the *identity* of the earlier step but its
*periodicity* — which is what this module measures. This follows Bianchi & Piva (2012) and
Lukáš & Fridrich: analyse the coefficient distribution statistically over many blocks, never
per block.

Cost: one orthonormal DCT over the whole native frame yields all 64 frequencies, after which
every (frequency, quality, region) combination is plain numpy. No re-encoding at all, which makes
this roughly an order of magnitude cheaper than the ladder approach it replaces.

Runs on the **native** track — the DCT grid must be the encoder's, and an 8-aligned crop
guarantees it.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.fft import dctn

from app.analysis.contract import TechniqueResult, clip01
from app.analysis.techniques._common import colormap_rgb, composite_overlay

__all__ = ["run"]

BLOCK = 8
#: Candidate prior qualities. Only those whose step is informative at a given frequency are
#: actually consulted there (see ``MIN_STEP``).
QUALITIES: tuple[int, ...] = (40, 48, 55, 62, 70, 76, 82, 86, 90, 94)
#: Mid-frequency AC coefficients. Low frequencies are dominated by image content and very high
#: ones are mostly zero after any real compression; both make striation unreadable.
FREQS: tuple[tuple[int, int], ...] = ((1, 2), (2, 1), (0, 3), (3, 0), (2, 2), (1, 3))
#: A quantisation step of 1 or 2 makes nearly every integer a multiple, so "periodicity" is
#: trivially perfect and carries no information. Such candidates are skipped.
MIN_STEP = 3
REGIONS = 5
#: Minimum absolute striation before a region may claim a ghost at some quality.
GHOST_FLOOR = 0.22
#: Minimum striation, present in *every* region, before we name a dominant (final) quality.
#: Deliberately below ``GHOST_FLOOR``: identifying the file's own quality is a weaker claim than
#: claiming a second compression history, so it is allowed to fire on weaker evidence. Measured
#: separation is wide — a genuine single history scores ~0.44, an unreadable one ~-0.02.
DOMINANT_FLOOR = 0.15
#: Fewest usable coefficients per region cell; below this the phase estimate is pure noise.
MIN_SAMPLES = 150
#: Tolerance, in units of the step, around each multiple that still counts as "on the lattice".
TOL = 0.25

#: JPEG Annex K luminance quantisation table, indexed ``[v][u]``.
_LUMA_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


def run(
    native_rgb_u8: np.ndarray,
    mime: str = "image/jpeg",
    native_box: tuple[int, int, int, int] = (0, 0, 0, 0),
    *,
    tiles: int = 3,
) -> TechniqueResult:
    """``tiles`` is accepted for profile compatibility but unused: the DCT is whole-frame."""
    if mime != "image/jpeg":
        return TechniqueResult(0.0, None, {
            "applicable": False,
            "note": "Source is not a JPEG, so it carries no quantisation history to read. "
                    "This check is skipped and does not count as evidence of authenticity.",
        })

    luma = cv2.cvtColor(native_rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = luma.shape
    ny, nx = h // BLOCK, w // BLOCK
    if ny < REGIONS * 4 or nx < REGIONS * 4:
        return TechniqueResult(0.0, None, {
            "applicable": True,
            "note": "Image too small for region-wise double-compression analysis.",
            "min_side": int(min(h, w)),
        })

    # Trimmed to a multiple of REGIONS so every cell holds the same number of blocks.
    ny -= ny % REGIONS
    nx -= nx % REGIONS
    blocks = (luma[: ny * BLOCK, : nx * BLOCK]
              .reshape(ny, BLOCK, nx, BLOCK).transpose(0, 2, 1, 3)
              .astype(np.float64) - 128.0)
    # One orthonormal DCT yields every frequency. JPEG's own forward DCT is orthonormal too, so
    # the coefficient scale matches the quantisation table directly.
    coeffs = dctn(blocks, axes=(-2, -1), norm="ortho")            # (ny, nx, 8, 8)

    cy, cx = ny // REGIONS, nx // REGIONS
    striation: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    steps_used: dict[tuple[int, int], dict[int, int]] = {}
    for (u, v) in FREQS:
        cells = (coeffs[:, :, v, u]
                 .reshape(REGIONS, cy, REGIONS, cx)
                 .transpose(0, 2, 1, 3)
                 .reshape(REGIONS, REGIONS, -1))
        per_q: dict[int, np.ndarray] = {}
        used: dict[int, int] = {}
        for q in QUALITIES:
            step = _step(q, u, v)
            if step < MIN_STEP:
                continue
            val = _striation(cells, step)
            if val is None:
                continue
            per_q[q] = val
            used[q] = step
        if per_q:
            striation[(u, v)] = per_q
            steps_used[(u, v)] = used

    if not striation:
        return TechniqueResult(0.0, None, {
            "applicable": True,
            "note": "No informative quantisation step could be tested at this image's size.",
        })

    # The file's own quality is the one whose periodicity is present in *every* region, so score
    # candidates by their minimum across regions rather than their mean.
    aggregate = {q: float(min(d[q].min() for d in striation.values() if q in d))
                 for q in QUALITIES if any(q in d for d in striation.values())}
    best_q = max(aggregate, key=lambda k: aggregate[k])
    # ``max`` over an all-negative aggregate is arbitrary: it names a quality no region is
    # actually periodic at. Only claim a dominant quality when something is genuinely periodic,
    # otherwise report None and let every quality compete as a ghost candidate.
    dominant_q = best_q if aggregate[best_q] >= DOMINANT_FLOOR else None

    # Ghost = periodicity a region has that its peers do not, at a quality other than the dominant
    # one. Contrasting against the per-quality median across regions removes the content-driven
    # baseline, which is what makes the finding specific to compression history.
    #
    # Aggregated across frequencies by *median*, not max. A genuine prior compression striates
    # most mid-frequencies at once; a coincidental alignment shows up at one. Taking the max would
    # let a single lucky frequency drive the verdict.
    per_q_contrast: dict[int, list[np.ndarray]] = {}
    for per_q in striation.values():
        for q, val in per_q.items():
            if q == dominant_q:
                continue
            per_q_contrast.setdefault(q, []).append(val - np.median(val))
    contrast = {q: np.median(np.stack(vs, axis=0), axis=0)
                for q, vs in per_q_contrast.items()}
    # Absolute striation, likewise medianed, so a region must actually be periodic — not merely
    # more periodic than an unusually quiet neighbourhood.
    per_q_abs: dict[int, list[np.ndarray]] = {}
    for per_q in striation.values():
        for q, val in per_q.items():
            if q != dominant_q:
                per_q_abs.setdefault(q, []).append(val)
    absolute = {q: np.median(np.stack(vs, axis=0), axis=0) for q, vs in per_q_abs.items()}

    ghost = np.zeros((REGIONS, REGIONS), dtype=np.float64)
    ghost_quality = np.zeros((REGIONS, REGIONS), dtype=np.int32)
    for q, val in contrast.items():
        hit = (val > GHOST_FLOOR) & (absolute[q] > GHOST_FLOOR)
        better = hit & (val > ghost)
        ghost[better] = val[better]
        ghost_quality[better] = q
    # A region compressed at q55 and re-saved at q90 stays near its original lattice at several
    # nearby steps. Report the *coarsest* one within 20% of the peak, which is the closest to the
    # true prior quality; the finest is usually an artefact of the final encode.
    for iy in range(REGIONS):
        for ix in range(REGIONS):
            if ghost[iy, ix] <= GHOST_FLOOR:
                continue
            peak = ghost[iy, ix]
            eligible = [q for q, val in contrast.items()
                        if val[iy, ix] >= 0.8 * peak and absolute[q][iy, ix] > GHOST_FLOOR]
            if eligible:
                q0 = min(eligible)
                ghost_quality[iy, ix] = q0
                ghost[iy, ix] = float(contrast[q0][iy, ix])

    fired = ghost > GHOST_FLOOR
    fired_frac = float(fired.mean())
    peak_contrast = float(ghost.max())
    top = float(np.sort(ghost[fired])[-3:].mean()) if fired.any() else 0.0

    # Extent saturates at 8% of the frame. Demanding more would under-score the common case: a
    # small pasted region is just as conclusive as a large one, and a decisive peak contrast on a
    # single region is stronger evidence than a diffuse smear over half the image.
    if fired.any():
        score = clip01(0.62 * min(1.0, peak_contrast / 0.45)
                       + 0.38 * min(1.0, fired_frac / 0.08))
    else:
        score = 0.0
    per_region = [
        {"region": [int(ix), int(iy)], "ghost_quality": int(ghost_quality[iy, ix]),
         "contrast": round(float(ghost[iy, ix]), 4)}
        for iy in range(REGIONS) for ix in range(REGIONS) if fired[iy, ix]
    ]

    visual = _render_overlay(native_rgb_u8, ghost, fired, h // REGIONS, w // REGIONS)

    details = {
        "applicable": True,
        "dominant_quality": None if dominant_q is None else int(dominant_q),
        "dominant_quality_support": round(aggregate[best_q], 4),
        "dominant_quality_floor": DOMINANT_FLOOR,
        "ghost_regions": int(fired.sum()),
        "total_regions": int(fired.size),
        "ghost_region_fraction": round(fired_frac, 4),
        "peak_contrast": round(peak_contrast, 4),
        "mean_top3_contrast": round(top, 4),
        "ghost_floor": GHOST_FLOOR,
        "per_region": sorted(per_region, key=lambda r: -r["contrast"])[:16],
        "frequencies_tested": [[int(u), int(v)] for (u, v) in striation],
        "quality_ladder": list(QUALITIES),
        "steps_tested": {f"{u},{v}": used for (u, v), used in steps_used.items()},
        "min_step_considered": MIN_STEP,
        "region_grid": [REGIONS, REGIONS],
        "blocks_analysed": int(ny * nx),
        "native_box": [int(v) for v in native_box],
        "note": ("One or more regions carry a secondary quantisation periodicity the rest of the "
                 "frame does not share — direct evidence they were compressed separately before "
                 "being composited." if fired.any() else
                 "Every region shares one quantisation periodicity, consistent with a single "
                 "compression pass. Absence of a split is weak evidence, not proof of "
                 "authenticity." if dominant_q is not None else
                 "No quantisation periodicity was detectable in any region, so this check could "
                 "not establish even a single compression history. It reports nothing rather "
                 "than a low number: absence of evidence here is not evidence of absence."),
    }
    return TechniqueResult(score, visual, details)


def _step(quality: int, u: int, v: int) -> int:
    """The JPEG luminance quantisation step for frequency ``(u, v)`` at ``quality``."""
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return max(1, int(math.floor((_LUMA_TABLE[v, u] * scale + 50.0) / 100.0)))


def _striation(cells: np.ndarray, step: int) -> np.ndarray | None:
    """Per-region excess of coefficients landing near a multiple of ``step``.

    One value per region in ``[-2*TOL, 1-2*TOL]``. ``2*TOL`` is the expected hit rate if
    coefficient phases were uniform — i.e. if no quantisation at this step had ever happened — so
    a value near zero means "no periodicity here" and a positive value means "this region was
    quantised at this step".

    Coefficients that quantise to zero carry no phase information and are excluded from both the
    numerator and the denominator, rather than being counted as misses.
    """
    r = np.abs(cells) / float(step)
    valid = r > 0.5
    counts = valid.sum(axis=-1)
    if counts.min() < MIN_SAMPLES:
        return None
    frac = np.where(valid, r - np.floor(r), TOL * 2.0)   # invalid -> guaranteed miss
    near = (np.minimum(frac, 1.0 - frac) < TOL) & valid
    return near.sum(axis=-1) / counts - 2.0 * TOL


def _render_overlay(base_rgb: np.ndarray, ghost: np.ndarray, fired: np.ndarray,
                    cell_h: int, cell_w: int) -> np.ndarray:
    """Tint regions carrying a foreign periodicity; intensity is the contrast strength."""
    if not fired.any():
        return base_rgb.copy()
    heat = np.zeros(base_rgb.shape[:2], dtype=np.float32)
    for iy in range(REGIONS):
        for ix in range(REGIONS):
            if fired[iy, ix]:
                heat[iy * cell_h:(iy + 1) * cell_h,
                     ix * cell_w:(ix + 1) * cell_w] = float(np.clip(ghost[iy, ix] / 0.5, 0, 1))
    heat_rgb = colormap_rgb(heat, cv2.COLORMAP_MAGMA)
    return composite_overlay(base_rgb, heat_rgb, alpha=0.45)
