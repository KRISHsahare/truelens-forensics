"""Error Level Analysis.

Re-saving a JPEG at its **own** quality pushes every DCT coefficient back onto the same
quantisation lattice it already occupies. A region of the file that has been through that encode
is a fixed point and reproduces almost exactly; a region that was pasted in from a different
source, resampled, retouched, or quantised with a different table does **not** reproduce, and
lights up.

Runs on the **native** (never resampled) track: resampling across DCT block boundaries would
make this measure our own interpolation filter instead of the file's compression history.

Fixes over the prototype
------------------------

* **Re-save at the file's own estimated quality**, not a fixed q90. Measured on controlled
  fixtures, re-saving at q90/q95/q98 gives a spliced-to-clean error ratio of only 1.1-1.4x,
  because the mismatched quality produces large error *everywhere* and drowns the local signal.
  At the file's own quality the same fixtures separate by 8-17x, and an untouched file gives
  essentially zero. This is the single change that makes ELA usable at all.
* **Absolute fixed-point floor.** A clean, well-converged JPEG reproduces to within rounding.
  We gate on the 99th percentile of per-block error, so a pristine file reports "nothing
  measurable" rather than a number derived from its own rounding noise.
* **Content-confound discount.** This is the honest part. ELA cannot tell "different
  compression history" from "more high-frequency detail": a sharp subject against a smooth
  background reproduces *worse* than a splice does. We therefore measure the rank correlation
  between per-block error and per-block edge energy, and discount the score when that
  correlation is high, reporting the coefficient so the UI can say why.
* **Adaptive, bounded display gain.** The prototype's ``255 / max`` stretches a pristine file to
  full contrast so it *looks* dramatic. Gain here is normalised to the 99.9th percentile of
  error and clamped, then reported; when the floor gate trips the overlay is flagged as
  non-meaningful so the frontend renders "no measurable error" instead of amplified noise.

Non-JPEG sources score 0 with an explanatory note rather than a discounted number: a PNG has no
prior compression history, so any error measured is our own re-encode, not evidence.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.analysis.contract import TechniqueResult, clip01
from app.analysis.techniques._common import block_stats, colormap_rgb, composite_overlay

__all__ = ["run"]

BLOCK = 32
#: Used only when the quantisation table is unreadable.
FALLBACK_QUALITY = 90
#: 99th-percentile of per-block luma error below which the re-encode is a fixed point and there
#: is nothing to measure. Calibrated against controlled fixtures: untouched files land near
#: 0.013, real splices at 0.06-0.22.
NEGLIGIBLE_P99 = 0.02
#: Spearman rho between block error and block edge energy above which the finding is explained
#: by image detail rather than by compression history.
CONTENT_RHO = 0.30
#: Score multiplier applied when the finding is content-confounded.
CONFOUND_DISCOUNT = 0.45
MIN_GAIN, MAX_GAIN = 8.0, 2000.0


def run(
    native_rgb_u8: np.ndarray,
    mime: str = "image/jpeg",
    est_jpeg_quality: int | None = None,
    *,
    quality: int | None = None,
    block: int = BLOCK,
    scale: float | None = None,
) -> TechniqueResult:
    """``native_rgb_u8`` is an (H, W, 3) uint8 RGB array at original resolution."""
    h, w = native_rgb_u8.shape[:2]
    if mime != "image/jpeg":
        return TechniqueResult(0.0, None, {
            "applicable": False,
            "note": "Source is not a JPEG, so it has no prior compression history. Any error "
                    "measured here would be produced by our own re-encode rather than by the "
                    "file, so Error Level Analysis reports nothing and the reliability score "
                    "is reduced instead.",
            "min_side": int(min(h, w)),
        })
    if min(h, w) < block * 2:
        return TechniqueResult(0.0, None, {
            "applicable": True,
            "note": "Image too small for block-wise Error Level Analysis.",
            "min_side": int(min(h, w)),
        })

    resave_quality = int(quality or est_jpeg_quality or FALLBACK_QUALITY)
    resave_quality = max(20, min(98, resave_quality))

    buf = io.BytesIO()
    # subsampling=0 (4:4:4). Matching the original chroma subsampling is impossible in general,
    # and 4:2:0 would smear chroma error across 2x2 blocks, masking a luma-localised splice.
    Image.fromarray(native_rgb_u8).save(buf, "JPEG", quality=resave_quality, subsampling=0)
    buf.seek(0)
    resaved = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)

    diff = np.abs(native_rgb_u8.astype(np.float32) - resaved)
    # Luminance-weighted mean error per pixel; the eye weights channels the same way.
    gray_err = diff @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    means, _, _ = block_stats(gray_err, block)
    mean_of_means = float(means.mean())
    p99_blocks = float(np.percentile(means, 99))
    # Coefficient of variation: scale-invariant measure of how unevenly error is distributed.
    cov = float(means.std() / (mean_of_means + 1e-6))

    # Robust hot-block detection. The prototype used mean + 2*std, which is itself dragged up
    # by the hot blocks it is trying to find. Median + MAD does not move.
    med = float(np.median(means))
    mad = float(np.median(np.abs(means - med))) + 1e-6
    z = (means - med) / (1.4826 * mad)
    hot = z > 4.0
    hot_ratio = float(hot.mean())

    # Spatial clustering: scattered hot blocks are texture; a contiguous hot *region* is a splice.
    clustered = _largest_cluster_fraction(hot.astype(np.uint8))

    negligible = p99_blocks < NEGLIGIBLE_P99
    rho = _detail_correlation(gray_err, native_rgb_u8, block)
    content_confounded = (not negligible) and rho > CONTENT_RHO

    if negligible:
        score = 0.0
    else:
        score = clip01(
            0.30 * min(1.0, cov / 1.5)
            + 0.26 * min(1.0, hot_ratio * 12.0)
            + 0.34 * clustered
            + 0.10 * min(1.0, np.log10(max(p99_blocks, 1e-9) / NEGLIGIBLE_P99))
        )
        if content_confounded:
            score = clip01(score * CONFOUND_DISCOUNT)

    # Adaptive gain: normalise to the 99.9th percentile so one hot pixel cannot flatten the map,
    # and clamp so a near-clean file is not stretched into looking like a heatmap.
    hi = float(np.percentile(gray_err, 99.9))
    gain = scale if scale is not None else float(np.clip(160.0 / max(hi, 1e-6), MIN_GAIN, MAX_GAIN))
    vis_gray = np.clip(gray_err * gain, 0, 255).astype(np.uint8)
    heat = colormap_rgb(vis_gray.astype(np.float32) / 255.0, cv2.COLORMAP_INFERNO)
    visual = composite_overlay(native_rgb_u8, heat, alpha=0.55)

    details = {
        "applicable": True,
        "mean_error": round(mean_of_means, 5),
        "median_error": round(med, 5),
        "p99_block_error": round(p99_blocks, 5),
        "negligible_floor": NEGLIGIBLE_P99,
        "block_coefficient_of_variation": round(cov, 4),
        "hot_block_ratio": round(hot_ratio, 5),
        "hot_blocks": int(hot.sum()),
        "total_blocks": int(means.size),
        "largest_hot_cluster_fraction": round(clustered, 4),
        "detail_correlation_rho": round(float(rho), 4),
        "content_confounded": content_confounded,
        "resave_quality": resave_quality,
        "resave_quality_source": ("explicit" if quality else
                                  "estimated_from_quantisation_table" if est_jpeg_quality
                                  else "fallback_default"),
        "display_gain": round(gain, 2),
        "block_size": block,
        "negligible_error_suppressed": negligible,
        "overlay_meaningful": not negligible,
    }
    if negligible:
        details["note"] = (
            f"Re-encoding at this file's own JPEG quality (q{resave_quality}) reproduces it almost exactly, so "
            "every region sits on the same quantisation lattice. Nothing measurable; the overlay "
            "would only show amplified rounding noise."
        )
    elif content_confounded:
        details["note"] = (
            f"The bright region tracks local edge energy (rho={rho:.2f}), so it is at least partly "
            "explained by this area simply having more fine detail than the rest of the frame — "
            "a sharp subject against a smooth background produces the same signature as a "
            f"splice. Score discounted by {1.0 - CONFOUND_DISCOUNT:.0%}. Treat as weak evidence only."
        )
    return TechniqueResult(score, visual, details)


def _detail_correlation(gray_err: np.ndarray, rgb_u8: np.ndarray, block: int) -> float:
    """Spearman-style rank correlation between block error and block edge energy.

    Ranks rather than raw values, because both distributions are heavy-tailed and a single sharp
    edge would otherwise dominate a Pearson correlation.
    """
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    err_blocks, _, _ = block_stats(gray_err, block)
    edge_blocks, _, _ = block_stats(edges, block)
    if err_blocks.size < 9:
        return 0.0
    return float(_rank(err_blocks.ravel()) @ _rank(edge_blocks.ravel()))


def _rank(v: np.ndarray) -> np.ndarray:
    """Mean-centred, unit-norm ranks — the dot product of two of these is Spearman's rho."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    return ranks / norm if norm > 0 else ranks



def _largest_cluster_fraction(hot_mask: np.ndarray) -> float:
    """Fraction of hot blocks in the largest connected component.

    Distinguishes "the whole image is noisy" (many isolated hot blocks -> low score) from
    "one contiguous region recompressed differently" (one big component -> high score), which
    is the actual splice signature.
    """
    if hot_mask.sum() == 0:
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(hot_mask, connectivity=8)
    if n <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return clip01(largest / max(1.0, float(hot_mask.sum())) * 1.6)
