"""Noise-residual inconsistency.

A camera sensor imprint (PRNU) plus photon shot noise gives a photograph a roughly *uniform*
noise floor. A region pasted in from another image — or synthesised by a diffusion model —
carries a different noise floor, so per-block sigma becomes an outlier.

Estimator: high-pass residual ``gray - gaussian_blur(gray, sigma=1)``, then a robust per-block
sigma via median absolute deviation. MAD rather than standard deviation because edges and
texture would otherwise dominate and every textured image would look spliced.

Fixes over the prototype:

* The per-block double list comprehension is replaced by ``block_mad`` (vectorised reshape).
* Block size scales with the image, and ``analysis_scale`` is threaded through, so the detector
  keeps the same physical sensitivity regardless of input resolution.
* Added spatial clustering: isolated outlier blocks are texture, a contiguous outlier *region*
  is a splice.
* **The cluster term is gated on the outlier count.** ``largest_component / total_outliers``
  saturates at 1.0 whenever only a handful of blocks fire and happen to touch, which made a
  q32-compressed photograph with 10 stray blocks out of 1200 score 0.61 on the strength of
  clustering alone. A real splice covers many blocks; ten touching ones is texture.
* **A noise-floor usability gate.** Below roughly 1.5 grey levels of median residual there is no
  sensor noise left to compare — what remains is quantisation structure, whose block-to-block
  variation reflects content and compression rather than provenance.
* **A compression-confound flag.** At low JPEG quality, quantisation is itself block-based, so
  per-block noise estimates legitimately vary with local content. That is a real limitation, not
  a bug: the flag lets the fusion model learn to discount it, and the reliability score already
  penalises heavy compression independently.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.analysis.contract import TechniqueResult, clip01
from app.analysis.techniques._common import block_mad, colormap_rgb, composite_overlay

__all__ = ["run"]

BASE_BLOCK = 48
#: Reference width the base block size was tuned at.
REFERENCE_WIDTH = 1600
#: Fewest outlier blocks before spatial clustering is allowed to contribute.
MIN_CLUSTER_BLOCKS = 8
#: Median per-block residual sigma below which there is no usable sensor noise to compare.
MIN_USABLE_SIGMA = 1.5
#: Estimated JPEG quality at or below which quantisation itself dominates the residual.
CONFOUND_QUALITY = 55
#: Score multiplier when the finding is compression-confounded.
CONFOUND_DISCOUNT = 0.45


def run(gray_u8: np.ndarray, analysis_scale: float = 1.0,
        est_jpeg_quality: int | None = None) -> TechniqueResult:
    h, w = gray_u8.shape[:2]
    # Keep the block size physically constant across resolutions.
    block = max(16, min(96, int(round(BASE_BLOCK * max(w, h) / REFERENCE_WIDTH))))
    block = block // 8 * 8 or 16

    gray = gray_u8.astype(np.float32)
    # sigma=1 high-pass: removes structure, keeps sensor noise.
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)

    sigma = block_mad(residual, block)
    if sigma is None or sigma.size < 9:
        return TechniqueResult(
            0.0, None,
            {"note": "Image too small for block-wise noise analysis.",
             "block_size": block, "min_side": int(min(h, w))},
        )

    # Work in log space: noise sigma is multiplicative (shot noise scales with signal), so a
    # ratio, not a difference, is the right comparison.
    log_sigma = np.log(sigma + 1e-3)
    med = float(np.median(log_sigma))
    mad = float(np.median(np.abs(log_sigma - med))) + 0.05
    z = np.abs(log_sigma - med) / (1.4826 * mad)
    median_sigma = float(np.exp(med))

    outlier = z > 4.0
    n_outlier = int(outlier.sum())
    outlier_ratio = float(outlier.mean())
    clustered = _largest_cluster_fraction(outlier.astype(np.uint8))
    clustered *= min(1.0, n_outlier / MIN_CLUSTER_BLOCKS)

    # A genuinely uniform image has near-zero MAD, which makes z explode on rounding noise.
    # Guard: if the absolute spread of noise levels is tiny, nothing is inconsistent.
    absolute_spread = float(np.exp(log_sigma.max()) - np.exp(log_sigma.min()))
    low_contrast_noise = absolute_spread < 0.35
    no_noise_floor = median_sigma < MIN_USABLE_SIGMA
    confounded = (est_jpeg_quality is not None
                  and int(est_jpeg_quality) <= CONFOUND_QUALITY)

    if low_contrast_noise or no_noise_floor:
        score = 0.0
    else:
        score = clip01(0.42 * min(1.0, outlier_ratio * 9.0) + 0.58 * clustered)
        if confounded:
            score = clip01(score * CONFOUND_DISCOUNT)

    # Upscale the block z-grid to pixel resolution for the overlay.
    heat = cv2.resize(np.clip(z / 8.0, 0, 1).astype(np.float32), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    heat_rgb = colormap_rgb(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    visual = composite_overlay(base, heat_rgb, alpha=0.5)

    ny, nx = sigma.shape
    details = {
        "outlier_block_ratio": round(outlier_ratio, 5),
        "outlier_blocks": n_outlier,
        "total_blocks": int(sigma.size),
        "min_cluster_blocks": MIN_CLUSTER_BLOCKS,
        "largest_outlier_cluster_fraction": round(clustered, 4),
        "median_noise_sigma": round(median_sigma, 4),
        "min_usable_sigma": MIN_USABLE_SIGMA,
        "noise_floor_usable": not no_noise_floor,
        "noise_sigma_range": [round(float(np.exp(log_sigma.min())), 4),
                              round(float(np.exp(log_sigma.max())), 4)],
        "absolute_sigma_spread": round(absolute_spread, 4),
        "robust_z_threshold": 4.0,
        "block_size": block,
        "grid_shape": [int(ny), int(nx)],
        "analysis_scale": round(analysis_scale, 4),
        "low_contrast_noise": low_contrast_noise,
        "estimated_jpeg_quality": est_jpeg_quality,
        "compression_confounded": confounded,
        "overlay_meaningful": not (low_contrast_noise or no_noise_floor),
    }
    if low_contrast_noise:
        details["note"] = (
            "Noise floor is essentially uniform across the image; no region stands out. "
            "Score suppressed to avoid false positives on flat, heavily-compressed images."
        )
    elif no_noise_floor:
        details["note"] = (
            f"Median per-block noise is {median_sigma:.2f} grey levels, below the {MIN_USABLE_SIGMA:.1f} needed for this "
            "test to mean anything. At that level the residual is dominated by JPEG "
            "quantisation structure rather than sensor noise, so block-to-block variation "
            "reflects content and compression, not provenance. Reports nothing rather than "
            "guessing; the reliability score is reduced instead."
        )
    elif confounded:
        details["note"] = (
            f"This file was saved at roughly JPEG q{est_jpeg_quality}. Quantisation at that level is itself "
            "8x8-block-based, so per-block noise estimates vary with local content even in a "
            f"completely genuine photograph. Finding discounted by {1.0 - CONFOUND_DISCOUNT:.0%}; treat as weak "
            "evidence only."
        )
    return TechniqueResult(score, visual, details)


def _largest_cluster_fraction(mask: np.ndarray) -> float:
    """Fraction of flagged blocks belonging to the largest connected component."""
    if mask.sum() == 0:
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return clip01(largest / max(1.0, float(mask.sum())) * 1.6)

