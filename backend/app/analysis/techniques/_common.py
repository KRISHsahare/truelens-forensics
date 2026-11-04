"""Shared numeric helpers for techniques.

Import closure is deliberately ``numpy`` + ``cv2`` only, so pulling this into a spawn worker
stays cheap.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "block_mad",
    "block_stats",
    "colormap_rgb",
    "composite_overlay",
    "normalise01",
]


def block_stats(arr: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block mean and std, fully vectorised.

    The prototype used a nested list comprehension over every block, calling ``np.mean`` per
    block: ~125 ms of pure Python overhead on a 3 MP image. Reshaping to
    ``(ny, block, nx, block)`` and reducing over ``axis=(1, 3)`` puts block ``(iy, ix)`` at
    ``[iy, :, ix, :]`` and does the same work in ~20 ms.

    Returns ``(means, stds, grid_shape)``; grids are ``(ny, nx)``. Empty grids collapse to a
    single cell covering whatever pixels exist, so callers never see a zero-length array.
    """
    h, w = arr.shape[:2]
    ny, nx = h // block, w // block
    if ny == 0 or nx == 0:
        flat = arr.reshape(-1) if arr.ndim == 2 else arr.reshape(-1, arr.shape[2])
        return (np.array([[float(flat.mean())]]), np.array([[float(flat.std())]]), (1, 1))
    trimmed = arr[: ny * block, : nx * block]
    reshaped = trimmed.reshape(ny, block, nx, block) if arr.ndim == 2 else \
        trimmed.reshape(ny, block, nx, block, arr.shape[2])
    axes = (1, 3) if arr.ndim == 2 else (1, 3, 4)
    return reshaped.mean(axis=axes), reshaped.std(axis=axes), (ny, nx)


def block_mad(residual: np.ndarray, block: int) -> np.ndarray | None:
    """Robust per-block noise sigma via median absolute deviation.

    MAD is used instead of std because edges and texture would otherwise dominate the noise
    estimate. ``1.4826`` is the consistency factor making MAD a unbiased sigma estimator for
    Gaussian noise.
    """
    h, w = residual.shape[:2]
    ny, nx = h // block, w // block
    if ny == 0 or nx == 0:
        return None
    trimmed = residual[: ny * block, : nx * block].reshape(ny, block, nx, block)
    med = np.median(trimmed, axis=(1, 3), keepdims=True)
    abs_dev = np.abs(trimmed - med)
    return 1.4826 * np.median(abs_dev, axis=(1, 3))


def normalise01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to float32 [0, 1]; constant arrays collapse to 0."""
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def colormap_rgb(values01: np.ndarray, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Map [0,1] floats to a uint8 **RGB** image (cv2 returns BGR)."""
    u8 = np.clip(np.asarray(values01, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cmap), cv2.COLOR_BGR2RGB)


def composite_overlay(
    base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend a heatmap over the original image.

    Heatmaps are always composited rather than returned bare: a reviewer cannot localise a
    noise anomaly from an unanchored JET image, and a bare spectrum gives no spatial context.
    The FFT analyser is the one exception — its output *is* the frequency domain, so it is
    returned bare and the UI labels it as such.
    """
    if heat_rgb.shape[:2] != base_rgb.shape[:2]:
        heat_rgb = cv2.resize(
            heat_rgb, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    a = np.clip(alpha, 0.0, 1.0)
    return cv2.addWeighted(base_rgb, 1.0 - a, heat_rgb, a, 0.0)
