"""8x8 DCT block-grid alignment.

JPEG encodes on an 8x8 grid starting at pixel (0, 0). Quantisation makes each block's edges
slightly discontinuous with its neighbours, so the absolute horizontal difference
``|I[x+1] - I[x]|`` averaged over a column spikes when ``x`` is the last column of a block.
Summing that profile by residue class mod 8 reveals the grid's **phase** — which of the 8
possible offsets the file's blocks actually sit on.

Every region of a genuine single-source JPEG shares one phase, because the encoder laid the
whole grid down at once. A pasted region came from a *different* file with a *different* phase
and a different crop offset, so its blocks do not line up with the plate. Detecting two or more
strongly-prominent phases in one image is direct evidence of splicing.

Runs on the **native** track: any resample shifts the grid by a sub-pixel amount and destroys
the phase. Our native crop is 8-aligned for exactly this reason.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.analysis.contract import TechniqueResult, clip01
from app.analysis.techniques._common import colormap_rgb, composite_overlay

__all__ = ["run"]

GRID = 8
#: Minimum peak prominence (over the mean of the 8 residues) for a phase to count as real.
PROMINENCE_THRESHOLD = 0.10


def run(native_gray_u8: np.ndarray, mime: str = "image/jpeg",
        *, regions: int = 4) -> TechniqueResult:
    if mime != "image/jpeg":
        return TechniqueResult(0.0, None, {
            "applicable": False,
            "note": "Source is not a JPEG, so it has no 8x8 DCT block grid to align. Skipped; "
                    "this does not count as evidence of authenticity.",
        })

    h, w = native_gray_u8.shape[:2]
    if min(h, w) < 128:
        return TechniqueResult(0.0, None, {
            "applicable": True,
            "note": "Image too small for reliable block-grid phase estimation.",
            "min_side": int(min(h, w)),
        })

    gray = native_gray_u8.astype(np.float32)
    global_profile = _phase_profile(gray)
    global_phase, global_prom = _peak_phase(global_profile)

    per_region: list[dict] = []
    phase_votes: dict[int, float] = {}
    region_phase_map = np.zeros((regions, regions), dtype=np.float32)

    rh = (h // regions) // GRID * GRID
    rw = (w // regions) // GRID * GRID
    if rh < 64 or rw < 64:
        rh, rw, regions = h // GRID * GRID, w // GRID * GRID, 1

    for iy in range(regions):
        for ix in range(regions):
            y0, x0 = iy * rh, ix * rw
            patch = gray[y0:y0 + rh, x0:x0 + rw]
            if patch.shape[0] < 64 or patch.shape[1] < 64:
                continue
            prof = _phase_profile(patch)
            phase, prom = _peak_phase(prof)
            region_phase_map[iy, ix] = phase
            weight = float(prom) if prom >= PROMINENCE_THRESHOLD else 0.0
            per_region.append({
                "region": [ix, iy],
                "phase": int(phase),
                "prominence": round(float(prom), 4),
                "matches_global": phase == global_phase,
                "counted": weight > 0,
            })
            if weight > 0:
                phase_votes[phase] = phase_votes.get(phase, 0.0) + weight

    total_weight = sum(phase_votes.values()) + 1e-9
    fractions = {p: v / total_weight for p, v in sorted(phase_votes.items())}
    disagreeing = [p for p, f in fractions.items() if p != global_phase and f >= 0.10]
    disagree_mass = sum(fractions[p] for p in disagreeing)
    distinct_phases = len([f for f in fractions.values() if f >= 0.10])

    # A weak global phase means there is no grid to misalign with — typical of PNG-origin or
    # very-low-quality content. Suppress rather than report noise as a finding.
    if global_prom < PROMINENCE_THRESHOLD:
        score = 0.0
        suppressed = True
    else:
        suppressed = False
        score = clip01(
            0.62 * min(1.0, disagree_mass / 0.35)
            + 0.38 * clip01((distinct_phases - 1) / 2.0)
        )

    visual = _render_overlay(native_gray_u8, region_phase_map, global_phase, regions, rh, rw)

    details = {
        "applicable": True,
        "global_phase": int(global_phase),
        "global_prominence": round(float(global_prom), 4),
        "prominence_threshold": PROMINENCE_THRESHOLD,
        "phase_distribution": {str(p): round(f, 4) for p, f in fractions.items()},
        "disagreeing_phases": [int(p) for p in disagreeing],
        "disagreeing_mass": round(float(disagree_mass), 4),
        "distinct_strong_phases": int(distinct_phases),
        "regions_analysed": len(per_region),
        "region_grid": [int(regions), int(regions)],
        "per_region": per_region[:24],
        "suppressed": suppressed,
        "note": ("No clear block grid was found, so phase alignment cannot be assessed."
                 if suppressed else
                 "Phase is the residue class (mod 8) at which 8x8 DCT block boundaries fall."),
    }
    return TechniqueResult(score, visual, details)


def _phase_profile(gray: np.ndarray) -> np.ndarray:
    """Sum the edge-discontinuity energy by column residue mod 8, and likewise for rows."""
    # Column profile: d[x] spans pixels x and x+1, so a block boundary at x=7..8 shows at x=7.
    dcol = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    drow = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    n_c = (len(dcol) // GRID) * GRID
    n_r = (len(drow) // GRID) * GRID
    if n_c < GRID or n_r < GRID:
        return np.zeros(GRID, dtype=np.float64)
    col_prof = dcol[:n_c].reshape(-1, GRID).mean(axis=0)
    row_prof = drow[:n_r].reshape(-1, GRID).mean(axis=0)
    return (col_prof + row_prof) / 2.0


def _peak_phase(profile: np.ndarray) -> tuple[int, float]:
    """Return the dominant phase and its prominence over the profile mean."""
    if profile.size < GRID or profile.sum() <= 0:
        return 0, 0.0
    mean = float(profile.mean())
    peak = int(np.argmax(profile))
    prominence = (float(profile[peak]) - mean) / (mean + 1e-9)
    return peak, prominence


def _render_overlay(gray: np.ndarray, phase_map: np.ndarray, global_phase: int,
                    regions: int, rh: int, rw: int) -> np.ndarray:
    """Highlight regions whose grid phase disagrees with the global one."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    mask = (phase_map != global_phase).astype(np.float32)
    if mask.sum() == 0:
        return base
    heat = np.zeros_like(gray, dtype=np.float32)
    for iy in range(regions):
        for ix in range(regions):
            if mask[iy, ix] > 0:
                heat[iy * rh:(iy + 1) * rh, ix * rw:(ix + 1) * rw] = 1.0
    heat_rgb = colormap_rgb(heat, cv2.COLORMAP_AUTUMN)
    return composite_overlay(base, heat_rgb, alpha=0.40)
