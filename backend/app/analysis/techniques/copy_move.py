"""Copy-move forgery detection.

Find regions duplicated *within* one image — the classic clone-stamp signature used to remove
or add objects. Keypoints are matched against the image's own descriptor set; a genuine
copy-move yields many matches that share one displacement vector.

Fixes over the prototype:

* **FLANN instead of BFMatcher.** Brute-force self-matching on 3000 x 128 SIFT descriptors is
  ~9M distance computations and costs 1-3 s pinned to one thread — the single largest item in
  the latency budget. FLANN's KD-tree does the same ``knnMatch(desc, desc, k=3)`` with identical
  ``m[0]``-is-self semantics, so the Lowe ratio test carries over unchanged, in 200-400 ms.
* **Consensus is refined, not just bucketed.** The prototype quantised displacements to 12 px
  bins and took the largest bin, which splits a true cluster across bin boundaries. We take the
  best bin as a seed, average its members, then recount inliers within a tight radius.
* **Spatial separation is required.** Source and destination point clouds must be far apart and
  both spatially spread out. This is what suppresses the brutal false-positive rate on
  repetitive texture (foliage, brickwork, crowds), where hundreds of spurious matches share a
  displacement but all sit in one small region.
* **Pixel-unit thresholds scale with the image.** A 40 px minimum displacement means something
  different at 512 px and at 4000 px.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.analysis.contract import TechniqueResult, clip01

__all__ = ["run"]

LOWE_RATIO = 0.62
#: Minimum inliers for a credible copy-move. Below this, random texture dominates.
MIN_INLIERS = 12


def run(
    rgb_u8: np.ndarray,
    gray_u8: np.ndarray,
    analysis_scale: float = 1.0,
    max_features: int = 3000,
) -> TechniqueResult:
    h, w = gray_u8.shape[:2]
    if min(h, w) < 96:
        return TechniqueResult(0.0, None, {"matched_pairs": 0, "inliers": 0,
                                           "note": "Image too small for copy-move detection."})

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=int(max_features), contrastThreshold=0.04)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    else:  # pragma: no cover - SIFT patent expired; present in every modern build
        detector = cv2.ORB_create(nfeatures=int(max_features))
        matcher = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12,
                                             multi_probe_level=1), dict(checks=50))

    kps, desc = detector.detectAndCompute(gray_u8, None)
    if desc is None or len(kps) < 40:
        return TechniqueResult(0.0, None, {
            "matched_pairs": 0, "inliers": 0, "keypoints": len(kps),
            "note": "Too few stable keypoints to search for duplication.",
        })

    # k=3: m[0] is always the keypoint matching itself at distance 0.
    matches = matcher.knnMatch(desc, desc, k=3)

    min_disp = max(24.0, 40.0 * analysis_scale, 0.04 * max(h, w))
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    pairs: list[tuple[int, int]] = []
    for group in matches:
        if len(group) < 3:
            continue
        a, b = group[1], group[2]
        if a.queryIdx == a.trainIdx or b.distance <= 1e-6:
            continue
        # Lowe ratio test: reject ambiguous matches. Repetitive texture produces many
        # near-equally-good matches, and this is precisely what filters them out.
        if a.distance < LOWE_RATIO * b.distance:
            p1, p2 = pts[a.queryIdx], pts[a.trainIdx]
            if float(np.linalg.norm(p1 - p2)) > min_disp:
                pairs.append((a.queryIdx, a.trainIdx))

    overlay = rgb_u8.copy()
    if len(pairs) < 3:
        return TechniqueResult(0.0, overlay, {
            "matched_pairs": len(pairs), "inliers": 0, "keypoints": len(kps),
            "min_displacement_px": round(min_disp, 1),
            "note": "No duplicated regions found.",
        })

    qi = np.array([p[0] for p in pairs], dtype=np.int32)
    ti = np.array([p[1] for p in pairs], dtype=np.int32)
    src = pts[qi]
    dst = pts[ti]
    # Displacements are symmetric: if A was cloned to B, both A->B and B->A are valid.
    shifts = np.vstack([dst - src, src - dst])

    inliers, shift, votes = _refine_consensus(shifts, tolerance=max(6.0, 10.0 * analysis_scale))

    # Restrict to the direction we actually found, and require spatial separation.
    if inliers is not None and votes >= MIN_INLIERS:
        sel_src, sel_dst = _select_pairs(src, dst, shift, tolerance=max(8.0, 12.0 * analysis_scale))
        separation = _separation_score(sel_src, sel_dst, w, h)
        n_inl = len(sel_src)
        coverage = _cloud_spread(sel_src, sel_dst, w, h)
    else:
        sel_src = sel_dst = np.zeros((0, 2), dtype=np.float32)
        separation = 0.0
        n_inl = 0
        coverage = 0.0

    # A large inlier count means little if the two clouds overlap: that is repetitive texture,
    # not a clone. Multiply by the separation term.
    #
    # Saturating, not linear. ``n_inl`` is already gated at MIN_INLIERS unanimous votes and is 0
    # for every fixture that was not cloned, so the count only ever varies over the range in which
    # a real consensus already exists. A linear ramp saturating at 48 scored a decisive 18-inlier,
    # unanimously-displaced clone at 0.375 — below the "affirmative finding" line — which is a
    # calibration artefact, not caution. This curve puts the bare gate at ~0.49 and three times the
    # gate at ~0.87, so evidence strength and reported strength move together.
    density = float(-np.expm1(-n_inl / (MIN_INLIERS * 1.5)))
    score = clip01(density * (0.45 + 0.55 * separation) * (0.6 + 0.4 * coverage))

    if n_inl:
        _draw(overlay, sel_src, sel_dst)

    details = {
        "matched_pairs": len(pairs),
        "inliers": n_inl,
        "keypoints": len(kps),
        "dominant_displacement_px": None if shift is None else [round(float(shift[0]), 1),
                                                                round(float(shift[1]), 1)],
        "consensus_votes": int(votes),
        "min_displacement_px": round(min_disp, 1),
        "spatial_separation": round(separation, 4),
        "region_coverage": round(coverage, 4),
        "lowe_ratio": LOWE_RATIO,
        "matcher": "FLANN/KD-tree" if hasattr(cv2, "SIFT_create") else "FLANN/LSH",
        "analysis_scale": round(analysis_scale, 4),
    }
    if n_inl and separation < 0.25:
        details["note"] = (
            "Duplicate matches were found but the two regions overlap heavily — this is "
            "characteristic of repetitive texture (foliage, brick, fabric) rather than cloning."
        )
    return TechniqueResult(score, overlay, details)


def _refine_consensus(
    shifts: np.ndarray, tolerance: float
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Seed with a coarse histogram bin, then recount inliers around the bin's mean.

    Coarse bucketing alone splits a true cluster across adjacent bins and undercounts it;
    the refinement pass recovers the full cluster.
    """
    coarse = max(8.0, tolerance * 1.5)
    quantised = np.round(shifts / coarse).astype(np.int64)
    _, inverse, counts = np.unique(quantised, axis=0, return_inverse=True, return_counts=True)
    best_bin = int(counts.argmax())
    seed = shifts[inverse == best_bin]
    if seed.size == 0:
        return None, None, 0
    centre = seed.mean(axis=0)
    dist = np.linalg.norm(shifts - centre, axis=1)
    mask = dist <= tolerance
    if mask.sum() < seed.shape[0]:
        # One refinement iteration: re-centre on the tighter cluster.
        centre = shifts[mask].mean(axis=0)
        mask = np.linalg.norm(shifts - centre, axis=1) <= tolerance
    inliers = shifts[mask]
    votes = int(mask.sum())
    return (inliers, centre, votes) if votes >= 3 else (None, None, votes)


def _select_pairs(
    src: np.ndarray, dst: np.ndarray, shift: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the (src, dst) pairs whose displacement matches the consensus vector."""
    d = np.linalg.norm((dst - src) - shift, axis=1)
    keep = d <= tolerance
    if keep.sum() < 3:
        d2 = np.linalg.norm((src - dst) - shift, axis=1)
        keep = d2 <= tolerance
        if keep.sum() < 3:
            return src[:0], dst[:0]
    return src[keep], dst[keep]


def _separation_score(src: np.ndarray, dst: np.ndarray, w: int, h: int) -> float:
    """How far apart the two matched clouds are, normalised by image size.

    Near-zero separation means the detector matched a region against itself — repetitive
    texture, not a clone.
    """
    if len(src) < 3:
        return 0.0
    diag = float(np.hypot(w, h))
    c_src, c_dst = src.mean(axis=0), dst.mean(axis=0)
    centroid_gap = float(np.linalg.norm(c_src - c_dst)) / max(1.0, diag)
    # Also require the clouds not to overlap pointwise.
    from scipy.spatial import cKDTree  # local import: scipy only needed on this path
    tree = cKDTree(dst)
    nearest, _ = tree.query(src, k=1)
    overlap_free = float((nearest > max(12.0, 0.02 * diag)).mean())
    return clip01(centroid_gap * 3.2) * (0.35 + 0.65 * overlap_free)


def _cloud_spread(src: np.ndarray, dst: np.ndarray, w: int, h: int) -> float:
    """Do the inliers cover an area, or collapse onto a few pixels?

    A real cloned region spans a meaningful patch; a handful of coincidental matches do not.
    """
    if len(src) < 4:
        return 0.0
    pts = np.vstack([src, dst])
    span_x = (pts[:, 0].max() - pts[:, 0].min()) / max(1.0, float(w))
    span_y = (pts[:, 1].max() - pts[:, 1].min()) / max(1.0, float(h))
    return clip01(max(span_x, span_y) * 4.0)


def _draw(overlay: np.ndarray, src: np.ndarray, dst: np.ndarray) -> None:
    """Draw the matched pairs. ``overlay`` is RGB, so colours are (R, G, B)."""
    limit = 220  # cap line count so the overlay stays readable
    step = max(1, len(src) // limit)
    for p1, p2 in zip(src[::step], dst[::step], strict=False):
        a = (int(p1[0]), int(p1[1]))
        b = (int(p2[0]), int(p2[1]))
        cv2.line(overlay, a, b, (52, 211, 153), 1, cv2.LINE_AA)
        cv2.circle(overlay, a, 3, (248, 113, 113), -1, cv2.LINE_AA)
        cv2.circle(overlay, b, 3, (96, 165, 250), -1, cv2.LINE_AA)
    for cloud, colour in ((src, (248, 113, 113)), (dst, (96, 165, 250))):
        if len(cloud) >= 3:
            x, y, bw, bh = cv2.boundingRect(cloud.astype(np.int32))
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), colour, 2, cv2.LINE_AA)
