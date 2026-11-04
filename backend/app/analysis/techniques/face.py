"""Face-specific manipulation checks.

Replaces the prototype's Haar cascade with YuNet and adds landmark-driven checks. Four
independent signals per face, combined by taking the worst face (a single swapped face in a
group photo is the case that matters):

1. **Sharpness mismatch** — face region vs the surrounding background. A pasted face is usually
   resampled, so its high-frequency content differs from the plate.
2. **Blending seam** — gradient energy along the face ellipse vs immediately inside/outside it.
   Poisson-blended and GAN-composited faces leave a ring of elevated gradient at the blend
   boundary. We test the *ellipse*, not the bounding box: the box perimeter crosses hair and
   background edges that are legitimately sharp, which is why the naive version false-positives.
3. **Catchlight consistency** — the specular reflection of the light source in each eye. Both
   eyes of a real person show a highlight at a consistent relative position and intensity.
   Inpainted and swapped eyes frequently disagree, or lose the catchlight entirely.
4. **Lighting-direction asymmetry** — the sign of the horizontal luminance gradient in the two
   peri-ocular regions. One light source implies the same sign on both sides; opposite signs
   mean the two halves were lit differently.

Limitation stated plainly: YuNet provides 5 landmarks (eye *centres*, not corners or lids).
That is enough for these checks at reduced precision, and not enough for sub-pixel gaze
estimation. Each check therefore contributes modestly rather than decisively, and a missing
face lowers the reliability score instead of being read as "clean".
"""

from __future__ import annotations

import cv2
import numpy as np

from app.analysis.contract import TechniqueResult, clip01
from app.ml.facenet import Face, FaceDetectorUnavailable, detect_faces

__all__ = ["run"]

WEIGHTS = {"sharpness": 0.26, "seam": 0.30, "catchlight": 0.22, "lighting": 0.22}


def run(rgb_u8: np.ndarray, gray_u8: np.ndarray, model_path: str | None = None) -> TechniqueResult:
    kwargs = {"model_path": model_path} if model_path else {}
    try:
        faces = detect_faces(rgb_u8, **kwargs)
    except FaceDetectorUnavailable as exc:
        return TechniqueResult(0.0, None, {
            "faces": 0, "detector": "yunet", "available": False,
            "note": f"Face detection unavailable ({exc}). Face checks were skipped; this "
                    "lowers the reliability score rather than counting as clean.",
        })

    if not faces:
        return TechniqueResult(0.0, None, {
            "faces": 0, "detector": "yunet", "available": True,
            "note": "No face detected. Face-specific checks were skipped; this lowers the "
                    "reliability score rather than counting as evidence of authenticity.",
        })

    gray = gray_u8.astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    overlay = rgb_u8.copy()
    per_face: list[dict] = []
    face_scores: list[float] = []

    for face in faces[:6]:
        checks = {
            "sharpness": _sharpness_mismatch(gray, laplacian, face),
            "seam": _blending_seam(grad_mag, face),
            "catchlight": _catchlight_consistency(gray, face),
            "lighting": _lighting_asymmetry(gray, gx, face),
        }
        parts: list[dict] = []
        total = 0.0
        for name, res in checks.items():
            total += WEIGHTS[name] * res["score"]
            parts.append({"check": name, **{k: v for k, v in res.items()}})
        score = clip01(total)
        face_scores.append(score)
        per_face.append({
            "box": list(face.box),
            "detection_score": round(face.score, 4),
            "landmarks": {k: [round(v[0], 1), round(v[1], 1)]
                          for k, v in face.landmarks.items()},
            "score": round(score, 4),
            "checks": parts,
            "dominant_signal": max(checks, key=lambda k: WEIGHTS[k] * checks[k]["score"]),
        })
        _draw_face(overlay, face, score)

    details = {
        "faces": len(faces),
        "detector": "yunet",
        "available": True,
        "face_scores": [round(s, 4) for s in face_scores],
        "per_face": per_face,
        "weights": WEIGHTS,
        "landmark_count": 5,
    }
    if len(faces) > 1:
        details["note"] = (
            f"{len(faces)} faces detected. The reported score is the worst individual face, "
            "because a single swapped face is the case that matters."
        )
    return TechniqueResult(max(face_scores), overlay, details)


# --------------------------------------------------------------------- checks
def _face_region_mask(shape: tuple[int, int], face: Face, pad: float = 0.0) -> np.ndarray:
    """Elliptical mask inscribed in the (optionally padded) face box."""
    h, w = shape[:2]
    x, y, fw, fh = face.box
    px, py = int(fw * pad), int(fh * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + fw + px), min(h, y + fh + py)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2),
                ((x1 - x0) // 2 - 1, (y1 - y0) // 2 - 1), 0, 0, 360, 255, -1)
    full = np.zeros((h, w), dtype=np.uint8)
    full[y0:y1, x0:x1] = mask
    return full


def _sharpness_mismatch(gray: np.ndarray, laplacian: np.ndarray, face: Face) -> dict:
    """Compare high-frequency energy inside the face ellipse to a background ring around it."""
    inner = _face_region_mask(gray.shape, face, pad=0.0)
    outer = _face_region_mask(gray.shape, face, pad=0.35)
    ring = cv2.subtract(outer, inner)
    v_in = _masked_var(laplacian, inner)
    v_out = _masked_var(laplacian, ring)
    if v_in is None or v_out is None or v_out <= 1e-6:
        return {"score": 0.0, "note": "insufficient pixels", "ratio": None}
    ratio = v_in / v_out
    # Either direction is suspicious: much smoother (upscaled/regenerated) or much sharper
    # (pasted from a higher-resolution source) than the surrounding plate.
    score = clip01(abs(np.log(max(ratio, 1e-6))) / 1.9)
    return {"score": score, "laplacian_var_inside": round(float(v_in), 2),
            "laplacian_var_ring": round(float(v_out), 2), "ratio": round(float(ratio), 4)}


def _blending_seam(grad_mag: np.ndarray, face: Face) -> dict:
    """Gradient energy on the blend boundary vs immediately inside and outside it."""
    h, w = grad_mag.shape[:2]
    x, y, fw, fh = face.box
    band = max(2, int(0.045 * max(fw, fh)))

    core = _face_region_mask((h, w), face, pad=0.0)
    # Dilate/erode the ellipse to build the three concentric bands.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    dilated = cv2.dilate(core, k)
    eroded = cv2.erode(core, k)
    ring = cv2.subtract(dilated, eroded)
    inside = eroded
    outside = cv2.subtract(dilated, core)

    g_ring = _masked_mean(grad_mag, ring)
    g_in = _masked_mean(grad_mag, inside)
    g_out = _masked_mean(grad_mag, outside)
    if g_ring is None or g_in is None or g_out is None:
        return {"score": 0.0, "note": "insufficient pixels"}
    baseline = (g_in + g_out) / 2.0 + 1e-6
    excess = g_ring / baseline - 1.0
    # A natural face boundary (hair, jaw against background) also has high gradient. What
    # distinguishes a *blend seam* is that it exceeds BOTH sides symmetrically and by a lot.
    symmetry = 1.0 - abs(g_in - g_out) / (g_in + g_out + 1e-6)
    score = clip01(max(0.0, excess - 0.45) / 1.35) * (0.55 + 0.45 * symmetry)
    return {"score": score, "gradient_on_seam": round(float(g_ring), 3),
            "gradient_inside": round(float(g_in), 3),
            "gradient_outside": round(float(g_out), 3),
            "excess_ratio": round(float(excess), 4),
            "band_width_px": int(band), "side_symmetry": round(float(symmetry), 4)}


def _catchlight_consistency(gray: np.ndarray, face: Face) -> dict:
    """Compare the specular highlight position and intensity between the two eyes."""
    h, w = gray.shape[:2]
    re = face.landmarks.get("right_eye")
    le = face.landmarks.get("left_eye")
    if not re or not le:
        return {"score": 0.0, "note": "eye landmarks unavailable"}
    s = int(np.clip(0.20 * face.w, 11, 63))
    s |= 1  # odd

    def eye_patch(center: tuple[float, float]) -> np.ndarray | None:
        cx, cy = int(round(center[0])), int(round(center[1]))
        half = s // 2
        x0, y0, x1, y1 = cx - half, cy - half, cx + half + 1, cy + half + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        return gray[y0:y1, x0:x1]

    p_re, p_le = eye_patch(re), eye_patch(le)
    if p_re is None or p_le is None or p_re.size < 9 or p_le.size < 9:
        return {"score": 0.0, "note": "eye patches fall outside the frame"}

    def highlight(patch: np.ndarray) -> tuple[float, float, float, float]:
        smooth = cv2.GaussianBlur(patch, (3, 3), 0)
        _, _, _, max_loc = cv2.minMaxLoc(smooth)
        peak = float(smooth[max_loc[1], max_loc[0]])
        mean = float(smooth.mean())
        base = float(np.percentile(smooth, 25)) + 1e-6
        # Normalised offset of the highlight from the patch centre, in [-1, 1].
        ox = (max_loc[0] - patch.shape[1] / 2) / (patch.shape[1] / 2)
        oy = (max_loc[1] - patch.shape[0] / 2) / (patch.shape[0] / 2)
        return ox, oy, (peak - base) / base, mean

    ox1, oy1, c1, m1 = highlight(p_re)
    ox2, oy2, c2, m2 = highlight(p_le)

    pos_delta = float(np.hypot(ox1 - ox2, oy1 - oy2))
    contrast_ratio = float(min(c1, c2) / (max(c1, c2) + 1e-6))
    # Missing catchlight in one eye only. Both missing is common in low light and is not
    # by itself suspicious.
    one_missing = (c1 < 0.35) != (c2 < 0.35)

    pos_score = clip01(max(0.0, pos_delta - 0.42) / 0.9)
    contrast_score = clip01(max(0.0, 0.62 - contrast_ratio) / 0.62) if max(c1, c2) > 0.35 else 0.0
    score = clip01(0.52 * pos_score + 0.30 * contrast_score + (0.18 if one_missing else 0.0))
    return {
        "score": score,
        "catchlight_offset_delta": round(pos_delta, 4),
        "right_eye": {"offset": [round(ox1, 3), round(oy1, 3)],
                      "contrast": round(c1, 4), "mean_luma": round(m1, 2)},
        "left_eye": {"offset": [round(ox2, 3), round(oy2, 3)],
                     "contrast": round(c2, 4), "mean_luma": round(m2, 2)},
        "intensity_ratio": round(contrast_ratio, 4),
        "one_eye_missing_catchlight": bool(one_missing),
        "patch_size_px": int(s),
    }


def _lighting_asymmetry(gray: np.ndarray, gx: np.ndarray, face: Face) -> dict:
    """Do both halves of the face agree about which direction the light comes from?"""
    h, w = gray.shape[:2]
    x, y, fw, fh = face.box
    ew, eh = int(fw * 0.26), int(fh * 0.16)
    re = face.landmarks.get("right_eye")
    le = face.landmarks.get("left_eye")
    if not re or not le or ew < 3 or eh < 3:
        return {"score": 0.0, "note": "eye landmarks unavailable or face too small"}

    def box_grad(center: tuple[float, float]) -> tuple[float, float] | None:
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0, y0 = max(0, cx - ew // 2), max(0, cy - eh // 2)
        x1, y1 = min(w, x0 + ew), min(h, y0 + eh)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return float(gx[y0:y1, x0:x1].mean()), float(gray[y0:y1, x0:x1].mean())

    g_re = box_grad(re)
    g_le = box_grad(le)
    if g_re is None or g_le is None:
        return {"score": 0.0, "note": "peri-ocular regions fall outside the frame"}

    grad_r, luma_r = g_re
    grad_l, luma_l = g_le
    # Gradient magnitude must be meaningful on both sides before the sign comparison says
    # anything; a flat, low-contrast face gives noisy signs.
    strength = min(abs(grad_r), abs(grad_l))
    if strength < 0.35:
        return {"score": 0.0, "note": "insufficient shading gradient to infer light direction",
                "gradient_right": round(grad_r, 4), "gradient_left": round(grad_l, 4)}

    opposite_sign = (grad_r * grad_l) < 0
    # How strongly opposed, normalised.
    agreement = (grad_r * grad_l) / (abs(grad_r) * abs(grad_l) + 1e-6)  # cosine in 1-D: +/-1
    luma_gap = abs(luma_r - luma_l) / (max(luma_r, luma_l) + 1e-6)

    score = 0.0
    if opposite_sign:
        score = clip01(0.55 * min(1.0, abs(agreement)) + 0.45 * min(1.0, luma_gap * 3.0))
    details = {
        "score": score,
        "gradient_right": round(grad_r, 4),
        "gradient_left": round(grad_l, 4),
        "sign_agreement": round(float(agreement), 4),
        "luma_right": round(luma_r, 2),
        "luma_left": round(luma_l, 2),
        "luminance_gap": round(float(luma_gap), 4),
        "opposite_lighting_direction": bool(opposite_sign),
        "region_size_px": [int(ew), int(eh)],
    }
    return details


# --------------------------------------------------------------------- helpers
def _masked_var(arr: np.ndarray, mask: np.ndarray) -> float | None:
    vals = arr[mask > 0]
    return float(vals.var()) if vals.size >= 32 else None


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float | None:
    vals = arr[mask > 0]
    return float(vals.mean()) if vals.size >= 16 else None


def _draw_face(overlay: np.ndarray, face: Face, score: float) -> None:
    """Draw the box, landmarks and per-face score. Overlay is RGB."""
    x, y, fw, fh = face.box
    colour = ((52, 211, 153) if score < 0.35 else
              (251, 191, 36) if score < 0.60 else (248, 113, 113))
    cv2.rectangle(overlay, (x, y), (x + fw, y + fh), colour, 2, cv2.LINE_AA)
    cv2.ellipse(overlay, (x + fw // 2, y + fh // 2), (fw // 2, fh // 2), 0, 0, 360, colour, 1,
                cv2.LINE_AA)
    for pt in face.landmarks.values():
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, (96, 165, 250), -1, cv2.LINE_AA)
    label = f"{score:.2f}"
    ty = y - 8 if y - 8 > 12 else y + fh + 18
    cv2.putText(overlay, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 3,
                cv2.LINE_AA)
    cv2.putText(overlay, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
