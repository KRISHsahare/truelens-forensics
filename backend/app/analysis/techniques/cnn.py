"""Learned detector: EfficientNet-B0, plus the optional CLIP linear probe.

**Parent-process only.** This technique imports torch, so it must never be dispatched to the
process pool: on Windows spawn each child would re-import torch (~400 MB, 1.5-3 s) and the pool
would take 8-24 s to build. It runs on a dedicated single-thread executor in the parent, where
the model was loaded once at startup.

Grad-CAM is **lazy** by default. It costs a full backward pass (~1.8x a forward) for a visual
the reviewer often does not need, so it is only computed when the AI score clears a threshold
or when the caller explicitly asks for it.

The two detectors are reported separately *and* with their agreement. They have deliberately
different inductive biases — a high-capacity fine-tuned CNN that can memorise a generator's
fingerprint, versus a linear probe on frozen CLIP features that cannot — so concurrence is more
informative than either number alone, and disagreement is an honest signal that the input is
out of distribution for at least one of them.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.analysis.contract import TechniqueResult, clip01
from app.analysis.techniques._common import colormap_rgb, composite_overlay

__all__ = ["run"]

GRADCAM_ALPHA = 0.50


def run(
    rgb_u8: np.ndarray,
    *,
    tta: bool = True,
    want_gradcam: bool = False,
    gradcam_threshold: float = 0.45,
    weights_path: str | None = None,
    clip_probe_path: str | None = None,
    use_clip: bool = False,
) -> TechniqueResult:
    """``weights_path`` comes from config: the runtime is a process-wide singleton and must be
    built against the path the deployment actually uses, not a module default."""
    from app.ml.clip_probe import get_probe
    from app.ml.cnn_runtime import get_runtime

    runtime = get_runtime(weights_path)
    effnet = runtime.score(runtime.encode_for_resize(rgb_u8), tta=tta) if runtime.available else None

    clip_score = None
    clip_reason = None
    if use_clip:
        # ``enabled`` is what gates the load, and the caller has already decided: passing the config
        # flag through as the path alone would leave a configured probe permanently disabled.
        probe = get_probe(clip_probe_path, enabled=True)
        clip_score = probe.score(rgb_u8)
        if clip_score is None:
            clip_reason = probe.load_error

    if effnet is None and clip_score is None:
        reasons = [r for r in (runtime.load_error, clip_reason) if r]
        return TechniqueResult(0.0, None, {
            "available": False,
            "reason": "; ".join(reasons) or "no detector weights loaded",
            "note": "The learned detector is not loaded, so this technique contributed nothing "
                    "to the verdict. That is recorded as NaN in the fusion features and lowers "
                    "the reliability score — it is NOT treated as evidence of authenticity.",
            "hint": "Train weights with ml/scripts/train_cnn.py and place them at the path in "
                    "CNN_WEIGHTS_PATH.",
        })

    # When only one detector is present, use it. When both are, the primary score stays
    # EfficientNet's — it is the calibrated one — and CLIP is reported as corroboration.
    score = effnet if effnet is not None else clip_score

    agreement = None
    if effnet is not None and clip_score is not None:
        gap = abs(effnet - clip_score)
        agreement = {
            "score_gap": round(gap, 4),
            # Both on the same side of 0.5 and within 0.25 of each other.
            "concur": bool(gap <= 0.25 and ((effnet >= 0.5) == (clip_score >= 0.5))),
            "interpretation": (
                "The fine-tuned CNN and the frozen-CLIP linear probe agree, which is stronger "
                "evidence than either alone: they have very different capacities and training "
                "objectives."
                if gap <= 0.25 and (effnet >= 0.5) == (clip_score >= 0.5)
                else
                "The two detectors disagree. At least one is out of distribution for this "
                "image; treat the learned signal as weak and rely on the classical techniques."
            ),
        }

    visual = None
    gradcam_info: dict = {"computed": False}
    if runtime.available and (want_gradcam or (effnet is not None and effnet >= gradcam_threshold)):
        cam = runtime.gradcam(runtime.encode_for_resize(rgb_u8))
        if cam is not None:
            h, w = rgb_u8.shape[:2]
            cam_full = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
            heat = colormap_rgb(cam_full, cv2.COLORMAP_JET)
            visual = composite_overlay(rgb_u8, heat, alpha=GRADCAM_ALPHA)
            gradcam_info = {
                "computed": True,
                "activation_grid": list(cam.shape),
                "coverage_fraction": round(float((cam > 0.5).mean()), 4),
                "peak_intensity": round(float(cam.max()), 4),
                "mean_intensity": round(float(cam.mean()), 4),
                "ms": round(runtime.last_gradcam_ms, 1),
                "alpha": GRADCAM_ALPHA,
                "note": "Bright regions are where the network found the most evidence. This is "
                        "an attribution of the model's decision, not a map of ground truth.",
            }

    details = {
        "available": True,
        "model": "EfficientNet-B0 (1-logit, BCEWithLogitsLoss)" if effnet is not None else None,
        "efficientnet_p_fake": None if effnet is None else round(effnet, 5),
        "clip_probe_p_fake": None if clip_score is None else round(clip_score, 5),
        "clip_probe_enabled": bool(use_clip),
        "detector_agreement": agreement,
        "test_time_augmentation": bool(tta) and effnet is not None,
        "gradcam": gradcam_info,
        "input_size": int(runtime.input_size) if runtime.available else None,
        "resize_mode": "letterbox" if runtime.available else None,
    }
    return TechniqueResult(clip01(score), visual, details)
