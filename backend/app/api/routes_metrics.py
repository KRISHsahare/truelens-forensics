"""``GET /api/metrics`` — the numbers behind the Methodology page.

Everything here is a *claim about the models*, not about any user's image: dataset composition, the
held-out scores, the confusion matrix, the ROC and precision-recall curves, and the fusion weights
that were fitted rather than hand-tuned. The frontend renders them without reformatting, so this
route's job is to pass them through faithfully and to be unambiguous about what is missing.

**A missing artifact is reported, never padded.** When ``metrics.json`` is absent the response says
so and explains why the page cannot make a claim. Substituting zeros would render a chart that looks
measured, and on a forensics product the difference between "we did not test this" and "this scored
0.0" is the whole ballgame.

The technique catalogue is served from the registry rather than duplicated into a JSON file, so the
Methodology page cannot drift from what the pipeline actually runs — a documented technique that no
longer exists is worse than no documentation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Request

from app.analysis.fusion import EXPERT_WEIGHTS
from app.analysis.registry import TECHNIQUES
from app.api.deps import get_settings

__all__ = ["router"]

log = logging.getLogger("truelens.api.metrics")

router = APIRouter(prefix="/api", tags=["metrics"])


@router.get("/metrics", summary="Model evaluation results, fusion weights and the technique catalogue")
async def metrics(request: Request) -> dict[str, Any]:
    settings = get_settings(request)
    out: dict[str, Any] = {
        "detectors": _detectors(settings),
        "fusion": _fusion(settings),
        "techniques": [
            {
                "name": s.name,
                "label": s.label,
                "group": s.group,
                "track": s.track,
                "overlay": s.overlay,
                "heavy": s.heavy,
                "summary": s.summary,
            }
            for s in TECHNIQUES
        ],
    }
    measured = _read_json(settings.metrics_path)
    if measured is None:
        out["evaluation"] = {
            "available": False,
            "reason": f"No metrics file at {settings.metrics_path}.",
            "hint": ("Train and export with ml/scripts/train_cnn.py, then copy the emitted "
                     "metrics.json into artifacts/."),
            # What the page must say instead of a chart. Kept as data so the frontend does not
            # hard-code its own copy of the excuse.
            "meaning": ("Without an evaluation file, no accuracy, AUC or calibration claim on this "
                        "page is supported by evidence."),
        }
    else:
        out["evaluation"] = {"available": True, **measured}
    return out


def _detectors(settings: Any) -> dict[str, Any]:
    """What is loaded, and what would load if the files were there.

    ``present`` is a filesystem fact and ``enabled`` is a configuration fact, and they are reported
    separately because the interesting deployments are the ones where they disagree: CLIP switched
    on with no probe file is a degraded detector that most status pages would call "on".
    """
    return {
        "efficientnet": {
            "path": settings.cnn_weights_path,
            "present": os.path.exists(settings.cnn_weights_path),
            "enabled": True,
        },
        "clip_probe": {
            "path": settings.clip_probe_path,
            "present": os.path.exists(settings.clip_probe_path),
            "enabled": bool(settings.enable_clip_probe),
        },
        "face_model": {
            "path": settings.yunet_model_path,
            "present": os.path.exists(settings.yunet_model_path),
            "enabled": True,
        },
    }


def _fusion(settings: Any) -> dict[str, Any]:
    """The fusion engine's provenance: which weights are in use and whether they were fitted.

    ``source`` mirrors what a result's ``fusion.source`` field says, so a reader comparing the
    Methodology page against an analysis can see the same story in both places.
    """
    path = settings.fusion_path
    present = bool(path) and os.path.exists(path)
    return {
        "artifact": path,
        "present": present,
        "source": "learned" if present else "expert",
        "calibrated": False,
        "note": (
            "A fitted fusion model is loaded, so the sub-scores are its output rather than a "
            "hand-weighted sum."
            if present else
            "No fitted fusion artifact was found, so the documented expert weights below are in "
            "use. They are a prior: sensible, explainable, and not evidence about how well they "
            "predict."
        ),
        "expert_weights": {k: float(v) for k, v in EXPERT_WEIGHTS.items()},
    }


def _read_json(path: str) -> dict[str, Any] | None:
    """Read an artifacts JSON file, or None.

    Malformed JSON is logged and reported as absent. A half-written metrics file is indistinguishable
    from no file for the purposes of what the page is allowed to claim, and parsing it into a partial
    dict would let a truncated ``auc`` become ``null`` on a chart that otherwise looks measured.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("metrics file %s unreadable: %s", path, type(exc).__name__)
        return None
    return data if isinstance(data, dict) else None
