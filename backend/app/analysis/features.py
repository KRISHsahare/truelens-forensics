"""The fusion feature vector — one definition, used by both training and inference.

This module exists as its own file for one reason: ``ml/scripts/calibrate_fusion.py`` and
``app/analysis/fusion.py`` must agree on the feature vector *exactly*, column-for-column and in
order. A saved model has no way to notice that column 14 changed meaning, and it will keep
producing confident numbers that mean something else. So the vector is defined here once, both
sides import it, and the model records the feature names it was trained on so a mismatch fails
loudly at load time instead of silently mis-scoring.

**The NaN rule.** A technique that could not run produces ``NaN``, never ``0.0``. That
distinction is the whole point: ``0.0`` means "we measured this and found nothing suspicious",
while ``NaN`` means "we do not know". Collapsing them is how a forensics tool quietly decides
that every PNG is authentic — ELA returns 0 on a PNG because it cannot run, and a model trained
on zeros would read that as evidence of cleanliness.

Because of that rule the vector carries two columns per technique: the score, and whether the
score means anything (``.measured``). Gradient-boosted models consume NaN natively; the logistic
fallback imputes NaN to 0 and relies on the ``.measured`` companion to recover the difference.
"""

from __future__ import annotations

import math
from typing import Any

from app.analysis.explainers import Explanation, explain
from app.analysis.registry import TECHNIQUES
from app.analysis.types import JobArrays

__all__ = ["CONTEXT_FEATURES", "FEATURES", "TECHNIQUE_FEATURES", "extract", "names", "vectorize"]

#: Per-technique columns. ``.score`` is the analyser's own output; the extras let the model see
#: *why* a score is what it is, and are the reason a learned fusion beats a weighted sum: an ELA
#: of 0.72 that is content-confounded is not the same measurement as an ELA of 0.72 that is not.
TECHNIQUE_FEATURES: dict[str, tuple[str, ...]] = {
    "metadata":  ("score", "measured", "generator_flag"),
    "c2pa":      ("score", "measured", "verified", "invalid"),
    "ela":       ("score", "measured", "confounded", "cluster_fraction", "hot_block_ratio"),
    "jpeg_ghost": ("score", "measured", "ghost_region_fraction", "peak_contrast"),
    "block_grid": ("score", "measured", "disagreeing_mass", "distinct_strong_phases"),
    "noise":     ("score", "measured", "confounded", "outlier_block_ratio", "cluster_fraction"),
    "copy_move": ("score", "measured", "inlier_log"),
    "face":      ("score", "measured", "faces_log"),
    "frequency": ("score", "measured", "smooth_tail", "depleted", "axis_score", "slope_norm"),
    "cnn":       ("score", "measured", "clip_score", "agreement_gap", "concur"),
}

#: File-level context the fusion needs to know about the *input*, not the findings. These are
#: never NaN: they come from the container, which is always available.
CONTEXT_FEATURES: tuple[str, ...] = (
    "log_min_side",
    "bpp_log",
    "is_jpeg",
    "round_dimensions",
    "resized",
    "jpeg_quality_norm",
)

TECHNIQUE_COLUMNS: tuple[str, ...] = tuple(
    f"{name}.{field}"
    for name in (s.name for s in TECHNIQUES)
    for field in TECHNIQUE_FEATURES[name]
)

#: Every column, in the order ``vectorize`` emits. Fixed at import; the saved model stores this
#: tuple and refuses to score if it differs.
FEATURES: tuple[str, ...] = TECHNIQUE_COLUMNS + CONTEXT_FEATURES


def names() -> tuple[str, ...]:
    return FEATURES


def _num(d: dict[str, Any], *keys: str, default: float = math.nan) -> float:
    """First present numeric key, coerced to float; ``None`` and non-numbers become NaN."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            return v if math.isfinite(v) else math.nan
    return default


def _ratio(d: dict[str, Any], key: str, default: float = math.nan) -> float:
    v = _num(d, key, default=default)
    return v if math.isnan(v) else max(0.0, min(1.0, v))


def _flag(d: dict[str, Any], key: str) -> float:
    """Boolean finding -> 1.0/0.0. Absent means False, not unknown: these keys are only omitted
    by branches that did not evaluate the condition."""
    return 1.0 if d.get(key) else 0.0


def _measured(explanation: Explanation | None, status: str) -> float:
    """1.0 when the technique produced a real measurement, 0.0 when it had nothing to say.

    Read from the ``Explanation`` rather than re-derived from ``details`` keys, on purpose. The
    explainer is the one place that interprets a result dict, and it already has to distinguish
    "we measured and found nothing" from "we could not measure" in order to write an honest
    sentence. Re-implementing that test here would create a second interpreter, and the two would
    drift — the model would train on one definition of "missing" while the UI displayed another.
    """
    if status != "ok":
        return 0.0
    if explanation is None:
        return 0.0
    return 0.0 if explanation.weight == "unavailable" else 1.0


def extract(
    outcomes: dict[str, Any],
    arrays: JobArrays,
    explanations: dict[str, Explanation] | None = None,
) -> dict[str, float]:
    """Build the named feature dict for one job.

    ``outcomes`` maps technique name -> ``TaskOutcome``. Techniques absent from it (skipped by
    the orchestrator, or a reduced-profile run) are emitted as NaN rather than omitted, so the
    column count never depends on which techniques ran.

    ``explanations`` should be the same ``Explanation`` objects the API returns, so the model and
    the reviewer agree on which numbers mean something. When it is not supplied — the training
    script building features straight from a fixture set, for instance — they are derived here by
    calling the explainer, never by re-implementing its judgement.
    """
    feats: dict[str, float] = {}

    for spec in TECHNIQUES:
        cols = TECHNIQUE_FEATURES[spec.name]
        oc = outcomes.get(spec.name)
        prefix = spec.name
        score_key, meas_key = f"{prefix}.score", f"{prefix}.measured"

        if oc is None or oc.status == "skipped":
            feats[score_key] = math.nan
            feats[meas_key] = 0.0
            for extra in cols[2:]:
                feats[f"{prefix}.{extra}"] = math.nan
            continue

        d = oc.details or {}
        expl = (explanations or {}).get(spec.name)
        if expl is None:
            expl = explain(spec.name, oc.score, d, status=oc.status)
        measured = _measured(expl, oc.status)
        # A score from a check that could not run is not a low score. It is no information.
        feats[score_key] = (float(oc.score) if measured else math.nan)
        feats[meas_key] = measured
        feats.update(_extras(spec.name, d))

    # --- context -------------------------------------------------------------
    min_side = max(1, arrays.min_side)
    feats["log_min_side"] = math.log10(min_side)
    feats["bpp_log"] = math.log10(max(1e-4, arrays.bpp))
    feats["is_jpeg"] = 1.0 if arrays.mime == "image/jpeg" else 0.0
    feats["round_dimensions"] = 1.0 if arrays.round_dimensions else 0.0
    # < 1.0 means the analysis track is a downscale, so native-resolution detail was already lost.
    feats["resized"] = 1.0 if arrays.analysis_scale < 0.999 else 0.0
    q = arrays.est_jpeg_quality
    feats["jpeg_quality_norm"] = (float(q) / 100.0) if q else math.nan

    return {k: (v if isinstance(v, float) else float(v)) for k, v in feats.items()}


def _extras(name: str, d: dict[str, Any]) -> dict[str, float]:
    """The technique-specific explanation columns."""
    g = lambda *k, default=math.nan: _num(d, *k, default=default)  # noqa: E731
    if name == "metadata":
        det = d.get("deterministic_flag")
        return {"metadata.generator_flag": 1.0 if det else 0.0}
    if name == "c2pa":
        status = d.get("status", "absent")
        return {"c2pa.verified": 1.0 if status == "verified" else 0.0,
                "c2pa.invalid": 1.0 if status == "invalid" else 0.0}
    if name == "ela":
        return {"ela.confounded": _flag(d, "content_confounded"),
                "ela.cluster_fraction": _ratio(d, "largest_hot_cluster_fraction"),
                "ela.hot_block_ratio": _ratio(d, "hot_block_ratio")}
    if name == "jpeg_ghost":
        return {"jpeg_ghost.ghost_region_fraction": _ratio(d, "ghost_region_fraction"),
                "jpeg_ghost.peak_contrast": _num(d, "peak_contrast")}
    if name == "block_grid":
        return {"block_grid.disagreeing_mass": _ratio(d, "disagreeing_mass"),
                "block_grid.distinct_strong_phases": g("distinct_strong_phases")}
    if name == "noise":
        return {"noise.confounded": _flag(d, "compression_confounded"),
                "noise.outlier_block_ratio": _ratio(d, "outlier_block_ratio"),
                "noise.cluster_fraction": _ratio(d, "largest_outlier_cluster_fraction")}
    if name == "copy_move":
        n = g("inliers")
        # Log-scale: 3 inliers and 300 inliers are not 100x apart in evidential terms.
        return {"copy_move.inlier_log": math.nan if math.isnan(n) else math.log1p(n)}
    if name == "face":
        n = g("faces")
        return {"face.faces_log": math.nan if math.isnan(n) else math.log1p(n)}
    if name == "frequency":
        slope = g("high_band_slope")
        axis = (d.get("axis_periodicity") or {})
        return {"frequency.smooth_tail": _flag(d, "smooth_tail"),
                "frequency.depleted": _flag(d, "high_frequency_depleted"),
                "frequency.axis_score": _ratio(axis, "score") if axis else math.nan,
                # Centred on the middle of the natural band so 0 means "looks like a photograph".
                "frequency.slope_norm": math.nan if math.isnan(slope) else (slope + 1.225)}
    if name == "cnn":
        agree = d.get("detector_agreement") or {}
        # Only the CLIP probe gets its own column: ``cnn.score`` is already the EfficientNet's
        # fused output, and repeating it would double-count one detector in the learned fusion.
        clip = g("clip_probe_p_fake")
        gap = agree.get("score_gap")
        return {"cnn.clip_score": clip,
                "cnn.agreement_gap": float(gap) if gap is not None else math.nan,
                "cnn.concur": 1.0 if agree.get("concur") else (math.nan if not agree else 0.0)}
    return {}


def vectorize(feats: dict[str, float], *, impute: bool = False) -> Any:
    """Ordered feature row matching ``FEATURES``.

    ``impute=True`` replaces NaN with 0.0 for the logistic model, which cannot consume NaN. The
    accompanying ``.measured`` columns survive imputation and are how the linear model still
    distinguishes "scored 0.2" from "did not run" — a reason those columns are mandatory rather
    than decorative.
    """
    import numpy as np

    missing = [c for c in FEATURES if c not in feats]
    if missing:  # pragma: no cover - guards against a feature added on one side only
        raise ValueError(f"feature vector missing columns: {missing[:6]}")
    row = np.array([feats[c] for c in FEATURES], dtype=np.float64)
    if impute:
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    return row
