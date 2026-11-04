"""Evaluation output: the three JSON files the Methodology page reads.

``metrics.json`` is a *contract*, not a log. ``frontend/src/lib/types.ts`` declares an
``Evaluation`` interface and the page renders its fields without reformatting them, so a key that
is missing, misnamed, or holding a ``numpy.float64`` that survived into JSON as ``NaN`` shows up
as a blank chart or a literal ``null%``. ``validate()`` therefore checks the shape before a file
is written, and ``backend/tests/test_ml_pipeline.py`` asserts the writer's output against the
TypeScript interface's required fields.

**NaN is never written as a number.** scikit-learn returns ``nan`` for precision on a split with
no predicted positives, and JSON has no NaN. Writing ``NaN`` produces a file ``json.load`` accepts
only with ``allow_nan=True`` and that ``JSON.parse`` rejects outright — the page then shows the
"no metrics" state for a file that exists. Every metric passes through ``_plain()``, which turns a
non-finite value into ``None``; the frontend already renders ``None`` as "not measured" rather
than as zero, which is the distinction the whole product is built on.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Sequence

__all__ = [
    "confusion_payload",
    "ece",
    "metrics_payload",
    "roc_payload",
    "pr_payload",
    "validate",
    "write_evaluation",
]


def _plain(value: Any) -> float | None:
    """A JSON-safe float, or None for anything not finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(value: Any, digits: int = 4) -> float | None:
    v = _plain(value)
    return None if v is None else round(v, digits)


def metrics_payload(y_true: Sequence[int], proba: Sequence[float], *, threshold: float) -> dict[str, Any]:
    """Threshold-dependent scores plus the two rank-based ones.

    ``accuracy`` and the rest are reported *at the chosen threshold*, and ``threshold`` travels
    with them: an accuracy figure without its operating point is a number about a decision that
    was never made. ``roc_auc`` is threshold-free and therefore the only figure here that survives
    a change of decision cost, which is why the page leads with it.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else math.nan
    return {
        "accuracy": _round(accuracy_score(y, pred)),
        "precision": _round(precision_score(y, pred, zero_division=0)),
        "recall": _round(recall_score(y, pred, zero_division=0)),
        "f1": _round(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": _round(balanced_accuracy_score(y, pred)),
        "specificity": _round(specificity),
        "roc_auc": _round(roc_auc_score(y, p)),
        "pr_auc": _round(_average_precision(y, p)),
        "threshold": _round(threshold),
        "support": {"n": int(y.size), "real": int((y == 0).sum()), "fake": int((y == 1).sum())},
    }


def _average_precision(y, p) -> float:
    from sklearn.metrics import average_precision_score

    return average_precision_score(y, p)


def roc_payload(y_true: Sequence[int], proba: Sequence[float]) -> dict[str, Any]:
    """ROC in scikit-learn's own units, downsampled to a curve a browser can draw.

    ``roc_curve`` on 50k test images returns one point per distinct score — a megabyte of JSON for
    a line nobody can resolve on screen. The stride keeps the endpoints and every threshold
    crossing that matters: AUC is recomputed on the decimated set only for the *displayed* curve,
    while ``metrics.roc_auc`` above always uses the full one.
    """
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(np.asarray(y_true), np.asarray(proba, dtype=np.float64))
    thr = _finite_thresholds(thr)
    fpr, tpr, thr = _decimate(fpr, tpr, thr)
    return {
        "fpr": [round(float(v), 5) for v in fpr],
        "tpr": [round(float(v), 5) for v in tpr],
        "thresholds": [round(float(v), 5) for v in thr],
    }


def _finite_thresholds(thr):
    """Pull the +/-inf sentinels on a curve's threshold axis into [0,1].

    ``roc_curve`` opens its descending threshold list with ``inf`` — the point above every score,
    where nothing is predicted positive and the curve sits at (0,0). That origin is the point the
    chart needs, so the column cannot drop it, and ``Infinity`` is not JSON, so it cannot be
    written either. Scores here are probabilities, so the sentinel moves to just past the extreme
    real threshold inside the interval it came from.
    """
    import numpy as np

    out = np.asarray(thr, dtype=np.float64).copy()
    finite = out[np.isfinite(out)]
    if not finite.size:
        return np.full(out.shape, 0.5)
    out[out == np.inf] = min(float(finite.max()) + 1e-6, 1.0)
    out[out == -np.inf] = max(float(finite.min()) - 1e-6, 0.0)
    out[~np.isfinite(out)] = 0.5
    return np.clip(out, 0.0, 1.0)


def pr_payload(y_true: Sequence[int], proba: Sequence[float]) -> dict[str, Any]:
    """Precision-recall, reported only when the positive class is rare enough to be informative.

    On a 50/50 split PR and ROC say the same thing; on the deployed distribution — where almost
    every upload is a real photograph — PR is the curve that matters and ROC flatters the model.
    The file records which regime the evaluation was run in so the page is not read as a claim
    about the other one.
    """
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(np.asarray(y_true), np.asarray(proba, dtype=np.float64))
    prec, rec, thr = _align_pr(prec, rec, thr)
    prec, rec, thr = _decimate(prec, rec, thr)
    return {
        "precision": [round(float(v), 5) for v in prec],
        "recall": [round(float(v), 5) for v in rec],
        "thresholds": [round(float(v), 5) for v in thr],
    }


def _align_pr(prec, rec, thr):
    """Three columns of equal length and finite thresholds, or nothing that a browser can plot.

    ``precision_recall_curve`` returns precision/recall one element longer than thresholds: the
    extra pair is the sentinel past the largest score, where no sample is predicted positive and
    precision is defined as 1.0 by convention rather than measured. Dropping it loses an endpoint
    no curve needs and makes the columns line up. Depending on the scikit-learn version the
    matching threshold is either absent or ``-inf`` — the latter parses through ``json.dump`` and
    dies in ``JSON.parse``, so it is filtered here instead of relying on the version.
    """
    import numpy as np

    prec = np.asarray(prec, dtype=np.float64)
    rec = np.asarray(rec, dtype=np.float64)
    thr = np.asarray(thr, dtype=np.float64)
    if thr.size != prec.size:
        prec, rec = prec[: thr.size], rec[: thr.size]
    keep = np.isfinite(thr)
    return prec[keep], rec[keep], np.clip(thr[keep], 0.0, 1.0)


def _decimate(*columns, max_points: int = 240):
    import numpy as np

    n = len(columns[0])
    if any(len(c) != n for c in columns):
        raise ValueError(f"decimating columns of unequal length: {[len(c) for c in columns]}")
    if n <= max_points:
        return columns
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return tuple(np.asarray(c)[idx] for c in columns)


def confusion_payload(y_true: Sequence[int], proba: Sequence[float], *, threshold: float,
                      labels: Sequence[str] = ("real", "fake")) -> dict[str, Any]:
    """Counts in the shape ``{labels, matrix}`` the frontend's chart expects.

    ``matrix[i][j]`` is "truth i, predicted j", which is the convention the chart's axis labels
    assume. Getting it backwards is invisible in a symmetric 50/50 case and makes an
    over-flagging model look like an under-flagging one in every other case, so the convention is
    stated here and asserted in the tests.
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    pred = (np.asarray(proba, dtype=np.float64) >= float(threshold)).astype(int)
    m = confusion_matrix(np.asarray(y_true), pred, labels=[0, 1])
    return {"labels": list(labels), "matrix": [[int(v) for v in row] for row in m]}


def ece(y_true: Sequence[int], proba: Sequence[float], *, bins: int = 10) -> float | None:
    """Expected calibration error: the gap between a stated probability and the observed rate.

    Accuracy answers "is the ranking right". ECE answers "does 0.8 mean 80%", which is the only
    question a number printed as a percentage on a forensics page has to answer. A model can be
    near-perfectly ranked and badly calibrated at the same time, and the gauge would still be
    lying.
    """
    import numpy as np

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    if p.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if not count:
            continue
        total += (count / p.size) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def write_evaluation(
    path: str | os.PathLike[str],
    *,
    metrics: dict[str, Any],
    dataset: dict[str, Any],
    model: str,
    confusion: dict[str, Any] | None = None,
    roc: dict[str, Any] | None = None,
    pr: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    robustness: list[dict[str, Any]] | None = None,
    split: dict[str, Any] | None = None,
    tta: dict[str, Any] | None = None,
    trained_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write ``metrics.json`` and return its path. Validates first; refuses to write a bad file.

    The curve files are written beside it with the same payload, because the frontend fetches
    ``/api/metrics`` for the scalars and reads the curves from the same document — keeping them in
    one file is what stops a page from pairing a new AUC with an old curve.
    """
    target = os.fspath(path)
    payload: dict[str, Any] = {
        "available": True,
        "trained_at": trained_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "dataset": dataset,
        "metrics": metrics,
    }
    if split is not None:
        payload["split"] = split
    if confusion is not None:
        payload["confusion_matrix"] = confusion
    if roc is not None:
        payload["roc_curve"] = roc
    if pr is not None:
        payload["pr_curve"] = pr
    if calibration is not None:
        payload["calibration"] = calibration
    if robustness is not None:
        payload["robustness"] = robustness
    if tta is not None:
        payload["tta"] = tta
    if extra:
        payload.update(extra)

    problems = validate(payload)
    if problems:
        raise ValueError("refusing to write an unusable metrics file: " + "; ".join(problems))

    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
    return target


#: The fields ``frontend/src/lib/types.ts::Evaluation`` renders unconditionally. A missing key
#: here is a runtime ``undefined`` on the page, which React shows as an empty cell — the worst
#: kind of defect on a product whose subject is precision.
REQUIRED = ("trained_at", "model", "dataset", "metrics")
REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "threshold")


def validate(payload: dict[str, Any]) -> list[str]:
    """Every way this file can be structurally wrong, checked before it is written."""
    problems: list[str] = []
    for key in REQUIRED:
        if key not in payload:
            problems.append(f"missing top-level key {key!r}")
    metrics = payload.get("metrics") or {}
    for key in REQUIRED_METRICS:
        if key not in metrics:
            problems.append(f"metrics missing {key!r}")
        elif metrics[key] is None and key in ("accuracy", "roc_auc"):
            problems.append(f"metrics.{key} is null — the evaluation produced no value for it")
    dataset = payload.get("dataset") or {}
    for key in ("name", "train", "val", "test"):
        if key not in dataset:
            problems.append(f"dataset missing {key!r}")
    if "note" not in dataset:
        problems.append("dataset.note is missing — every evaluation must say what it measured")
    cm = payload.get("confusion_matrix")
    if cm is not None:
        matrix = cm.get("matrix") or []
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            problems.append("confusion_matrix.matrix must be 2x2")
    for name in ("roc_curve", "pr_curve"):
        curve = payload.get(name)
        if curve is not None:
            lengths = {len(v) for k, v in curve.items() if isinstance(v, list)}
            if len(lengths) > 1:
                problems.append(f"{name} columns differ in length: {sorted(lengths)}")
    try:
        text = json.dumps(payload, allow_nan=False, sort_keys=True, default=str)
    except ValueError as exc:
        problems.append(f"payload is not JSON-serialisable without NaN: {exc}")
    else:
        if "NaN" in text or "Infinity" in text:
            problems.append("payload contains a non-finite number")
    return problems
