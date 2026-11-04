"""Prove that a produced artifact satisfies the contract the running service expects.

    python ml/scripts/check_artifacts.py --weights ml/.smoke/model.pt
    python ml/scripts/check_artifacts.py --metrics artifacts/metrics.json --fusion artifacts/fusion.joblib
    python ml/scripts/check_artifacts.py --clip-probe artifacts/clip_probe.joblib

Every check here is a failure mode that a successful training run can still hide:

* A checkpoint whose head has two logits loads into nothing and the API starts with the CNN
  reported unavailable — the same symptom as no file at all, so it reads as "forgot to copy the
  weights" for a day.
* A checkpoint that loads but whose in-place activations were never disabled raises only when
  Grad-CAM runs, which is the first time a reviewer asks "why did the model say that".
* A metrics file with a ``NaN`` in it parses in Python and throws in ``JSON.parse``, so the
  Methodology page shows its "nothing measured" panel while the file sits there looking complete.
* A fusion blob trained against an older column list scores real images with the wrong columns
  and reports a confident number. ``fusion._Learned.load`` refuses it — this script says so
  *before* a deploy rather than in the log afterwards.
* A CLIP probe saved as a dict, or fitted on another encoder's width, fails the same
  ``ClipProbe._load`` that catches every exception and reports "unavailable" — a silent no-op
  rather than an error, on the one path where the product's second opinion disappears.

The checks import the runtime, not a copy of its expectations. That is the point: a verifier that
re-implements what it verifies drifts exactly like the duplicated definition it was meant to
catch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import add_paths  # noqa: E402

add_paths()

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2, same as the backend's own scripts


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.passed: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        (self.passed if ok else self.problems).append(label + (f": {detail}" if detail and not ok else ""))
        return ok

    def require(self, ok: bool, label: str, detail: str = "") -> None:
        self.check(ok, label, detail)


def check_weights(path: str, report: Report) -> None:
    import numpy as np

    from app.ml.cnn_runtime import CnnRuntime, reset_runtime

    report.check(Path(path).is_file(), f"checkpoint exists at {path}", "file not found")
    if not Path(path).is_file():
        return

    reset_runtime()
    runtime = CnnRuntime(path)
    loaded = runtime.load()
    report.require(loaded, "CnnRuntime loads the checkpoint", str(runtime.load_error))
    if not loaded:
        return
    report.require(runtime.available, "runtime reports available")

    head = runtime._model.classifier[1]  # noqa: SLF001 - this is the contract being checked
    report.require(int(getattr(head, "out_features", 0)) == 1,
                   "head is a single logit", f"got {getattr(head, 'out_features', 0)} outputs")

    rng = np.random.default_rng(0)
    arr = rng.random((1, 3, runtime.input_size, runtime.input_size), dtype=np.float32)
    score = runtime.score(arr, tta=True)
    report.require(score is not None and 0.0 <= score <= 1.0, "score() returns P(fake) in [0,1]",
                   str(score))

    cam = runtime.gradcam(arr)
    report.require(cam is not None and cam.ndim == 2 and cam.shape == (7, 7),
                   "Grad-CAM produces a 7x7 map", str(None if cam is None else cam.shape))
    if cam is not None:
        report.require(bool(np.isfinite(cam).all()), "Grad-CAM map is finite")
    # Grad-CAM running at all is the proof that the score path's inference_mode did not poison the
    # graph and that the in-place SiLU modules were rewritten: both failures look like a healthy
    # checkpoint right up until the backward pass.
    print(f"  score={score!r} gradcam_ms={runtime.last_gradcam_ms:.1f}")


def check_metrics(path: str, report: Report) -> None:
    from truelens_ml.evalout import validate

    raw = Path(path).read_text(encoding="utf-8")
    report.require("NaN" not in raw and "Infinity" not in raw,
                   "metrics.json contains no NaN/Infinity token")
    payload = json.loads(raw)
    problems = validate(payload)
    report.require(not problems, "metrics.json passes the writer's own validation",
                   "; ".join(problems))
    report.require(bool(payload.get("dataset", {}).get("note")),
                   "metrics.json says what its numbers were measured on")


def check_fusion(path: str, report: Report) -> None:
    import joblib
    import numpy as np

    from app.analysis.features import FEATURES, vectorize
    from app.analysis.fusion import _Learned  # noqa: SLF001 - load() is the contract

    blob = joblib.load(path)
    report.require(isinstance(blob, dict) and {"ai", "edited", "feature_names"} <= set(blob),
                   "fusion blob has ai/edited/feature_names", str(sorted(blob)) if isinstance(blob, dict) else type(blob).__name__)
    names = list(blob.get("feature_names") or [])
    report.require(names == list(FEATURES),
                   "fusion feature_names match app.analysis.features.FEATURES column-for-column",
                   f"{len(names)} stored vs {len(FEATURES)} current")
    if names != list(FEATURES):
        return
    report.require(_Learned.load(path) is not None, "fusion._Learned accepts the artifact")

    row = vectorize({c: (np.nan if i % 4 == 0 else 0.3) for i, c in enumerate(FEATURES)},
                    impute=True)
    for key in ("ai", "edited"):
        model = blob[key]
        proba = model.predict_proba(row.reshape(1, -1))[0]
        report.require(len(proba) == 2 and abs(float(proba.sum()) - 1.0) < 1e-6,
                       f"{key} returns a two-class probability", str(proba))
    report.require(bool(blob.get("calibrated")), "fusion blob declares itself calibrated",
                   "uncalibrated scores must not be printed as percentages")


def check_clip_probe(path: str, report: Report) -> None:
    """The head against the only thing ``ClipProbe`` ever asks of it.

    ``app/ml/clip_probe.py`` does ``joblib.load(path)`` and then
    ``head.predict_proba(feat[None, :])[0, 1]`` on an L2-normalised ViT-L/14 embedding. It catches
    every exception and degrades to "unavailable", which is the right behaviour in production and
    the wrong thing to discover from a log line a week after a training run "succeeded" — a saved
    dict, a Ridge without ``predict_proba``, or a head fitted on 512-dim features all fail that way.
    """
    import joblib
    import numpy as np

    from app.ml.clip_probe import ARCH, INPUT_SIZE

    report.require(Path(path).is_file(), f"clip probe exists at {path}", "file not found")
    if not Path(path).is_file():
        return
    head = joblib.load(path)
    report.require(isinstance(head, dict) is False and hasattr(head, "predict_proba"),
                   "probe file holds an estimator with predict_proba",
                   "it must be the estimator itself; ClipProbe passes an embedding straight to it")
    if not hasattr(head, "predict_proba"):
        return

    dim = int(getattr(head, "n_features_in_", 0))
    report.require(dim > 0, "probe declares the feature width it was fitted on",
                   f"no n_features_in_ on {type(head).__name__}")
    report.require(dim == 768, f"probe expects the {ARCH} embedding width",
                   f"got {dim} columns; a ViT-L/14 encoder emits 768")
    if dim == 0:
        return

    row = np.zeros((1, dim), dtype=np.float64)
    row[0, 0] = 1.0 / np.sqrt(2.0)
    row[0, 1] = 1.0 / np.sqrt(2.0)  # an L2-normalised embedding, as features() returns one
    proba = np.asarray(head.predict_proba(row)[0], dtype=float)
    report.require(len(proba) == 2 and abs(float(proba.sum()) - 1.0) < 1e-6,
                   "probe returns a two-class probability for a normalised embedding", str(proba))
    report.require(0.0 <= float(proba[1]) <= 1.0, "probe's P(fake) is in [0,1]", str(proba[1]))
    print(f"  dim={dim} p_fake(normalised unit row)={float(proba[1]):.4f} "
          f"input_size={INPUT_SIZE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--fusion", default="")
    ap.add_argument("--clip-probe", default="")
    args = ap.parse_args()

    report = Report()
    if args.weights:
        print(f"weights: {args.weights}")
        check_weights(args.weights, report)
    if args.metrics:
        print(f"metrics: {args.metrics}")
        check_metrics(args.metrics, report)
    if args.fusion:
        print(f"fusion: {args.fusion}")
        check_fusion(args.fusion, report)
    if args.clip_probe:
        print(f"clip probe: {args.clip_probe}")
        check_clip_probe(args.clip_probe, report)
    if not (args.weights or args.metrics or args.fusion or args.clip_probe):
        ap.error("nothing to check: pass --weights, --metrics, --fusion and/or --clip-probe")

    for line in report.passed:
        print(f"  ok   {line}")
    for line in report.problems:
        print(f"  FAIL {line}")
    print(f"\n{len(report.passed)} passed, {len(report.problems)} failed")
    return 1 if report.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
