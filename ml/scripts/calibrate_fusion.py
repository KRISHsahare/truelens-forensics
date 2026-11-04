"""Fit the two calibrated fusion classifiers and write ``fusion.joblib``.

    # real labelled corpus, into the directory the backend reads
    python ml/scripts/calibrate_fusion.py --corpus D:/datasets/fusion

    # throwaway run that exercises every stage of this script without any downloads
    python ml/scripts/calibrate_fusion.py --generated --corpus ml/.smoke/fusion_corpus \
        --out ml/.smoke/fusion.joblib --weights ml/.smoke/model.pt

The hand-tuned weights it replaces were guesses, and the ``fused * 1.6`` stretch on top of them
was theatre. What is fitted here is two ``HistGradientBoostingClassifier`` heads over the 48
columns of ``app.analysis.features.FEATURES`` — one for P(generated), one for P(manipulated) —
each followed by a sigmoid (Platt) calibrator fitted on rows the base estimator never saw.

**Why the calibrator is fitted separately rather than by cross-validation over all rows.**
``CalibratedClassifierCV(cv="prefit")`` no longer exists in the pinned scikit-learn, and its
``FrozenEstimator`` replacement puts the user in charge of the disjointness: "the user has to take
care manually that data for model fitting and calibration are disjoint". So the split is done here,
by group, and both halves are named in the report. Getting this wrong is not a small accuracy loss
— it produces a probability map fitted to the estimator's own memorisation, and a gauge that then
prints "87%" for a number that has never been tested against a frequency.

**What this script refuses to do.** Write a blob that cannot be scored (a head with no positives on
the calibration half), stay silent about a corpus too small to calibrate, or place a model trained
on synthetic constructions into the ``artifacts/`` directory the backend loads on boot. A fusion
model is multiplied into every verdict the product emits; a confidently wrong one raises apparent
reliability while corrupting the answer, which is the specific failure the whole reliability layer
was built to avoid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import add_paths, artifact  # noqa: E402

add_paths()

# A cp1252 console turns every em dash in this script's own diagnostics into mojibake, and a report
# nobody can read is not a diagnostic.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2

_MIN_ROWS = 24
_MIN_PER_CLASS_IN_CALIB = 2


def _print(msg: str = "") -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ model fit
def fit_head(X_train, y_train, X_calib, y_calib, *, feature_names: list[str],
             seed: int, method: str, cv: int = 3) -> tuple:
    """Base estimator on the training groups, calibrator on the held-out groups.

    Returns ``(calibrated, base, report)``. The base head's own scores are reported alongside the
    calibrated ones because the difference is the whole argument for the calibrator: a boosted
    tree's ``predict_proba`` is a ranking that has never been checked against how often its
    top-``0.9`` rows were actually positive, and on a small corpus that gap is usually the first
    thing that closes when sample count grows.

    Two things the shape of this vector forces:

    * **Constant columns are dropped, not silently binned.** ``HistGradientBoostingClassifier``
      raises inside its own binning pass (``sliding_window_view`` on a single distinct value) when a
      column never varies — which is the normal state of ``c2pa.*`` or ``face.*`` on a small or
      single-modality corpus, and would otherwise kill the run over a column that carries no
      information anyway. ``VarianceThreshold`` in front of the model removes them, and the report
      names them, so "this head never saw credentials" is a printed fact rather than a crash.
      Pipeline rather than a custom estimator on purpose: the blob is unpickled by the API process,
      which has no path to ``ml/``, so every class in it has to be scikit-learn's own.
    * **NaN reaches the trees unimputed during fitting.** ``learned_fuse`` imputes at scoring time,
      but the split the model is *fitted* on is where "this technique did not run" has to become a
      direction in the tree. A model fitted on zeros cannot tell a missing ELA from a clean one.
    """
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline

    from truelens_ml.evalout import ece

    if len(np.unique(y_train)) < 2:
        raise ValueError("training half holds only one class for this head; nothing to rank")
    if len(np.unique(y_calib)) < 2:
        raise ValueError("calibration half holds only one class for this head; a Platt map "
                         "cannot be fitted to a constant — add more groups to the corpus")

    with warnings.catch_warnings():
        # ``nanvar`` complains about all-NaN and zero-degree-of-freedom slices. Both are the
        # expected shape of a small corpus — a technique that never ran contributes a column of
        # NaN, which is exactly the column VarianceThreshold is here to drop — and a wall of
        # RuntimeWarnings buries the two lines this script exists to print.
        warnings.simplefilter("ignore", RuntimeWarning)
        base = Pipeline([
            ("select", VarianceThreshold()),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.07,
                max_iter=150,
                max_depth=3,
                max_leaf_nodes=8,
                min_samples_leaf=4,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            )),
        ]).fit(X_train, y_train)

    # ``ensemble=False``: one Platt map over the whole calibration half, using the frozen base head's
    # own decision values. ``cv`` only has to be small enough that every stratified fold stays
    # two-class, which is why it is a flag rather than a constant — the default of 3 survives the
    # dozen-row corpora a reviewer can build by hand, and 5 does not.
    calib = CalibratedClassifierCV(FrozenEstimator(base), method=method, cv=cv, ensemble=False)
    calib.fit(X_calib, y_calib)

    kept = np.asarray(base.named_steps["select"].get_support(), dtype=bool)
    p_base = np.asarray(base.predict_proba(X_calib)[:, 1], dtype=float)
    p_cal = np.asarray(calib.predict_proba(X_calib)[:, 1], dtype=float)
    raw = np.asarray(base.decision_function(X_calib), dtype=float)
    y = np.asarray(y_calib)
    report = {
        "n_train": int(X_train.shape[0]),
        "n_calib": int(X_calib.shape[0]),
        "positives_train": int(np.sum(y_train)),
        "positives_calib": int(np.sum(y)),
        "auc": round(float(roc_auc_score(y, p_cal)), 4),
        "brier_uncalibrated": round(float(brier_score_loss(y, p_base)), 4),
        "brier_calibrated": round(float(brier_score_loss(y, p_cal)), 4),
        "ece_uncalibrated": round(float(ece(y, p_base)), 4) if ece(y, p_base) is not None else None,
        "ece_calibrated": round(float(ece(y, p_cal)), 4) if ece(y, p_cal) is not None else None,
        "proba_range": [round(float(p_cal.min()), 4), round(float(p_cal.max()), 4)],
        "decision_range": [round(float(raw.min()), 4), round(float(raw.max()), 4)],
        "columns_used": int(kept.sum()),
        "columns_dropped": [feature_names[i] for i in np.flatnonzero(~kept)],
    }
    return calib, base, report


# ---------------------------------------------------------------- guard rails
def write_problems(out_path: str, *, generated: bool, allow_synthetic: bool,
                   rows: int, min_rows: int, populated: dict[str, float],
                   n_calib: int, min_calib: int, labels: tuple[list[int], list[int]],
                   min_per_class: int) -> list[str]:
    """Reasons this run must not write a blob, in the order a reader would want them.

    Returning a list instead of raising is deliberate: a training run that dies on the first
    complaint makes you fix one thing per run, and the second complaint was the one that mattered.
    """
    problems: list[str] = []
    if rows < min_rows:
        problems.append(
            f"{rows} usable rows is below --min-rows={min_rows}; a 48-column boosted model and a "
            "Platt map cannot both be estimated from that. The labelled expert prior is the "
            "honest output at this sample size — pass more images rather than --min-rows 1.")
    if n_calib < min_calib:
        problems.append(
            f"the calibration half has {n_calib} rows, below --min-calib={min_calib}. Every row it "
            "is short is a row the Platt map is fitted on and then not tested against, so the "
            "reported ECE describes the estimator's own training data. Lower --frac-train, or pass "
            "more images.")
    for key, column in (("ai", 0), ("edited", 1)):
        counts = {value: sum(1 for v in labels[column] if v == value) for value in (0, 1)}
        if min(counts.values()) < min_per_class:
            problems.append(
                f"{key} head: only {counts[1]} positive and {counts[0]} negative rows in the "
                f"calibration half, need --min-per-class={min_per_class} of each. A monotone "
                "calibration map fitted to one class is a constant, and the service would print "
                "that constant as a probability.")
    if generated and not allow_synthetic and _is_live_artifact(out_path):
        problems.append(
            f"{out_path} is the file the backend loads on boot, and this corpus was built by "
            "--generated. Synthetic constructions teach the heads what *these constructions* look "
            "like; writing them into artifacts/ would change every verdict in the product. Pass "
            "--out ml/.smoke/fusion.joblib for a contract test, or --allow-synthetic if you truly "
            "mean it.")
    dead = [name for name, share in populated.items() if share < 0.5]
    if dead and _is_live_artifact(out_path):
        problems.append(
            "these techniques measured under half the corpus, so their columns carry no signal for "
            f"the heads: {dead}. Writing this to {out_path} would have the service score real "
            "uploads against a model that never saw faces, credentials or a learned detector. "
            "Fix the corpus or the checkpoint, or point --out somewhere else for a contract test.")
    return problems


def _is_live_artifact(path: str) -> bool:
    return Path(path).name == "fusion.joblib" and Path(path).parent.name == "artifacts"


# ------------------------------------------------------------------------ run
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="", help="directory with authentic/ ai_generated/ edited/ "
                                                 "ai_and_edited/ sub-trees")
    ap.add_argument("--generated", action="store_true",
                    help=f"build a labelled corpus of known constructions into --corpus first "
                         f"(requires the directory to be empty)")
    ap.add_argument("--groups-per-class", type=int, default=6)
    ap.add_argument("--variants-per-group", type=int, default=2)
    ap.add_argument("--image-height", type=int, default=640)
    ap.add_argument("--image-width", type=int, default=800)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--out", default="", help="where to write fusion.joblib "
                                              "(default: the artifacts directory the backend reads)")
    ap.add_argument("--report", default="", help="optional JSON sidecar path for the run summary")
    ap.add_argument("--weights", default="", help="CNN checkpoint to score the corpus with")
    ap.add_argument("--clip-probe", default="", help="CLIP probe to score the corpus with")
    ap.add_argument("--no-cnn", action="store_true",
                    help="leave the cnn columns unmeasured (faster; the heads then cannot learn "
                         "from the learned detector)")
    ap.add_argument("--method", choices=("sigmoid", "isotonic"), default="sigmoid")
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-rows", type=int, default=_MIN_ROWS)
    ap.add_argument("--min-calib", type=int, default=12,
                    help="rows required in the calibration half before anything is fitted")
    ap.add_argument("--calib-cv", type=int, default=3,
                    help="folds the Platt/isotonic map is estimated over")
    ap.add_argument("--min-per-class", type=int, default=_MIN_PER_CLASS_IN_CALIB,
                    help="rows of each label required in the calibration half")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="permit a --generated corpus to be written into artifacts/")
    args = ap.parse_args()

    from truelens_ml import fusionset

    if not args.corpus:
        ap.error("--corpus is required (it is also where --generated writes)")
    if args.generated:
        _print(f"building a generated corpus in {args.corpus}")
        counts = fusionset.build_generated_corpus(
            args.corpus, groups_per_class=args.groups_per_class,
            variants_per_group=args.variants_per_group, size=(args.image_height, args.image_width),
            seed=args.seed)
        _print(f"  wrote {sum(counts.values())} images: " +
               ", ".join(f"{c} {n}" for c, n in counts.items()))

    samples = fusionset.discover(args.corpus,
                                limit_per_class=args.limit_per_class or None)
    _print(f"corpus: {json.dumps(fusionset.tally(samples))}")

    splits = fusionset.split_by_group(samples, frac_train=args.frac_train, seed=args.seed)
    for name, part in splits.items():
        for cls in fusionset.CLASSES:
            n = sum(1 for s in part if s.cls == cls)
            if n == 0:
                _print(f"  note: split {name!r} contains no {cls} images")
        _print(f"  {name}: {len(part)} rows, {len({s.key for s in part})} groups")

    overrides: dict[str, object] = {"cnn_tta": False}
    if args.weights:
        if not Path(args.weights).is_file():
            ap.error(f"--weights {args.weights} does not exist")
        overrides["cnn_weights_path"] = args.weights
    if args.no_cnn:
        # A path that cannot exist: the runtime reports unavailable, the analyser returns
        # available=False, and the cnn columns stay NaN — which is how the heads learn that "no
        # detector" is not "detector said clean".
        overrides.update({"cnn_weights_path": "<disabled>", "clip_probe_enabled": False})
    if args.clip_probe:
        overrides.update({"clip_probe_path": args.clip_probe, "clip_probe_enabled": True})

    def progress(i: int, n: int) -> None:
        if i and i % 10 == 0:
            _print(f"  extracting {i}/{n}")

    _print("running every analyser over the corpus (this is the slow part)")
    started = time.time()
    matrix, names, unreadable = fusionset.extract_features(
        splits["train"] + splits["calib"], config_overrides=overrides, progress=progress)
    _print(f"  {matrix.shape[0]} rows x {matrix.shape[1]} columns in {time.time() - started:.1f}s")
    if unreadable:
        _print(f"  {len(unreadable)} unreadable file(s), first: {unreadable[0]}")

    # Rows must be re-paired with the split after extraction, since a skipped file shifts positions.
    kept = set()
    for part in ("train", "calib"):
        for s in splits[part]:
            kept.add(s.path)
    ordered = [s for s in splits["train"] + splits["calib"] if s.path in kept]
    if len(ordered) != matrix.shape[0]:
        raise SystemExit(f"internal error: {len(ordered)} samples vs {matrix.shape[0]} rows")
    idx = {s.path: i for i, s in enumerate(ordered)}
    tr = np_array([idx[s.path] for s in splits["train"]])
    ca = np_array([idx[s.path] for s in splits["calib"]])

    populated = fusionset.populated_columns(matrix, names)
    _print("  measured share per technique: " +
           ", ".join(f"{k}={v:.2f}" for k, v in list(populated.items())[:10]))
    quiet = [k for k, v in populated.items() if v < 0.5]
    if quiet:
        _print(f"  unmeasured on this corpus: {quiet} — the heads cannot learn from them")

    out_path = args.out or artifact("fusion.joblib")
    train_labels = fusionset.labels(splits["train"])
    calib_labels = fusionset.labels(splits["calib"])

    # Guard before fitting, not after. A corpus that is too small does not have to reach
    # scikit-learn to find out: the default cross-validation inside the calibrator raises
    # "n_splits=5 greater than the number of samples", a traceback that names neither the split nor
    # the flag to fix, and buries the answer the operator actually needs.
    problems = write_problems(
        out_path, generated=args.generated, allow_synthetic=args.allow_synthetic,
        rows=int(matrix.shape[0]), min_rows=args.min_rows, populated=populated,
        n_calib=int(len(ca)), min_calib=args.min_calib,
        labels=calib_labels, min_per_class=args.min_per_class)
    if problems:
        _print("\nrefusing to write:")
        for p in problems:
            _print(f"  - {p}")
        return 2

    heads: dict[str, object] = {}
    head_reports: dict[str, dict] = {}
    warnings_seen: list[str] = []
    for key, column in (("ai", 0), ("edited", 1)):
        y_train = np_array(train_labels[column])
        y_calib = np_array(calib_labels[column])
        calib, _base, report = fit_head(matrix[tr], y_train, matrix[ca], y_calib,
                                        feature_names=list(names), seed=args.seed,
                                        method=args.method, cv=args.calib_cv)
        heads[key] = calib
        head_reports[key] = report
        _print(f"\n{key}: auc={report['auc']} "
               f"brier {report['brier_uncalibrated']} -> {report['brier_calibrated']} "
               f"ece {report['ece_uncalibrated']} -> {report['ece_calibrated']} "
               f"(train {report['n_train']}, calib {report['n_calib']}, "
               f"{report['columns_used']}/{len(names)} columns)")
        if report["columns_dropped"]:
            _print(f"  ignored (constant on this corpus): {report['columns_dropped'][:8]}"
                   + (" ..." if len(report["columns_dropped"]) > 8 else ""))
        if (report["ece_calibrated"] is not None and report["ece_uncalibrated"] is not None
                and report["ece_calibrated"] > report["ece_uncalibrated"]):
            note = (f"{key}: the calibration map made expected calibration error worse "
                    f"({report['ece_uncalibrated']} -> {report['ece_calibrated']}). On a base head "
                    "that is already near-perfect the map's only work is to pull probabilities "
                    "back toward the corpus prior, which is what a dozen rows can support and "
                    "nothing more. The calibrated head is still the one that ships — an "
                    "uncalibrated boosted-tree score is a ranking, not a percentage — but this "
                    "corpus is too small to read the number as evidence about the calibrator.")
            _print(f"  WARNING {note}")
            warnings_seen.append(note)

    from app.analysis.features import FEATURES

    if list(names) != list(FEATURES):
        raise SystemExit("feature columns drifted from app.analysis.features.FEATURES; refusing "
                         "to write a blob the service would reject anyway")

    summary = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": os.path.abspath(args.corpus),
        "generated_corpus": bool(args.generated),
        "feature_names": list(FEATURES),
        "n_features": len(FEATURES),
        "method": args.method,
        "seed": args.seed,
        "split": {k: len(v) for k, v in splits.items()},
        "groups": {k: len({s.key for s in v}) for k, v in splits.items()},
        "tally": fusionset.tally(samples),
        "unreadable": len(unreadable),
        "measured_share": populated,
        "heads": head_reports,
        "warnings": warnings_seen,
        "weights": args.weights or None,
        "clip_probe": args.clip_probe or None,
        "note": ("The heads score the deployed feature vector. Their probabilities are fitted on "
                 "the calibration half of this corpus and their meaning is bounded by it: see the "
                 "dataset note for what those images are."),
    }
    blob = {
        "ai": heads["ai"],
        "edited": heads["edited"],
        "feature_names": list(FEATURES),
        "calibrated": True,
        "meta": summary,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(blob, out_path)
    _print(f"\nwrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")

    # The service's own loader is the acceptance test, so a blob this script is happy with and the
    # API silently declines to load is not possible.
    from app.analysis.fusion import _Learned  # noqa: SLF001 - load() is the contract

    _Learned._cache.pop(out_path, None)
    if _Learned.load(out_path) is None:
        _print("  FAIL _Learned.load() rejected the artifact it was just given")
        return 1
    _print("  _Learned.load() accepts it; the learned fusion path is live for this file")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        _print(f"  wrote {args.report}")
    return 0


def np_array(values):
    import numpy as np

    return np.asarray(values, dtype=np.int64)


if __name__ == "__main__":
    raise SystemExit(main())
