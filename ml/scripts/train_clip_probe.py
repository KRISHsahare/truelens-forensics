"""Train the optional UnivFD-style CLIP linear probe and write ``clip_probe.joblib``.

    python ml/scripts/train_clip_probe.py --corpus data/corpus --out artifacts/clip_probe.joblib
    python ml/scripts/train_clip_probe.py --self-test --out ml/.smoke/clip_probe.joblib

**Why a linear probe at all.** An EfficientNet fine-tuned end-to-end on one generator family
learns that family's fingerprint — the specific upsample, the specific GAN checkerboard — and its
headline accuracy hides how badly it transfers to the next architecture. A single linear layer on
*frozen* CLIP features has almost no capacity, so it cannot memorise a fingerprint; the best it
can do is find the broad "synthetic versus photographic" direction in a space that was never built
for forgery detection. That is why the backend reports the two scores side by side and surfaces
their *agreement*: concurrence between detectors with deliberately different inductive biases is
worth more than either number, and disagreement is an out-of-distribution signal in itself.

**Why it is optional and stays optional.** ``open-clip-torch`` is in the ``[clip]`` extra and the
ViT-L/14 weights are ~900 MB on first use, so nothing here is required for the product to run. The
API's ``ClipProbe`` degrades to ``available = False`` and the CNN technique records ``cnn.clip_score``
as NaN — never as 0.0 — which is the same "silence is not evidence of authenticity" rule the rest of
the system is built on.

**The artifact shape is fixed by the consumer.** ``app/ml/clip_probe.py`` does
``joblib.load(path)`` and then ``head.predict_proba(feat[None, :])[0, 1]``, so the file must contain
the *estimator itself* — not a dict with metadata. The run summary therefore goes to a sidecar JSON
beside it, and the estimator is saved with the plain scikit-learn classes the API process can
unpickle without a path to ``ml/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import add_paths, artifact  # noqa: E402

add_paths()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2

#: ViT-L/14's embedding width. The probe is fitted against whatever the encoder emits and this is
#: only the expected value, used to reject a cache built by a different architecture.
FEATURE_DIM = 768


def _print(msg: str = "") -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------- encoding
def encode(samples, *, batch: int = 16) -> tuple:
    """Frozen-CLIP embeddings for ``samples``: ``(matrix, paths)``.

    One encoder instance for the whole corpus — the same mistake the prototype made with
    ``torch.load`` per request is making it per batch here.

    The features are L2-normalised exactly as ``ClipProbe.features`` normalises them at serving
    time. A probe fitted on un-normalised embeddings and then applied to normalised ones is a model
    fitted on a different input distribution than it scores, and the difference is invisible in the
    file name.
    """
    import numpy as np
    import open_clip
    import torch
    from PIL import Image

    from app.ml.clip_probe import ARCH, PRETRAINED

    _print(f"loading {ARCH}/{PRETRAINED} (one time, ~900 MB on first use)")
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    model.eval()

    chunks: list[np.ndarray] = []
    kept: list[str] = []
    for start in range(0, len(samples), batch):
        batch_samples = samples[start:start + batch]
        tensors, paths = [], []
        for s in batch_samples:
            try:
                pil = Image.open(s.path).convert("RGB")
            except Exception as exc:  # noqa: BLE001 - one unreadable file must not cost the run
                _print(f"  skip {s.path}: {type(exc).__name__}")
                continue
            tensors.append(preprocess(pil))
            paths.append(s.path)
        if not tensors:
            continue
        with torch.inference_mode():
            f = model.encode_image(torch.stack(tensors))
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        chunks.append(f.float().cpu().numpy())
        kept.extend(paths)
        if (start // batch) % 20 == 0:
            _print(f"  encoded {min(start + batch, len(samples))}/{len(samples)}")
    if not chunks:
        raise SystemExit("no sample could be encoded; check the corpus path and the CLIP install")
    return np.vstack(chunks), kept


def encode_all(samples, *, batch: int, cache: str) -> dict[str, Any]:
    """``path -> feature row`` for every sample, from cache when it already covers them.

    Keyed by path rather than by position, because the cache outlives the split it was built for: a
    positional array is only valid for the exact corpus order of the run that wrote it, and reusing
    one after adding images would silently pair the wrong embedding with the wrong label.

    The cache exists because encoding 50k images on CPU takes hours while the probe itself is a
    two-second fit you will want to re-run at several values of C.
    """
    import numpy as np

    from app.ml.clip_probe import ARCH, PRETRAINED

    cache_path = Path(cache) if cache else None
    table: dict[str, Any] = {}
    if cache_path and cache_path.is_file():
        with np.load(cache_path, allow_pickle=True) as z:
            if str(z["arch"]) != ARCH or str(z["pretrained"]) != PRETRAINED:
                raise SystemExit(
                    f"cache {cache_path} was built by {str(z['arch'])}/{str(z['pretrained'])}, "
                    f"not {ARCH}/{PRETRAINED}. A probe fitted to another encoder's space would "
                    "still score — it just would not mean anything. Delete the cache or pass a "
                    "new path.")
            feats = z["feats"]
            table = {str(p): feats[i] for i, p in enumerate(z["paths"])}
        missing = [s.path for s in samples if s.path not in table]
        _print(f"cache covers {len(samples) - len(missing)} of {len(samples)} images")
    else:
        missing = [s for s in samples if s.path not in table]

    if missing:
        X, paths = encode(missing, batch=batch)
        table.update({p: X[i] for i, p in enumerate(paths)})
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ordered = [s.path for s in samples if s.path in table]
            np.savez_compressed(
                cache_path,
                feats=np.array([table[p] for p in ordered]),
                paths=np.array(ordered, dtype=object),
                arch=ARCH, pretrained=PRETRAINED, allow_pickle=True)
            _print(f"  wrote {cache_path}")
    absent = [s.path for s in samples if s.path not in table]
    if absent:
        _print(f"  {len(absent)} image(s) never encoded, first: {absent[0]}")
    return table


def assemble(samples, table: dict[str, Any]) -> tuple:
    """The rows, labels and families for one split, in that split's own order."""
    import numpy as np

    keep = [s for s in samples if s.path in table]
    X = np.array([table[s.path] for s in keep], dtype=np.float64)
    y = np.asarray([int(s.label) for s in keep], dtype=np.int64)
    return X, y, [s.source for s in keep]


# ------------------------------------------------------------------------ fit
def fit_probe(X_train, y_train, X_val, y_val, *, seed: int) -> tuple:
    """Logistic regression on frozen features, C chosen on the validation split.

    ``class_weight='balanced'`` because a corpus pulled from Real-and-Fake-Faces plus a diffusion
    set is never 50/50, and an unweighted probe that predicts the majority class for every input
    still reports a respectable accuracy while being useless.

    ``lbfgs`` on 768 columns and thousands of rows converges quickly; ``max_iter=3000`` because a
    regularised linear probe near the separable limit is exactly the case where the default 100
    iterations stops early and reports a worse score than the model is capable of.

    Returns ``(best_model, grid)`` — the whole grid is reported, not only the winner, so a reader
    can see whether the val AUC was flat (the probe is at its capacity limit, and a bigger corpus
    will not help) or peaked (the capacity was the binding constraint).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    grid = []
    best = None
    for c in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                 max_iter=3000, random_state=seed)
        try:
            clf.fit(X_train, y_train)
        except Exception as exc:  # noqa: BLE001 - a non-converging C is a row in the table, not a crash
            grid.append({"C": c, "val_auc": None, "note": f"{type(exc).__name__}: {str(exc)[:80]}"})
            continue
        auc = float(roc_auc_score(y_val, clf.decision_function(X_val)))
        grid.append({"C": c, "val_auc": round(auc, 4)})
        if best is None or auc > best[2]:
            best = (clf, c, auc)
    if best is None:
        raise SystemExit("no C converged; the features are not finite or the split is empty")
    return best[0], best[1], grid


def evaluate_head(head, X, y, *, threshold: float) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    p = np.asarray(head.predict_proba(X)[:, 1], dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "positives": int(np.sum(y)),
    }


def per_source(head, X, y, sources) -> list[dict]:
    """AUC per generator family — the number UnivFD exists to report.

    Pooled accuracy across families is the figure that hides the whole problem: a probe that is
    0.99 on the family it trained near and 0.52 on an unseen one still prints 0.85 overall. Each
    family is scored on its own rows, and a family with only one label is reported as unmeasurable
    rather than as a perfect or a zero score.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    out = []
    for source in sorted(set(sources)):
        mask = np.asarray([s == source for s in sources], dtype=bool)
        yy = np.asarray(y)[mask]
        if yy.size < 4 or len(np.unique(yy)) < 2:
            out.append({"source": source, "n": int(mask.sum()), "roc_auc": None,
                        "note": "one class only, or too few rows to rank anything"})
            continue
        p = np.asarray(head.predict_proba(X[mask])[:, 1], dtype=float)
        out.append({"source": source, "n": int(mask.sum()),
                    "roc_auc": round(float(roc_auc_score(yy, p)), 4)})
    return out


# ------------------------------------------------------------------ self-test
def synthetic_features(n: int, dim: int, seed: int) -> tuple:
    """Separable fake embeddings so the fit/report/save path is testable without open_clip.

    Only the *pipeline* is being tested here — that a head gets fitted, that the C grid is scored on
    the validation split, that the sidecar is written, that ``joblib.load`` gives back an object with
    ``predict_proba`` over ``(1, dim)``. The discriminability belongs to CLIP and to a real corpus,
    which is why the report a self-test writes says so in its own text rather than letting a reader
    infer it from a plausible AUC.

    The signal-to-noise ratio is chosen so a linear probe *can* separate the two classes (about 0.96
    Bayes AUC at this scale). An isotropic draw with a small shift is not separable in 768 dims, and
    a self-test that returns 0.42 verifies only that nothing crashed — not that fitting, model
    selection and evaluation pass information through in the right direction.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    X = rng.normal(scale=0.4, size=(n, dim)) + np.outer(y, direction)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="", help="real/ and fake/ tree, as in ml/README.md")
    ap.add_argument("--out", default="", help="where to write clip_probe.joblib "
                                             "(default: the artifacts directory the backend reads)")
    ap.add_argument("--features-cache", default="", help="npz of a previous encode; written if absent")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fractions", default="0.7,0.15,0.15")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="fit and save on synthetic embeddings; needs no CLIP install and no data")
    ap.add_argument("--report", default="", help="sidecar JSON path (default: beside --out)")
    args = ap.parse_args()

    import joblib
    import numpy as np

    from truelens_ml.evalout import ece

    if args.self_test:
        _print("self-test: synthetic embeddings, no open_clip, no images")
        X, y = synthetic_features(600, FEATURE_DIM, args.seed)
        cut_train, cut_val = int(0.6 * len(X)), int(0.8 * len(X))
        X_train, y_train = X[:cut_train], y[:cut_train]
        X_val, y_val = X[cut_train:cut_val], y[cut_train:cut_val]
        X_test, y_test = X[cut_val:], y[cut_val:]
        test_sources = ["synthetic"] * len(X_test)
        generated = True
    else:
        if not args.corpus:
            ap.error("--corpus is required unless --self-test")
        generated = False
        try:
            import open_clip  # noqa: F401
        except ModuleNotFoundError as exc:
            _print(f"open_clip is not installed ({exc}).")
            _print("  pip install -e backend/.[clip]   # pulls open-clip-torch and the ViT-L/14 weights")
            _print("  or run --self-test to exercise this script's fit/report/save path without it.")
            return 3

        from truelens_ml.corpus import discover, split_samples

        fractions = tuple(float(f) for f in args.fractions.split(","))
        samples = discover(args.corpus, limit=args.limit or None)
        splits = split_samples(samples, fractions=fractions, seed=args.seed)
        _print(f"corpus: train {len(splits['train'])}, val {len(splits['val'])}, "
               f"test {len(splits['test'])}")
        table = encode_all(splits["train"] + splits["val"] + splits["test"],
                           batch=args.batch, cache=args.features_cache)
        X_train, y_train, _ = assemble(splits["train"], table)
        X_val, y_val, _ = assemble(splits["val"], table)
        X_test, y_test, test_sources = assemble(splits["test"], table)
        if not len(X_test):
            raise SystemExit("the test split encoded to zero rows; nothing to measure")

    for name, part in (("train", (X_train, y_train)), ("val", (X_val, y_val)),
                       ("test", (X_test, y_test))):
        if len(np.unique(part[1])) < 2:
            raise SystemExit(f"the {name} split has one class only; the probe cannot be fitted or "
                             "measured on it — check --fractions and the corpus size")

    started = time.time()
    head, best_c, grid = fit_probe(X_train, y_train, X_val, y_val, seed=args.seed)
    _print(f"fitted C={best_c} in {time.time() - started:.1f}s; "
           + " ".join(f"C{g['C']:g}:{g['val_auc']}" for g in grid))

    p_test = np.asarray(head.predict_proba(X_test)[:, 1], dtype=float)
    report = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "encoder": ("SYNTHETIC embeddings - this artifact is a pipeline test, not a detector"
                    if generated else
                    "open_clip ViT-L/14 (openai), frozen, L2-normalised embeddings"),
        "feature_dim": int(X_train.shape[1]),
        "expected_dim": FEATURE_DIM,
        "C": best_c,
        "grid": grid,
        "corpus": os.path.abspath(args.corpus) if args.corpus else None,
        "generated_features": generated,
        "train": {"n": int(X_train.shape[0]), "positives": int(np.sum(y_train))},
        "val": evaluate_head(head, X_val, y_val, threshold=0.5),
        "test": evaluate_head(head, X_test, y_test, threshold=0.5),
        "test_ece": round(float(ece(y_test, p_test)), 4) if ece(y_test, p_test) is not None else None,
        "per_source_test": per_source(head, X_test, y_test, test_sources),
    }
    _print(f"test: {json.dumps(report['test'])}  val: {json.dumps(report['val'])}")
    _print(f"  ECE at test: {report['test_ece']} (probability at threshold 0.5, balanced weights)")
    for row in report["per_source_test"]:
        _print(f"  {row['source']:24s} n={row['n']:6d} auc={row['roc_auc']}")
    if report["val"]["roc_auc"] < report["test"]["roc_auc"] - 0.05:
        _print("  note: test beats val by more than 0.05 AUC, so the two splits are not comparable "
               "in difficulty; treat the val number as the conservative one")
    if report["feature_dim"] != FEATURE_DIM and not generated:
        _print(f"  note: the encoder emitted {report['feature_dim']} dims, not the expected "
               f"{FEATURE_DIM} — check the architecture in app/ml/clip_probe.py matches this probe")

    out_path = args.out or artifact("clip_probe.joblib")
    if generated and _is_live_artifact(out_path):
        _print(f"refusing to write a self-test artifact to {out_path}; pass --out somewhere else")
        return 2
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, out_path)
    _print(f"wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")

    report_path = args.report or str(Path(out_path).with_name("clip_probe_report.json"))
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print(f"wrote {report_path}")

    # What the service will actually do with the file: unpickle it, then ask it for a probability
    # over one embedding. Anything that survives that is a usable artifact.
    reloaded = joblib.load(out_path)
    p = float(reloaded.predict_proba(X_test[:1])[0, 1])
    if not 0.0 <= p <= 1.0:
        _print(f"  FAIL reloaded head returned {p}, outside [0,1]")
        return 1
    _print(f"  reloaded head scores a (1,{X_test.shape[1]}) row -> p={p:.4f}, "
           "which is the contract app/ml/clip_probe.py relies on")
    return 0


def _is_live_artifact(path: str) -> bool:
    return Path(path).name == "clip_probe.joblib" and Path(path).parent.name == "artifacts"


if __name__ == "__main__":
    raise SystemExit(main())
