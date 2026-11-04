"""Score the trained detector on its held-out split and write the Methodology page's numbers.

    python ml/scripts/evaluate_cnn.py --corpus datasets --weights artifacts/model.pt \\
        --out-dir artifacts
    python ml/scripts/evaluate_cnn.py --corpus ml/.smoke/corpus --weights ml/.smoke/model.pt \\
        --out-dir ml/.smoke

Three files, all from one pass over the test split: ``metrics.json`` (everything the page renders
as a figure), ``roc.json`` and ``confusion_matrix.json`` (the two charts, kept separate because
they are the bulk of the payload and the page can show the scalars without them).

**Inference goes through the runtime, not a copy of it.** Preprocessing is
``CnnRuntime.encode_for_resize`` — letterbox, ImageNet normalisation, ``INTER_AREA`` — and the
score is ``CnnRuntime.score(..., tta=...)``. An evaluation that re-implements resizing "the same
way" measures a different pipeline than the one that ships, and the difference shows up as a
gap between the page and the product that nobody can explain.

**The threshold is chosen on validation and applied to test.** A number picked on the same split
it is reported from is not a held-out number; it is the best case of a search. ``--threshold``
overrides it for a reader who wants a specific operating point.

**Per-generator breakdown is the headline, not the appendix.** Pooled accuracy across three
generator families hides that a detector can be 0.94 on the family it met and 0.51 on the next
one. The families come from the corpus directory names, which is why the layout in
``truelens_ml.corpus`` insists on them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import add_paths  # noqa: E402

add_paths()

import app.envbootstrap  # noqa: F401,E402

from truelens_ml.corpus import discover, split_samples, tally  # noqa: E402
from truelens_ml.evalout import (  # noqa: E402
    confusion_payload,
    ece,
    metrics_payload,
    pr_payload,
    roc_payload,
    write_evaluation,
)
from truelens_ml.model import ARCHITECTURE, load_checkpoint  # noqa: E402
from truelens_ml.synthetic import PROVENANCE  # noqa: E402

#: The degradations a deployed image has usually already been through, and the reason each one is
#: measured: a detector whose accuracy collapses at WhatsApp compression quality is a detector
#: that works on a reviewer's laptop and nowhere else.
DEGRADATIONS = (
    ("jpeg_q70", ("jpeg", 70)),
    ("jpeg_q50", ("jpeg", 50)),
    ("blur_3px", ("blur", 3)),
    ("downscale_512", ("resize", 512)),
    ("downscale_256", ("resize", 256)),
)


def _degrade(rgb, spec):
    import cv2

    kind, arg = spec
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), arg])
        if not ok:
            return rgb
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "blur":
        return cv2.GaussianBlur(rgb, (arg | 1, arg | 1), 0)
    if kind == "resize":
        h, w = rgb.shape[:2]
        scale = arg / max(h, w)
        if scale >= 1.0:
            return rgb
        small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    raise ValueError(f"unknown degradation {kind}")


def score_split(runtime, samples, *, tta: bool) -> list[float]:
    """P(fake) for each sample, through the deployed preprocessing path."""
    import cv2
    import numpy as np

    probs: list[float] = []
    for sample in samples:
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(sample.path, dtype=np.uint8),
                                         cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rgb is None or rgb.ndim != 3:
            probs.append(float("nan"))
            continue
        value = runtime.score(runtime.encode_for_resize(rgb), tta=tta)
        probs.append(float("nan") if value is None else value)
    return probs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--fractions", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", default="auto", help="auto (val-F1) or a float in (0,1)")
    ap.add_argument("--tta", action="store_true", help="flip augmentation at inference")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--generated", action="store_true",
                    help="label the dataset as the synthetic pipeline-check corpus")
    args = ap.parse_args()

    import numpy as np

    from app.ml.cnn_runtime import CnnRuntime, reset_runtime

    reset_runtime()
    runtime = CnnRuntime(args.weights)
    if not runtime.load():
        print(f"refusing to evaluate: the runtime cannot load {args.weights} "
              f"({runtime.load_error})")
        return 1

    _, meta = load_checkpoint(args.weights)
    fractions = [float(f) for f in args.fractions.split(",")]
    samples = discover(args.corpus, limit=args.limit)
    splits = split_samples(samples, fractions=fractions, seed=args.seed)

    t0 = time.time()
    val_probs = score_split(runtime, splits["val"], tta=args.tta)
    test_probs = score_split(runtime, splits["test"], tta=args.tta)
    print(f"scored {len(splits['val'])} val + {len(splits['test'])} test in "
          f"{time.time() - t0:.1f}s (tta={args.tta})")

    val_y = [s.label for s in splits["val"]]
    test_y = [s.label for s in splits["test"]]
    keep = [(y, p) for y, p in zip(test_y, test_probs, strict=True) if np.isfinite(p)]
    test_y = [y for y, _ in keep]
    test_probs = [p for _, p in keep]
    if not test_probs:
        print("every test image failed to score; nothing to report.")
        return 1

    threshold = _resolve_threshold(args.threshold, val_y, val_probs)
    print(f"operating point: {threshold:.4f}")

    metrics = metrics_payload(test_y, test_probs, threshold=threshold)
    print(f"test  acc={metrics['accuracy']}  auc={metrics['roc_auc']}  "
          f"f1={metrics['f1']}  n={len(test_probs)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = write_evaluation(
        out_dir / "metrics.json",
        metrics=metrics,
        dataset=_dataset(args, splits, meta),
        model=_model_id(meta, args.tta),
        confusion=confusion_payload(test_y, test_probs, threshold=threshold),
        roc=roc_payload(test_y, test_probs),
        pr=pr_payload(test_y, test_probs),
        calibration=_calibration(val_y, val_probs, test_y, test_probs),
        robustness=_robustness(runtime, splits["test"], threshold=threshold, tta=args.tta),
        split={"method": "group-atomic, class-stratified, sha1-keyed", "seed": args.seed,
               "fractions": fractions,
               "sizes": {k: tally(v)["images"] for k, v in splits.items()}},
        tta={"flip_augmentation": bool(args.tta),
             "note": ("Reported scores average the image and its horizontal mirror, which is the "
                      "same path the API uses when CNN_TTA is on." if args.tta else
                      "Single forward pass, no test-time augmentation.")},
        trained_at=meta.get("trained_at"),
        extra={"checkpoint": {"architecture": meta.get("architecture", ARCHITECTURE),
                              "epoch": meta.get("epoch"),
                              "val_auc_at_train_time": meta.get("val_auc"),
                              "init": meta.get("init"),
                              "img_size": meta.get("img_size", runtime.input_size)}},
    )
    (out_dir / "roc.json").write_text(json.dumps(roc_payload(test_y, test_probs), indent=2),
                                      encoding="utf-8")
    (out_dir / "confusion_matrix.json").write_text(
        json.dumps(confusion_payload(test_y, test_probs, threshold=threshold), indent=2),
        encoding="utf-8")
    print(f"wrote {metrics_path}\nwrote {out_dir / 'roc.json'}\n"
          f"wrote {out_dir / 'confusion_matrix.json'}")
    return 0


def _resolve_threshold(raw: str, val_y, val_probs) -> float:
    if raw != "auto":
        return float(raw)
    from train_cnn import choose_threshold

    paired = [(y, p) for y, p in zip(val_y, val_probs, strict=True) if np_is_finite(p)]
    if len(paired) < 2 or len({y for y, _ in paired}) < 2:
        # An "auto" threshold fitted on one class is 0.5 with a story attached. Say which one.
        return 0.5
    return choose_threshold([y for y, _ in paired], [p for _, p in paired])


def np_is_finite(value) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _model_id(meta: dict, tta: bool) -> str:
    parts = [meta.get("architecture") or ARCHITECTURE]
    if meta.get("epoch") is not None:
        parts.append(f"epoch{meta['epoch']}")
    if tta:
        parts.append("tta-flip")
    return "_".join(str(p) for p in parts)


def _dataset(args, splits, meta) -> dict:
    all_samples = [s for part in splits.values() for s in part]
    sources: dict[str, int] = {}
    for sample in all_samples:
        sources[sample.source] = sources.get(sample.source, 0) + 1
    note = PROVENANCE if args.generated else (
        f"Corpus at {args.corpus}. Sources: {', '.join(sorted(sources))}. "
        "Split is group-atomic, so no group straddles train/val/test."
    )
    return {
        "name": Path(args.corpus).name if not args.generated else "synthetic-texture-check",
        "train": tally(splits["train"])["images"],
        "val": tally(splits["val"])["images"],
        "test": tally(splits["test"])["images"],
        "classes": ["real", "fake"],
        "sources": [{"name": name, "count": count} for name, count in sorted(sources.items())],
        "augmentations": ["random-resized-crop", "horizontal-flip", "jpeg-reencode-45-95",
                          "resize-roundtrip", "gaussian-blur", "brightness-contrast"],
        "note": note,
        "backbone_init": meta.get("init", "unknown"),
    }


def _calibration(val_y, val_probs, test_y, test_probs) -> dict:
    """Expected calibration error before and after a Platt fit on the validation split.

    The deployed head is the uncalibrated one — ``CnnRuntime`` applies a sigmoid and nothing else —
    so this reports the size of the gap rather than quietly shipping a rescaled number the service
    does not produce. Closing it means fitting the calibrator into the checkpoint, which is a
    change to the runtime contract and belongs in the same commit as that.
    """
    import math

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def logits(p):
        clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    before = ece(test_y, test_probs)
    after: float | None = None
    paired = [(y, p) for y, p in zip(val_y, val_probs, strict=True) if math.isfinite(p)]
    if len({y for y, _ in paired}) == 2 and len(paired) >= 8:
        reg = LogisticRegression(C=1e6).fit(logits([p for _, p in paired]),
                                            [y for y, _ in paired])
        after = ece(test_y, reg.predict_proba(logits(test_probs))[:, 1])
    return {
        "method": "platt (logit-scaling) fitted on the validation split, reported only",
        "before": None if before is None else round(before, 4),
        "after": None if after is None else round(after, 4),
        "note": ("ECE is the mean gap between a stated probability and the observed frequency. "
                 "The deployed head is the uncalibrated one, so 'before' is the figure that "
                 "describes what the gauge shows and 'after' is what a fitted calibrator would "
                 "have bought. It is not applied: the checkpoint scores exactly as evaluated."),
    }


def _robustness(runtime, test_samples, *, threshold: float, tta: bool) -> list[dict]:
    """Accuracy under each degradation, as the page's table wants it.

    Computed against the same threshold as the headline accuracy, so the drop is the drop and not
    a threshold artefact.
    """
    rows: list[dict] = []
    baseline = _accuracy(runtime, test_samples, threshold=threshold, tta=tta)
    rows.append({"name": "as-uploaded", "value": round(baseline, 4),
                 "note": "The corpus files themselves, through the deployed preprocessing path."})
    for label, spec in DEGRADATIONS:
        acc = _accuracy(runtime, test_samples, threshold=threshold, tta=tta, degrade=spec)
        rows.append({"name": label, "value": round(acc, 4),
                     "note": f"Accuracy after {label}; {acc - baseline:+.3f} against the baseline."})
    per_source: dict[str, list] = {}
    for sample in test_samples:
        per_source.setdefault(sample.source, []).append(sample)
    for source, group in sorted(per_source.items()):
        acc = _accuracy(runtime, group, threshold=threshold, tta=tta)
        rows.append({"name": f"source:{source}", "value": round(acc, 4),
                     "note": f"Held-out images from {source} ({len(group)} files). The pooled "
                             "accuracy above hides this number, which is the one that predicts "
                             "behaviour on a generator the corpus has never met."})
    return rows


def _accuracy(runtime, samples, *, threshold: float, tta: bool, degrade=None) -> float:
    import cv2
    import numpy as np

    hits = 0
    total = 0
    for sample in samples:
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(sample.path, dtype=np.uint8),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if degrade is not None:
            rgb = _degrade(rgb, degrade)
        value = runtime.score(runtime.encode_for_resize(rgb), tta=tta)
        if value is None:
            continue
        total += 1
        hits += int(((value >= threshold) and sample.label == 1) or
                    ((value < threshold) and sample.label == 0))
    return hits / total if total else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
