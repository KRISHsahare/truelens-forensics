"""Run the whole analysis pipeline over the fixture set and report the verdicts.

``smoke_techniques.py`` checks that each analyser responds to the artefact it claims to detect.
This checks the layer above it: that ten per-technique scores fuse into a verdict that agrees with
the ground truth the fixtures were constructed from, and that reliability behaves sensibly when
checks cannot run.

It is a *design* check, not an accuracy claim. The labels come from synthetic constructions whose
forgery mechanism is known by definition, so passing says the plumbing and the weighting do what
was intended on cases we built to be detectable. Real accuracy needs the gated datasets.

Expectations are written as what should happen and why, so a regression reads as a broken design
decision rather than a changed number.

    python scripts/smoke_fusion.py
    python scripts/smoke_fusion.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2

from app.analysis import registry as R  # noqa: E402
from app.analysis.explainers import explain  # noqa: E402
from app.analysis.features import extract  # noqa: E402
from app.analysis.fusion import fuse  # noqa: E402
from app.analysis.reliability import ReliabilityContext, compute_reliability  # noqa: E402
from app.analysis.types import TaskOutcome  # noqa: E402
from app.analysis.verdict import decide  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.core.orchestrator import worker_config  # noqa: E402
from app.core.pipeline import build_arrays  # noqa: E402
from app.core.security import probe_header  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tests", "fixtures", "images")

#: fixture -> (verdict, min_overall, max_overall, min_reliability)
#:
#: ``heavy_compression`` and ``screenshot`` are expected INCONCLUSIVE, not authentic. That is the
#: deliberate consequence of reliability gating: those files silence most checks, and reporting a
#: confident "Likely Authentic" for a file nobody could inspect would be the exact failure the
#: reliability layer exists to prevent.
#:
#: ``small`` is inconclusive for the same reason with even less to go on.
EXPECTED: dict[str, tuple[str, float, float, float]] = {
    "authentic":          ("authentic", 0.00, 0.34, 0.40),
    "spliced":            ("edited", 0.35, 1.00, 0.00),
    "double_compressed":  ("edited", 0.35, 1.00, 0.00),
    "copy_move":          ("edited", 0.35, 1.00, 0.00),
    "generator_metadata": ("ai_generated", 0.35, 1.00, 0.00),
    "ai_generated":       ("inconclusive", 0.00, 1.00, 0.00),
    "heavy_compression":  ("inconclusive", 0.00, 0.34, 0.00),
    "screenshot":         ("inconclusive", 0.00, 1.00, 0.00),
    "no_exif":            ("authentic", 0.00, 0.34, 0.00),
    "small":              ("inconclusive", 0.00, 1.00, 0.00),
}

#: The same ``FromConfig`` namespace the orchestrator hands its workers, read from the live settings
#: rather than restated here. An inline dict once omitted ``copy_move_features``, so the copy-move
#: analyser built its detector with ``None`` keypoints, raised, and the fixture designed to prove it
#: works reported a failed check instead of a detection.
CFG = worker_config(get_settings())


def _bind(arrays, arg):
    if isinstance(arg, R.FromArrays):
        return getattr(arrays, arg.field)
    if isinstance(arg, R.FromConfig):
        return CFG.get(arg.field)
    return arg.value


def run_one(name: str, *, verbose: bool = False) -> dict:
    path = None
    for ext in (".jpg", ".png"):
        cand = os.path.join(FIXTURE_DIR, name + ext)
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        raise FileNotFoundError(f"{name} not in {FIXTURE_DIR}; run make_fixtures.py")

    with open(path, "rb") as fh:
        data = fh.read()
    arrays = build_arrays(data, probe_header(data), analysis_max_dim=2048, native_max_dim=2560)

    outcomes: dict[str, TaskOutcome] = {}
    explanations = {}
    for spec in R.TECHNIQUES:
        if spec.heavy:  # no trained weights in CI; the unavailable path is covered separately
            outcomes[spec.name] = TaskOutcome(name=spec.name, status="ok", score=None,
                                              details={"available": False,
                                                       "reason": "no weights in smoke run"}, ms=0)
            explanations[spec.name] = explain(spec.name, None,
                                              {"available": False, "reason": "no weights"})
            continue
        mod = importlib.import_module(f"app.analysis.techniques.{spec.module}")
        args = [_bind(arrays, a) for a in spec.args]
        kwargs = {k: _bind(arrays, v) for k, v in spec.kwargs.items()}
        try:
            res = mod.run(*args, **kwargs)
            oc = TaskOutcome(name=spec.name, status="ok", score=res.score,
                             details=res.details, ms=0)
        except Exception as exc:  # noqa: BLE001 - the smoke run must report, not crash
            oc = TaskOutcome(name=spec.name, status="error", score=None,
                             details={"error_msg": f"{type(exc).__name__}: {exc}"[:200]}, ms=0)
        outcomes[spec.name] = oc
        explanations[spec.name] = explain(spec.name, oc.score, oc.details, status=oc.status)

    fusion, features = fuse(outcomes, arrays, explanations)
    reliability = compute_reliability(ReliabilityContext(
        arrays=arrays, outcomes=outcomes, explanations=explanations))
    result = decide(fusion, reliability, outcomes)
    # The invariant the whole layer rests on: a score is NaN exactly when the check had nothing
    # to measure. 0.0 is a measurement; NaN is silence.
    nan_cols = [k for k, v in features.items() if isinstance(v, float) and math.isnan(v)]

    if verbose:
        print(f"\n{'=' * 74}\n{name}: {result.verdict}  "
              f"(ai={result.ai_score:.3f} edited={result.edited_score:.3f} "
              f"overall={result.overall:.3f} reliability={reliability.score:.3f})")
        print(f"  fusion source={fusion.source} calibrated={fusion.calibrated} "
              f"support={fusion.support}")
        for t, p in sorted(fusion.contributions.items(), key=lambda kv: -kv[1]):
            if p > 0:
                print(f"    + {t:13s} {p:.3f}")
        for f in result.flags:
            print(f"    flag[{f.kind}] {f.id}" + (f" raises {f.raises}" if f.raises else ""))
        for r in result.reasons:
            print(f"    why: {r}")
        for pen in reliability.reasons:
            print(f"    reliability[-{pen['cost']:.2f}] {pen['reason'][:78]}")
        print(f"    NaN columns: {len(nan_cols)}")

    return {"name": name, "verdict": result.verdict, "overall": result.overall,
            "ai": result.ai_score, "edited": result.edited_score,
            "reliability": reliability.score, "coverage": reliability.coverage,
            "nan_cols": len(nan_cols), "source": fusion.source,
            "flags": [f.id for f in result.flags], "adjusted": result.adjusted}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    names = args.only or sorted(EXPECTED)
    rows = [run_one(n, verbose=args.verbose) for n in names]

    print(f"\n{'fixture':22s} {'expected':13s} {'verdict':13s} {'ai':>6s} {'edit':>6s} "
          f"{'over':>6s} {'rel':>6s} {'cov a/e':>8s} {'NaN':>4s}")
    print("-" * 96)
    failures = []
    for row in rows:
        exp, lo, hi, minrel = EXPECTED[row["name"]]
        ok = (row["verdict"] == exp and lo - 1e-9 <= row["overall"] <= hi + 1e-9
              and row["reliability"] >= minrel - 1e-9)
        if not ok:
            failures.append(row)
        mark = "  " if ok else "XX"
        print(f"{row['name']:22s} {exp:13s} {row['verdict']:13s} {row['ai']:6.3f} "
              f"{row['edited']:6.3f} {row['overall']:6.3f} {row['reliability']:6.3f} "
              f"{str(row['coverage'].get('ai', 0)) + '/' + str(row['coverage'].get('edited', 0)):>8s} "
              f"{row['nan_cols']:4d} {mark}")

    if failures:
        print(f"\n{len(failures)} fixture(s) off expectation:")
        for row in failures:
            exp, lo, hi, minrel = EXPECTED[row["name"]]
            print(f"  {row['name']}: expected {exp} overall in [{lo},{hi}] "
                  f"reliability>={minrel}; got {row['verdict']} "
                  f"overall={row['overall']:.3f} reliability={row['reliability']:.3f}")
        sys.exit(1)
    print("\nall fusion expectations met "
          f"(fusion source: {rows[0]['source']}, no trained model present)")


if __name__ == "__main__":
    main()
