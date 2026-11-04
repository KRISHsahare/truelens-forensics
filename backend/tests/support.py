"""Helpers shared by the test modules and ``conftest.py``.

A module rather than attributes on ``conftest`` because ``tests`` is a package: ``import conftest``
from inside a test module fails, and every test would then need a fixture just to reach a constant.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from app.analysis.registry import TECHNIQUES
from app.analysis.types import TaskOutcome

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "images"

#: Names of the techniques that need no trained artifact. The heavy ones (torch) are excluded from
#: the default in-process run so the suite stays seconds rather than minutes; their unavailable path
#: is asserted explicitly in ``test_cnn.py``.
HEAVY = frozenset(spec.name for spec in TECHNIQUES if spec.heavy)
FAST_NAMES = [spec.name for spec in TECHNIQUES if not spec.heavy]


def fixture_bytes(name: str) -> bytes:
    """Read one fixture image by stem (``"spliced"`` -> ``spliced.jpg`` or ``.png``)."""
    for ext in (".jpg", ".png"):
        path = FIXTURE_DIR / (name + ext)
        if path.exists():
            return path.read_bytes()
    raise FileNotFoundError(f"{name} is not in {FIXTURE_DIR}; run tests/fixtures/make_fixtures.py")


def score_of(outcome: TaskOutcome) -> float | None:
    """The score with NaN normalised to ``None``, so "not measured" reads one way everywhere.

    ``worker`` deliberately preserves a NaN score — coercing it to 0.0 would tell the user their
    image passed a test that could not run — so every assertion about "did this check speak" has to
    understand both encodings of silence.
    """
    if outcome.score is None:
        return None
    value = float(outcome.score)
    return None if math.isnan(value) else value


def make_arrays(**overrides):
    """A ``JobArrays`` for the engines: a well-preserved 640x480 JPEG unless a test says otherwise.

    The default is deliberately the *best case* for every check — JPEG, large enough, decent
    quality — so a test that wants a penalty has to cause it on purpose rather than inheriting one.
    """
    import numpy as np

    from app.analysis.types import JobArrays

    arrays = JobArrays(
        sha256="d" * 64, mime="image/jpeg", orig_w=640, orig_h=480, file_size=180_000,
        analysis_scale=1.0, native_box=(0, 0, 640, 480),
        rgb_u8=np.zeros((480, 640, 3), np.uint8),
        gray_u8=np.zeros((480, 640), np.uint8),
        native_rgb_u8=np.zeros((480, 640, 3), np.uint8),
        native_gray_u8=np.zeros((480, 640), np.uint8),
        orig_bytes=b"\xff\xd8\xff" + b"j" * 100,
        est_jpeg_quality=88, bpp=4.7, round_dimensions=False)
    for key, value in overrides.items():
        setattr(arrays, key, value)
    return arrays


def outcome(name: str, score, details=None, *, status: str = "ok") -> TaskOutcome:
    """One technique's outcome, with the silence spelled the way the engines expect.

    ``score=None`` is passed through as ``None`` (the check produced nothing at all) while
    ``float("nan")`` stays NaN (it ran and could not measure) — the two are different claims and the
    fusion treats them differently.
    """
    return TaskOutcome(name=name, status=status,
                       score=score if score is None else float(score),
                       details=dict(details or {}))


def explanations_for(outcomes, weights=None):
    """``Explanation`` objects with the given weights, defaulting to a weight that counts.

    Reliability derives coverage from these when the caller does not supply counts, so a test about
    penalties needs a way to say "this check spoke" without restating the explainer's logic.
    """
    from app.analysis.explainers import Explanation

    weights = weights or {}
    return {
        name: Explanation(why="t", found="t", weight=weights.get(name, "supporting"),
                          weight_note="t")
        for name in outcomes
    }


def real_explanations(outcomes):
    """Run the production explainer over a set of outcomes.

    Used instead of ``explanations_for`` whenever a test cares about *which* checks spoke, because
    the explainer is the single definition of that. Hand-written weights would let a test pass while
    the real explainers marked the same details unreadable.
    """
    from app.analysis.explainers import explain

    return {name: explain(name, oc.score, oc.details, status=oc.status)
            for name, oc in outcomes.items()}


#: What each analyser's ``details`` look like when it ran on a well-preserved camera JPEG and found
#: nothing. Not decoration: the engines read "did this check speak" from the explainer's weight, and
#: the explainers need their ``applicable`` / ``dominant_quality`` / ``available`` keys before they
#: will call a score a measurement. A baseline of empty dicts quietly un-measures half the registry,
#: and every assertion about coverage then tests the fixture instead of the product.
CLEAN_DETAILS: dict[str, dict[str, Any]] = {
    "metadata": {"has_exif": True, "exif_tag_count": 24, "camera_make": "Canon",
                 "camera_model": "EOS 800D", "software": "Canon 7.1.0", "findings": []},
    "c2pa": {"status": "absent"},
    "ela": {"applicable": True, "resave_quality": 90, "p99_block_error": 2.4,
            "negligible_floor": 2.0, "hot_blocks": 4, "total_blocks": 600,
            "largest_hot_cluster_fraction": 0.02, "hot_block_ratio": 0.007,
            "detail_correlation_rho": 0.31},
    "jpeg_ghost": {"applicable": True, "dominant_quality": 88, "dominant_quality_support": 0.42,
                   "dominant_quality_floor": 0.25, "ghost_regions": 0, "total_regions": 60,
                   "per_region": []},
    "block_grid": {"applicable": True, "phase_detected": True, "global_phase": [0, 0],
                   "global_prominence": 3.1, "prominence_threshold": 0.8,
                   "regions_analysed": 60, "disagreeing_mass": 0.0, "disagreeing_phases": []},
    "noise": {"noise_floor_usable": True, "median_noise_sigma": 1.9, "min_usable_sigma": 0.6,
              "outlier_blocks": 3, "total_blocks": 600, "noise_sigma_range": [1.6, 2.3],
              "estimated_jpeg_quality": 88},
    "copy_move": {"keypoints": 214, "matched_pairs": 0, "inliers": 0, "min_displacement_px": 25},
    "face": {"available": True, "detector": "yunet", "faces": []},
    "frequency": {"spectral_slope": -1.15, "natural_slope_range": [-1.6, -0.8],
                  "high_band_slope": -1.4, "smooth_tail": False, "high_frequency_depleted": False,
                  "axis_periodicity": {"score": 0.12, "peak_z": 0.9},
                  "centre_crop_of": [640, 480], "fft_size": 512},
    "cnn": {"available": True, "effnet_p_fake": 0.02, "tta": False},
}


def quiet_outcomes(**overrides):
    """Every check ran on a well-preserved JPEG and found nothing, except where a test overrides.

    Two of the registry's checks stay silent here by design — there is no C2PA manifest on an ordinary
    photograph and usually no face — which is why the coverage they produce is 3 and 5 rather than 4
    and 6, and why a test that wants full coverage has to say so explicitly.
    """
    outcomes = {name: outcome(name, 0.0, CLEAN_DETAILS.get(name)) for name in CLEAN_DETAILS}
    outcomes.update(overrides)
    return outcomes


def speaks(name: str, score, **details):
    """An outcome from a check that *ran*, with a test's own finding on top of the clean file.

    Overriding with a bare score would erase the details that make the explainer treat the result as
    a measurement, and the technique would quietly drop out of the fusion — the test would then be
    asserting against a check that never spoke. Silence is spelled with :func:`outcome` instead.
    """
    merged = dict(CLEAN_DETAILS.get(name) or {})
    merged.update(details)
    return outcome(name, score, merged)
