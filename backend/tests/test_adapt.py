"""The adapter that turns an internal outcome into the wire shape.

This is where the product's central honesty claim is either kept or broken. A technique can execute
perfectly and still have nothing to say about a file — ELA on a PNG, frequency analysis on a
spectrum the encoder flattened, face analysis on a photo with no face. What the client must receive
for those is ``null`` and a grey card, never ``0.0``, because the UI colours a low score green and a
green means "this test passed".

The defect this file exists to prevent is real and specific: unavailable techniques shipped a numeric
0.0 with ``measured: false``, and only the boolean told the truth.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis.adapt import build_technique_view
from app.analysis.explainers import Explanation, explain
from app.analysis.types import OverlayRef, TaskOutcome

UNAVAILABLE = Explanation.unavailable("why", "nothing to measure", "no weight")


def _outcome(name: str = "ela", **kw) -> TaskOutcome:
    return TaskOutcome(name=name, details={}, **kw)


def test_nan_score_is_serialised_as_no_score():
    view = build_technique_view(_outcome(score=float("nan")), UNAVAILABLE)
    assert view.score is None
    assert view.band == "not_measured"
    assert view.measured is False
    assert view.ran is True


def test_explainer_saying_unavailable_clears_the_number():
    """A float from a check that could not measure is silence wearing a number.

    ``0.052`` on the frequency card of a heavily compressed JPEG looks like "nearly clean" to every
    scale that reads it. The raw measurement stays in ``details``.
    """
    outcome = TaskOutcome(name="frequency", status="ok", score=0.052,
                          details={"applicable": False, "note": "tail destroyed"})
    view = build_technique_view(outcome, UNAVAILABLE)
    assert view.score is None
    assert view.band == "not_measured"
    assert view.measured is False
    assert view.details["applicable"] is False


def test_measured_technique_keeps_its_score_and_band():
    outcome = TaskOutcome(name="ela", status="ok", score=0.72, details={"applicable": True})
    explanation = Explanation("why", "found", "supporting", "note")
    view = build_technique_view(outcome, explanation)
    assert view.score == pytest.approx(0.72)
    assert view.measured is True
    assert view.applicable is True
    assert view.band == "high"


@pytest.mark.parametrize("score,expected", [(0.0, "low"), (0.349, "low"), (0.35, "moderate"),
                                            (0.599, "moderate"), (0.6, "high")])
def test_band_thresholds_are_the_specified_ones(score, expected):
    outcome = TaskOutcome(name="ela", status="ok", score=score, details={"applicable": True})
    view = build_technique_view(outcome, Explanation("w", "f", "weak", "n"))
    assert view.band == expected


def test_failed_status_is_never_reported_as_a_measurement():
    outcome = TaskOutcome(name="cnn", status="error", score=None, details={},
                          error_code="boom", error_msg="it broke")
    view = build_technique_view(outcome, explain("cnn", None, {}, status="error"))
    assert view.ran is False
    assert view.measured is False
    assert view.score is None
    assert view.error == "it broke"


def test_overlay_url_only_when_an_overlay_exists():
    ref = OverlayRef(name="ela", rel_path="job/ela.png", width=8, height=8,
                     media_type="image/png", expires_at=0.0)
    with_overlay = build_technique_view(_outcome(score=0.4, overlay=ref),
                                        Explanation("w", "f", "weak", "n"),
                                        overlay_url="/api/overlays/job/ela.png")
    without = build_technique_view(_outcome(score=0.4), Explanation("w", "f", "weak", "n"),
                                   overlay_url="/api/overlays/job/ela.png")
    assert with_overlay.overlay_url == "/api/overlays/job/ela.png"
    # The caller passes a signed URL for every technique; the view decides who actually has one.
    assert without.overlay_url is None


def test_details_are_json_safe():
    """Nothing that cannot survive ``json.dumps`` may reach the wire.

    An analyser returns numpy scalars because that is what its maths produces; a raw ``int64`` or
    ``float32`` fails serialisation at the response layer, far from the array that caused it.
    """
    outcome = TaskOutcome(
        name="noise", status="ok", score=np.float32(0.3),
        details={"blocks": np.int64(128), "grid": np.array([[1, 2], [3, 4]], dtype=np.uint8),
                 "nested": {"ratio": np.float64(0.5)}, "nan": float("nan")})
    view = build_technique_view(outcome, Explanation("w", "f", "weak", "n"))
    assert not any(isinstance(v, np.generic) or isinstance(v, np.ndarray)
                   for v in view.details.values())
    # NaN becomes JSON null rather than the literal ``NaN``, which is not valid JSON and which a
    # browser's ``response.json()`` rejects outright.
    assert view.details["nan"] is None
    assert view.details["nested"]["ratio"] == pytest.approx(0.5)
    json.dumps(view.model_dump(mode="json"), allow_nan=False)
