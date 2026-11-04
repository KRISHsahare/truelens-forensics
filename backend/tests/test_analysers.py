"""Analyser tests: the ``(score, visual, details)`` contract, and that each check answers to the
artefact it claims to detect.

The threshold table below is not an accuracy claim. These are synthetic fixtures built by
``tests/fixtures/make_fixtures.py`` with the forgery mechanism known by construction, so a check
firing on them proves the analyser is wired to its own signal — that ELA reads the paste boundary,
that copy-move votes on the displacement, that metadata finds the generator block. Real accuracy is a
question about real datasets, and it is answered by ``artifacts/metrics.json`` or not at all.

Every case runs through ``worker.run_group`` so the binder, the field trimmer and the outcome wrapper
are tested alongside the maths.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests.support import FAST_NAMES, real_explanations, score_of

#: fixture -> {technique: (kind, threshold)} where ``fire`` means the score must be at or above the
#: threshold and ``quiet`` at or below it.
EXPECT: dict[str, dict[str, tuple[str, float]]] = {
    "spliced": {"ela": ("fire", 0.35), "noise": ("fire", 0.35), "copy_move": ("quiet", 0.25)},
    "copy_move": {"copy_move": ("fire", 0.35), "ela": ("quiet", 0.25)},
    "double_compressed": {"jpeg_ghost": ("fire", 0.35), "ela": ("fire", 0.35)},
    "generator_metadata": {"metadata": ("fire", 0.35)},
    "authentic": {"ela": ("quiet", 0.25), "noise": ("quiet", 0.25),
                  "copy_move": ("quiet", 0.25), "jpeg_ghost": ("quiet", 0.25),
                  "metadata": ("quiet", 0.25)},
    "no_exif": {"metadata": ("quiet", 0.25)},
    "heavy_compression": {"ela": ("quiet", 0.25), "frequency": ("quiet", 0.25)},
}


@pytest.mark.parametrize("name", FAST_NAMES)
def test_contract_holds_on_every_fixture(arrays_factory, group_runner, name):
    """No analyser may raise, hang, or return a malformed triple — on any input we accept."""
    arrays = arrays_factory("authentic")
    outcome = group_runner(arrays, [name])[name]
    assert outcome.status in ("ok", "skipped", "error", "timeout"), outcome.status
    assert isinstance(outcome.details, dict)
    assert outcome.ms >= 0
    if outcome.score is not None:
        score = float(outcome.score)
        assert not np.isinf(score), f"{name} returned {score!r}"
        if not np.isnan(score):
            assert 0.0 <= score <= 1.0, f"{name} returned {score!r}"


@pytest.mark.parametrize("fixture,checks", sorted(EXPECT.items()))
def test_expected_signals(arrays_factory, group_runner, fixture, checks):
    arrays = arrays_factory(fixture)
    outcomes = group_runner(arrays, FAST_NAMES)
    for name, (kind, threshold) in checks.items():
        score = score_of(outcomes[name])
        assert score is not None, f"{fixture}: {name} reported nothing: {outcomes[name].details}"
        if kind == "fire":
            assert score >= threshold, f"{fixture}: {name} {score:.3f} < {threshold}"
        else:
            assert score <= threshold, f"{fixture}: {name} {score:.3f} > {threshold}"


def test_png_without_jpeg_quality_still_runs_ela_and_noise(arrays_factory, group_runner):
    """Regression: a ``None`` scalar is a *result*, not a missing input.

    ``est_jpeg_quality`` is legitimately None for a PNG. The trimmer used to treat any null field as
    "not sent to this group", so every non-JPEG file reported ELA and noise as errored techniques —
    a UI showing "this check failed" for a check that had simply nothing to read.
    """
    arrays = arrays_factory("screenshot")
    assert arrays.est_jpeg_quality is None, "the fixture is a PNG; the precondition is the point"
    outcomes = group_runner(arrays, ["ela", "noise", "block_grid", "jpeg_ghost"])
    for name, outcome in outcomes.items():
        assert outcome.status == "ok", f"{name}: {outcome.error_code} {outcome.details}"
        assert outcome.error_code != "ValueError", outcome.details


def test_trim_keeps_everything_the_group_needs(arrays_factory):
    """The trimmer is a bandwidth optimisation, not a filter that may starve an analyser."""
    from app.analysis.registry import BY_NAME
    from app.analysis.worker import needed_fields, trim_arrays

    arrays = arrays_factory("spliced")
    for name in FAST_NAMES:
        trimmed = trim_arrays(arrays, (name,))
        for field in needed_fields(BY_NAME[name]):
            value = getattr(trimmed, field)
            if field in ("rgb_u8", "gray_u8", "native_rgb_u8", "native_gray_u8", "orig_bytes"):
                assert value is not None, f"{name} lost {field}"
            elif isinstance(value, np.ndarray):
                assert value.size > 0


def test_copy_move_hit_is_backed_by_a_displacement_vote(arrays_factory, group_runner):
    """A copy-move claim needs a consensus displacement, not merely two similar keypoints.

    SIFT descriptors collide on texture alone; the thing that separates a clone from a coincidence is
    many pairs voting for the *same* offset and surviving RANSAC. That accounting is what this
    asserts is reported, and that a clean file produces none of it.
    """
    keys = ("matched_pairs", "consensus_votes", "inliers", "dominant_displacement_px",
            "min_displacement_px", "region_coverage")
    hit = group_runner(arrays_factory("copy_move"), ["copy_move"])["copy_move"].details
    for key in keys:
        assert key in hit, f"copy_move details missing {key}: {sorted(hit)}"
    assert hit["consensus_votes"] > 0 and hit["inliers"] > 0, hit
    dx, dy = hit["dominant_displacement_px"]
    assert abs(dx) + abs(dy) >= hit["min_displacement_px"] > 0

    clean = group_runner(arrays_factory("authentic"), ["copy_move"])["copy_move"].details
    # A null result is reported as a null result: no vote, no offset, and a count of zero rather
    # than the keys of a detection with their values set to nothing.
    assert "dominant_displacement_px" not in clean and "consensus_votes" not in clean, clean
    assert clean["matched_pairs"] == 0 and clean["inliers"] == 0, clean


def test_techniques_declare_applicability(arrays_factory, group_runner):
    """A check that cannot speak about a file says so in its details, so the UI can grey it out."""
    outcomes = group_runner(arrays_factory("screenshot"), FAST_NAMES)
    for name, outcome in outcomes.items():
        score = score_of(outcome)
        if score is None:
            assert any(k in outcome.details for k in ("applicable", "available", "note", "reason")), (
                f"{name} scored nothing with no explanation")


#: A number with four decimals and a Python container with its quotes are the two ways a sentence
#: can stop sounding like a product. Neither can be caught by a type check, so they are caught here.
#: Three decimals is allowed: a prominence of 0.007 is a real measurement, not float noise.
_REPR_LEAK = re.compile(r"\{\S*'|'\w+':|\[[^\]]*'|None\b|\d\.\d{4,}|\[\d")


def test_no_prose_leaks_the_python_that_produced_it(arrays_factory, group_runner):
    """Every reader-facing sentence is prose, not an interpolation of a ``details`` value.

    The explainers build their sentences out of the same dicts the raw-measurements panel shows, and
    one value passed through ``str()`` is enough to put ``{'chunk': 'parameters', 'keyword':
    'steps'}`` or ``5.483319`` in front of someone reading a forensic report. Three fixtures are
    used because the branches differ: a splice fires the strong sentences, generator metadata fires
    the container-valued ones, and a screenshot leaves most checks silent.
    """
    for fixture in ("spliced", "generator_metadata", "screenshot"):
        outcomes = group_runner(arrays_factory(fixture), FAST_NAMES)
        for name, exp in real_explanations(outcomes).items():
            for field in ("found", "weight_note"):
                text = getattr(exp, field)
                bad = _REPR_LEAK.search(text)
                assert bad is None, f"{fixture}/{name}.{field}: {bad.group(0)!r} in {text!r}"

