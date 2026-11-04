"""The engines: fusion, reliability, verdict.

Three layers that each fail in a different direction, and the whole product's honesty rests on them
not failing in the tempting one. Fusion must not read silence as cleanliness (noisy-OR over an empty
set is 0.0, and 0.0 shown as "authentic" is the bug the reliability layer exists to block).
Reliability must stay a statement about *our* measurement rather than about the image. And the
verdict must be able to say "inconclusive" for reasons it can name, and must never phrase a
probability as proof.
"""

from __future__ import annotations

import math

import joblib
import numpy as np
import pytest

from app.analysis.explainers import VERDICT_COPY
from app.analysis.features import FEATURES, extract, vectorize
from app.analysis.fusion import (
    DETERMINISTIC_AI_P,
    EXPERT_WEIGHTS,
    FusionResult,
    _Learned,
    fuse,
    learned_fuse,
    load_learned,
    together,
)
from app.analysis.registry import names as technique_names
from app.analysis.reliability import FLOOR, Reliability, ReliabilityContext, compute_reliability
from app.analysis.verdict import (MIN_RELIABILITY_FOR_FINDING, build_flags, decide)

from .support import make_arrays, outcome, real_explanations, speaks
from .support import quiet_outcomes as _quiet

ALL = list(technique_names())


def _fusion(outcomes):
    result, _features = fuse(outcomes, make_arrays())
    return result


# ========================================================================== fusion


def test_noisy_or_accumulates_agreement_rather_than_averaging_it():
    """One screaming test is not six murmuring ones, and a mean would divide it away."""
    single = _fusion(_quiet(jpeg_ghost=speaks("jpeg_ghost", 0.5)))
    trio = _fusion(_quiet(jpeg_ghost=speaks("jpeg_ghost", 0.5),
                          block_grid=speaks("block_grid", 0.5),
                          ela=speaks("ela", 0.5)))
    assert single.edited_score == pytest.approx(EXPERT_WEIGHTS["jpeg_ghost"] * 0.5)
    assert trio.edited_score > single.edited_score
    assert trio.edited_score < max(EXPERT_WEIGHTS[n] for n in ("jpeg_ghost", "block_grid", "ela"))


def test_noisy_or_of_nothing_is_zero_and_of_one_certainty_is_one():
    assert together() == 0.0
    assert together(1.0) == 1.0
    assert together(0.5, 0.5) == pytest.approx(0.75)
    assert together(-0.2, 1.4) == pytest.approx(1.0), "terms are clamped, not trusted"


def test_a_check_that_could_not_run_contributes_no_term_at_all():
    """The difference between "silent" and "clean" is the whole design.

    Three silent manipulation checks must leave the edited score exactly where the two that spoke
    put it — averaging a zero in would pull a real finding towards "authentic".
    """
    spoke = _quiet(ela=speaks("ela", 0.8), block_grid=speaks("block_grid", 0.6))
    with_silent = dict(spoke,
                       **{name: outcome(name, float("nan"), {"applicable": False})
                          for name in ("jpeg_ghost", "noise", "copy_move")})
    assert _fusion(with_silent).edited_score == pytest.approx(_fusion(spoke).edited_score)


def test_a_technique_that_errored_is_excluded_from_every_count():
    """Not scored as zero, not counted as coverage, not present in the contributions at all."""
    base = _fusion(_quiet())
    result = _fusion(_quiet(ela=outcome("ela", 0.0, status="error",
                                        details={"error_msg": "ValueError"})))
    assert "ela" not in result.contributions
    assert result.support["edited"] == base.support["edited"] - 1


def test_the_two_sub_scores_are_combined_by_or_not_by_a_mean():
    result = _fusion(_quiet(ela=speaks("ela", 1.0), frequency=speaks("frequency", 1.0)))
    assert result.ai_score == pytest.approx(EXPERT_WEIGHTS["frequency"])
    assert result.edited_score == pytest.approx(EXPERT_WEIGHTS["ela"])
    assert result.overall == pytest.approx(
        together(result.ai_score, result.edited_score))
    assert result.overall >= max(result.ai_score, result.edited_score)


def test_a_high_metadata_score_alone_moves_nothing():
    """The Instagram trap: "no EXIF" is what every social platform produces.

    Metadata may only move the AI sub-score through a fact in the file — a generator parameter
    block — never through its score.
    """
    assert _fusion(_quiet(metadata=speaks("metadata", 0.9))).ai_score == 0.0
    flagged = _quiet(metadata=speaks("metadata", 0.9,
                                     deterministic_flag="stable_diffusion"))
    assert _fusion(flagged).ai_score == pytest.approx(DETERMINISTIC_AI_P)


def test_generator_metadata_is_recorded_as_the_reason_it_moved():
    result = _fusion(_quiet(metadata=speaks("metadata", 0.9, deterministic_flag="comfyui")))
    assert result.notes and "generator" in result.notes[0]


def test_a_verified_credential_is_near_clean_and_an_invalid_one_is_a_real_suspicion():
    """A manifest that verifies says "here is the history"; one that does not says "someone broke
    it after signing". The gap between those two is the whole point of reading the status."""
    verified = _fusion(_quiet(c2pa=speaks("c2pa", 0.0, status="verified")))
    invalid = _fusion(_quiet(c2pa=speaks("c2pa", 0.5, status="invalid")))
    absent = _fusion(_quiet(c2pa=speaks("c2pa", 0.0, status="absent")))
    assert absent.ai_score == 0.0, "absence carries no weight in either direction"
    assert verified.ai_score == pytest.approx(0.05)
    assert invalid.ai_score == pytest.approx(0.55)
    assert verified.notes == ["C2PA manifest verified"]


def test_an_unavailable_detector_is_silent_rather_than_zero():
    """A missing CNN must not read as "the model looked and called this a photograph"."""
    base = _fusion(_quiet())
    result = _fusion(_quiet(cnn=outcome("cnn", None, {"available": False,
                                                      "reason": "no weights"})))
    assert result.contributions["cnn"] == 0.0
    assert result.support["ai"] == base.support["ai"] - 1
    assert result.ai_score == pytest.approx(base.ai_score), "silence is not a low score"


def test_the_expert_prior_is_labelled_as_the_prior_it_is():
    result = _fusion(_quiet())
    assert result.source == "expert" and result.calibrated is False
    assert result.notes == ["no technique contributed positive evidence"]


def test_every_technique_in_the_registry_has_a_declared_trust_ceiling():
    assert set(EXPERT_WEIGHTS) == set(ALL)
    assert all(0.0 <= w <= 1.0 for w in EXPERT_WEIGHTS.values())


def test_the_measured_ordering_of_trust_is_the_documented_one():
    """Not a preference: these are the three findings the analyser docstrings report as process
    artefacts rather than scene content, and ELA is the one that failed a control experiment."""
    assert EXPERT_WEIGHTS["jpeg_ghost"] == EXPERT_WEIGHTS["copy_move"] > EXPERT_WEIGHTS["block_grid"]
    assert EXPERT_WEIGHTS["ela"] < EXPERT_WEIGHTS["noise"]
    assert EXPERT_WEIGHTS["metadata"] == EXPERT_WEIGHTS["c2pa"] == 0.0


def test_contributions_are_reported_for_the_why_panel():
    result = _fusion(_quiet(ela=speaks("ela", 0.4)))
    assert result.contributions["ela"] == pytest.approx(0.45 * 0.4)
    assert set(result.contributions) <= set(ALL)


# ---------------------------------------------------------------- learned path


class _FakeModel:
    """Two classifiers with the scikit-learn surface ``learned_fuse`` uses."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, row):
        return np.array([[1.0 - self.p, self.p]])


def test_a_learned_model_is_used_when_it_loads(tmp_path):
    blob = {"ai": _FakeModel(0.8), "edited": _FakeModel(0.2),
            "feature_names": list(FEATURES), "calibrated": True}
    path = tmp_path / "fusion.joblib"
    joblib.dump(blob, path)
    result, _ = fuse(_quiet(), make_arrays(), model_path=str(path))
    assert result.source == "learned" and result.calibrated is True
    assert result.ai_score == pytest.approx(0.8)
    assert result.overall == pytest.approx(together(0.8, 0.2))


def test_a_model_trained_on_a_different_feature_vector_is_refused(tmp_path):
    """A silent mis-scoring model is worse than no model, so it must fall back to the prior."""
    blob = {"ai": _FakeModel(0.9), "edited": _FakeModel(0.9),
            "feature_names": [c + "_v1" for c in FEATURES], "calibrated": True}
    path = tmp_path / "stale.joblib"
    joblib.dump(blob, path)
    assert load_learned(str(path)) is None
    result, _ = fuse(_quiet(), make_arrays(), model_path=str(path))
    assert result.source == "expert"


def test_a_model_blob_that_is_not_readable_falls_back_quietly(tmp_path):
    path = tmp_path / "broken.joblib"
    path.write_bytes(b"not a joblib file")
    assert load_learned(str(path)) is None
    assert load_learned(None) is None


def test_learned_scores_are_not_claimed_calibrated_unless_the_blob_says_so(tmp_path):
    blob = {"ai": _FakeModel(0.6), "edited": _FakeModel(0.1),
            "feature_names": list(FEATURES), "calibrated": False}
    path = tmp_path / "raw.joblib"
    joblib.dump(blob, path)
    result, _ = fuse(_quiet(), make_arrays(), model_path=str(path))
    assert result.source == "learned" and result.calibrated is False


def test_imputation_keeps_the_measured_columns_that_preserve_silence():
    """The logistic model gets NaN filled with 0; ``.measured`` is how it still tells them apart."""
    outcomes = _quiet(ela=outcome("ela", float("nan"), {"applicable": False}))
    feats = extract(outcomes, make_arrays())
    assert math.isnan(feats["ela.score"]) and feats["ela.measured"] == 0.0
    row = vectorize(feats, impute=True)
    assert row[list(FEATURES).index("ela.score")] == 0.0
    assert row[list(FEATURES).index("ela.measured")] == 0.0


def test_the_feature_vector_is_the_same_length_whatever_ran():
    full = extract(_quiet(), make_arrays())
    sparse = extract({"ela": speaks("ela", 0.2)}, make_arrays())
    assert len(full) == len(sparse) == len(FEATURES)
    assert all(not math.isnan(sparse[f"{n}.measured"]) for n in ALL)


def test_learned_fuse_consumes_the_documented_vector_layout():
    model = _Learned(_FakeModel(0.3), _FakeModel(0.4), FEATURES, calibrated=True)
    result = learned_fuse(model, extract(_quiet(), make_arrays()))
    assert isinstance(result, FusionResult) and result.source == "learned"


# ==================================================================== reliability


def _ctx(arrays=None, outcomes=None, explanations=None, **kwargs) -> ReliabilityContext:
    """A context built the way the orchestrator builds one: real explainers, unless a test supplies
    hand-written weights (which is how a test forces a coverage count without restating the rules)."""
    outcomes = outcomes if outcomes is not None else _quiet()
    return ReliabilityContext(arrays=arrays or make_arrays(), outcomes=outcomes,
                              explanations=explanations
                              if explanations is not None
                              else real_explanations(outcomes), **kwargs)


def test_a_well_preserved_jpeg_where_everything_spoke_is_highly_reliable():
    """The only thing to dock on a clean landscape JPEG is that there was no face to inspect.

    The count is 3/5, not 4/6: an ordinary photograph has no C2PA manifest and usually no face, and
    the engine says so rather than crediting checks that had nothing to read.
    """
    rel = compute_reliability(_ctx())
    assert [r["id"] for r in rel.reasons] == ["no_face"]
    assert rel.score == pytest.approx(0.88)
    assert rel.band == "high"
    assert rel.coverage == {"ai": 3, "edited": 5}


@pytest.mark.parametrize("field,value,penalty_id", [
    ("orig_h", 120, "small"),
    ("bpp", 0.1, "low_bpp"),
    ("est_jpeg_quality", 20, "heavy_jpeg"),
    ("mime", "image/png", "not_jpeg"),
    ("round_dimensions", True, "screenshot_shape"),
    ("analysis_scale", 0.3, "resized"),
])
def test_each_input_quality_condition_is_named_by_itself(field, value, penalty_id):
    """A penalty must fire for a reason a reviewer can read out of the file, not a coincidence."""
    rel = compute_reliability(_ctx(arrays=make_arrays(**{field: value})))
    ids = [r["id"] for r in rel.reasons]
    assert penalty_id in ids, f"{field}={value} produced only {ids}"


def test_a_png_says_which_three_checks_it_silenced():
    rel = compute_reliability(_ctx(arrays=make_arrays(mime="image/png")))
    reason = next(r for r in rel.reasons if r["id"] == "not_jpeg")
    assert "Error Level Analysis" in reason["reason"]
    assert reason["affects"] == "edited" and reason["severity"] == "severe"


def test_a_missing_quality_estimate_is_not_read_as_a_low_one():
    """A PNG has no JPEG quality. ``None`` must not become "quality 0", i.e. a false penalty."""
    rel = compute_reliability(_ctx(arrays=make_arrays(est_jpeg_quality=None)))
    assert "heavy_jpeg" not in [r["id"] for r in rel.reasons]


def test_penalties_compose_multiplicatively_and_never_add():
    a = compute_reliability(_ctx(arrays=make_arrays(orig_h=120)))
    both = compute_reliability(_ctx(arrays=make_arrays(orig_h=120, mime="image/png")))
    small = next(p for p in a.reasons if p["id"] == "small")["cost"]
    not_jpeg = next(p for p in both.reasons if p["id"] == "not_jpeg")["cost"]
    assert small == pytest.approx(0.45) and not_jpeg == pytest.approx(0.40)
    assert "not_jpeg" not in [p["id"] for p in a.reasons]
    assert [p["id"] for p in both.reasons if p["id"] != "not_jpeg"] == \
        [p["id"] for p in a.reasons], "adding a problem must not remove one"
    assert both.score == pytest.approx(a.score * (1 - not_jpeg))
    assert both.score < a.score


def test_the_worst_case_sits_on_the_floor_and_still_explains_itself():
    """A verdict computed on almost nothing is still a verdict; 0.00 would claim the opposite."""
    outcomes = _quiet(**{name: outcome(name, None, status="error",
                                       details={"error_msg": "boom"}) for name in ALL})
    rel = compute_reliability(_ctx(arrays=make_arrays(orig_h=100, orig_w=100, bpp=0.01,
                                                      est_jpeg_quality=5, mime="image/png",
                                                      round_dimensions=True, analysis_scale=0.1),
                                   outcomes=outcomes))
    assert rel.score == FLOOR
    assert len(rel.reasons) >= 8
    assert all(r["reason"] for r in rel.reasons), "every penalty must be explainable"
    assert rel.coverage == {"ai": 0, "edited": 0}


def test_a_failed_check_is_named_in_the_reason_the_client_sees():
    outcomes = _quiet(noise=outcome("noise", None, status="error"))
    rel = compute_reliability(_ctx(outcomes=outcomes))
    entry = next(r for r in rel.reasons if r["id"] == "technique_error")
    assert "noise" in entry["reason"]


def test_thin_coverage_is_reported_per_group_not_globally():
    """Only the sub-score that is short of evidence should be labelled short."""
    rel = compute_reliability(_ctx(coverage={"ai": 1, "edited": 5}))
    ids = {r["id"]: r["affects"] for r in rel.reasons}
    assert "thin_ai" in ids and "thin_edited" not in ids
    assert ids["thin_ai"] == "ai"


def test_coverage_is_derived_from_the_explanations_when_nobody_counted_it():
    """Which checks spoke is read off the explainers, not off a score of zero.

    A depleted spectral tail and an unloaded detector are both "no measurement", and a landscape
    JPEG with no manifest and no face leaves exactly one generation check on the record.
    """
    outcomes = _quiet(cnn=outcome("cnn", float("nan"), {"available": False}),
                      frequency=outcome("frequency", float("nan"),
                                        {"high_frequency_depleted": True}),
                      face=outcome("face", float("nan"), {"available": True, "faces": []}))
    rel = compute_reliability(_ctx(outcomes=outcomes))
    assert rel.coverage == {"ai": 1, "edited": 5}
    ids = {r["id"] for r in rel.reasons}
    assert {"thin_ai", "cnn_missing", "spectral_depletion", "no_face"} <= ids


def test_no_face_is_a_note_rather_than_an_accusation():
    outcomes = _quiet(face=outcome("face", 0.0, {"available": True, "faces": []}))
    rel = compute_reliability(_ctx(outcomes=outcomes))
    entry = next(r for r in rel.reasons if r["id"] == "no_face")
    assert "Expected for most photographs" in entry["reason"]


def test_a_context_object_that_does_not_understand_the_file_fails_closed_not_loudly():
    """The predicates consult unvalidated technique details. A broken one must not fail the job."""

    class Broken:
        @property
        def min_side(self):
            raise RuntimeError("nope")
        mime = "image/jpeg"
        bpp = 1.0
        est_jpeg_quality = 80
        round_dimensions = False
        analysis_scale = 1.0

    rel = compute_reliability(_ctx(arrays=Broken()))
    assert "small" not in [r["id"] for r in rel.reasons], \
        "a predicate that raised must cost its penalty, not invent one"
    assert rel.score == pytest.approx(1 - 0.12), "the only remaining reason is the absent face"


@pytest.mark.parametrize("score,band", [(1.0, "high"), (0.75, "high"), (0.6, "moderate"),
                                        (0.5, "moderate"), (0.4, "low"), (0.3, "low"),
                                        (0.29, "very_low"), (0.0, "very_low")])
def test_the_reliability_band_is_a_step_function_of_the_score(score, band):
    assert Reliability(score=score, reasons=(), coverage={}).band == band


def test_every_penalty_carries_its_own_documented_cost_range():
    costs = [p.cost for _f, p in compute_penalty_registry()]
    assert all(0.0 < c < 1.0 for c in costs)


def compute_penalty_registry():
    from app.analysis import reliability
    return reliability._PENALTIES


def test_a_penalty_id_is_unique_and_is_what_the_ui_keys_on():
    ids = [p.id for _f, p in compute_penalty_registry()]
    assert len(ids) == len(set(ids))


# ======================================================================= verdict


def _rel(score: float, reasons=None) -> Reliability:
    return Reliability(score=score, reasons=tuple(reasons or ()), coverage={"ai": 3, "edited": 5})


def _decide(ai: float, edited: float, overall: float | None = None, *, reliability: float = 0.8,
            outcomes=None):
    fusion = FusionResult(ai_score=ai, edited_score=edited,
                          overall=together(ai, edited) if overall is None else overall,
                          source="expert", contributions={}, support={"ai": 3, "edited": 5},
                          notes=[])
    return decide(fusion, _rel(reliability), outcomes or {})


def test_quiet_checks_and_good_reliability_read_as_authentic():
    result = _decide(0.02, 0.05)
    assert result.verdict == "authentic"
    assert result.overall < 0.35


def test_silence_at_bad_reliability_is_inconclusive_not_authentic():
    """The case the reliability layer exists for.

    Compression silences four manipulation checks at once, the noisy-OR over an empty set is 0.0,
    and a tool without this gate hands a confident clean bill of health to the files it could not
    read.
    """
    result = _decide(0.0, 0.0, overall=0.0, reliability=0.2)
    assert result.verdict == "inconclusive"
    assert any("too many checks could not run" in r for r in result.reasons)
    assert "authentic" not in result.headline.lower()


def test_moderate_reliability_still_permits_an_authentic_call_with_a_caveat():
    result = _decide(0.02, 0.05, reliability=0.4)
    assert result.verdict == "authentic"
    assert any("reliability" in r.lower() for r in result.reasons)


def test_leading_generation_evidence_is_attributed():
    assert _decide(0.9, 0.2).verdict == "ai_generated"
    assert _decide(0.2, 0.9).verdict == "edited"


def test_two_supported_hypotheses_too_close_to_separate_are_inconclusive():
    result = _decide(0.7, 0.66)
    assert result.verdict == "inconclusive"
    assert any("whether it was synthesised or composited" in r for r in result.reasons)


def test_evidence_that_clears_nothing_is_inconclusive_rather_than_the_larger_number():
    result = _decide(0.4, 0.34, overall=0.5)
    assert result.verdict == "inconclusive"
    assert any("neither hypothesis clears the other" in r for r in result.reasons)


def test_a_generator_metadata_block_forces_the_verdict_and_moves_the_number():
    outcomes = _quiet(metadata=speaks("metadata", 0.9, deterministic_flag="midjourney"))
    result = _decide(0.1, 0.05, overall=0.1, outcomes=outcomes)
    assert result.verdict == "ai_generated"
    assert result.ai_score == pytest.approx(0.85)
    assert result.adjusted is True
    assert any(f.id == "generator_metadata" for f in result.flags)


@pytest.mark.parametrize("code,phrase", [
    ("generator_parameters", "generation-parameter block"),
    ("xmp_ai_source_type", "declares an AI source type"),
    ("generator_software", "names a generator"),
])
def test_the_generator_flag_names_which_fact_in_the_file_it_read(code, phrase):
    """A flag that says only "a generator marker" sends the reviewer to the raw details dict."""
    outcomes = _quiet(metadata=speaks("metadata", 0.9, deterministic_flag=code))
    flag = next(f for f in build_flags(outcomes) if f.id == "generator_metadata")
    assert phrase in flag.text


def test_an_unrecognised_generator_code_is_echoed_rather_than_dropped():
    outcomes = _quiet(metadata=speaks("metadata", 0.9, deterministic_flag="flux_kontext"))
    flag = next(f for f in build_flags(outcomes) if f.id == "generator_metadata")
    assert "flux_kontext" in flag.text


def test_a_displacement_consensus_floors_the_edited_score_and_says_so_in_its_own_text():
    outcomes = _quiet(copy_move=speaks("copy_move", 0.4, **{
        "inliers": 30, "spatial_separation": 0.8, "dominant_displacement_px": [120, -40]}))
    result = _decide(0.1, 0.3, overall=0.3, outcomes=outcomes)
    flag = next(f for f in result.flags if f.id == "copy_move_consensus")
    assert flag.raises == ("edited_score", 0.70)
    assert "30 keypoint pairs" in flag.text
    assert result.edited_score == pytest.approx(0.70) and result.adjusted


def test_the_consensus_flag_keys_on_geometry_not_on_the_score_it_is_meant_to_correct():
    """A gate on the score would make the flag a restatement of the number."""
    weak_score = _quiet(copy_move=speaks("copy_move", 0.05, **{
        "inliers": 40, "spatial_separation": 0.9, "dominant_displacement_px": [10, 10]}))
    assert any(f.id == "copy_move_consensus" for f in build_flags(weak_score))
    thin_voting = _quiet(copy_move=speaks("copy_move", 0.9, **{
        "inliers": 3, "spatial_separation": 0.9, "dominant_displacement_px": [10, 10]}))
    assert not any(f.id == "copy_move_consensus" for f in build_flags(thin_voting))


def test_a_broken_credential_floors_overall_without_deciding_the_mechanism():
    outcomes = _quiet(c2pa=speaks("c2pa", 0.5, status="invalid"))
    result = _decide(0.1, 0.1, overall=0.1, outcomes=outcomes)
    assert any(f.id == "credential_invalid" for f in result.flags)
    assert result.overall >= 0.50
    assert result.verdict != "authentic"


def test_a_verified_credential_is_reported_without_raising_anything():
    outcomes = _quiet(c2pa=speaks("c2pa", 0.0, status="verified"))
    result = _decide(0.1, 0.1, overall=0.1, outcomes=outcomes)
    assert any(f.id == "credential_verified" for f in result.flags)
    assert result.adjusted is False


def test_detector_disagreement_widens_the_margin_it_needs_to_attributed():
    outcomes = _quiet(cnn=speaks("cnn", 0.6, **{
        "detector_agreement": {"concur": False, "score_gap": 0.31}}))
    assert _decide(0.70, 0.42, outcomes={}).verdict == "ai_generated"
    assert _decide(0.70, 0.42, outcomes=outcomes).verdict == "inconclusive"
    assert any(f.id == "detectors_disagree" for f in build_flags(outcomes))


def test_overall_can_never_be_below_the_sub_scores_it_claims_to_combine():
    result = _decide(0.6, 0.1, overall=0.0)
    assert result.overall >= max(result.ai_score, result.edited_score)


def test_a_flag_moving_a_number_is_marked_as_having_moved_it():
    outcomes = _quiet(jpeg_ghost=speaks("jpeg_ghost", 0.9,
                                     ghost_regions=4, peak_contrast=0.8))
    untouched = _decide(0.1, 0.2, overall=0.2)
    adjusted = _decide(0.1, 0.2, overall=0.2, outcomes=outcomes)
    assert not untouched.adjusted and adjusted.adjusted
    assert adjusted.edited_score == pytest.approx(0.70)


@pytest.mark.parametrize("verdict", ["authentic", "ai_generated", "edited", "inconclusive"])
def test_every_verdict_explains_itself_and_never_claims_proof(verdict):
    cases = {"authentic": (0.02, 0.05), "ai_generated": (0.9, 0.1),
             "edited": (0.1, 0.9), "inconclusive": (0.6, 0.62)}
    ai, edited = cases[verdict]
    result = _decide(ai, edited)
    assert result.verdict == verdict
    assert result.reasons, "a verdict with no reason is the coin flip this layer replaces"
    assert result.headline and result.detail
    text = " ".join([result.headline, result.detail, *result.reasons]).lower()
    for absolute in (" definitely fake", "certifiably", "proof that", "guaranteed"):
        assert absolute not in text
    assert result.headline == VERDICT_COPY[verdict]["headline"]
    assert result.detail == VERDICT_COPY[verdict]["detail"], "the copy must match the verdict"
    if verdict != "inconclusive":
        assert VERDICT_COPY[verdict]["label"].startswith("Likely")


def test_labels_always_carry_their_hedge():
    """The spec's rule: never "fake"/"real" as absolute claims."""
    for name, copy in VERDICT_COPY.items():
        joined = " ".join(copy.values()).lower()
        assert " fake" not in joined and "is real" not in joined, name


def test_an_ela_that_could_not_separate_history_from_detail_is_flagged_as_neutral():
    outcomes = _quiet(ela=speaks("ela", 0.8, content_confounded=True))
    flags = build_flags(outcomes)
    flag = next(f for f in flags if f.id == "ela_inconclusive")
    assert flag.raises is None and flag.forces is None
    assert "neither supports nor refutes" in flag.text


def test_decide_survives_no_outcomes_at_all():
    result = _decide(0.2, 0.2, overall=0.3)
    assert result.verdict in ("authentic", "inconclusive", "edited", "ai_generated")
    assert result.flags == []


def test_the_explanation_weight_and_not_the_score_defines_unavailable():
    """A 0.0 from a check that could not run looks identical to a 0.0 from a clean check.

    Only ``Explanation.weight`` tells them apart, so coverage and the missing-check penalty read it
    rather than inferring from the number — otherwise there would be two interpreters of "missing",
    one in the model's features and another in the UI, and the drift would be invisible.
    """
    spoke = compute_reliability(_ctx(outcomes=_quiet()))
    silent = compute_reliability(_ctx(
        outcomes=_quiet(cnn=outcome("cnn", 0.0, {"available": False,
                                                 "reason": "no weights"}))))
    assert spoke.coverage["ai"] == 3 and silent.coverage["ai"] == 2
    assert "cnn_missing" not in [r["id"] for r in spoke.reasons]
    assert "cnn_missing" in [r["id"] for r in silent.reasons]
    assert silent.score < spoke.score


def test_a_job_where_almost_nothing_ran_is_inconclusive_even_at_ok_reliability():
    """The verdict gate reads coverage, not the reliability number.

    On the deployed service every pooled check came back `skipped` and the analysis still reported
    `authentic` with overall 0.0: the penalties for missing checks multiply to about 0.31, which sits
    a hair above the floor where "nothing found" is allowed to become a claim. A threshold that
    decides between "authentic" and "we measured nothing" by two hundredths is not a guard. Coverage
    is the direct fact, so it is the one that decides -- here at a reliability score that the old
    logic would have waved through.
    """
    rel = Reliability(score=0.62, reasons=(), coverage={"ai": 1, "edited": 1})
    result = decide(_fusion(_quiet()), rel, _quiet())

    assert result.verdict == "inconclusive"
    assert any("Only 2 of the ten" in r for r in result.reasons), result.reasons


def test_a_deterministic_fact_still_decides_when_no_check_measured_anything():
    """The coverage gate must not swallow the one class of evidence that is not statistical.

    A signed C2PA generative manifest or a generator parameter block is a fact about the file, and it
    survives even when every pixel analyser timed out. Making inconclusive unconditional would have
    turned "the file states it was generated" into "we could not tell", which is its own lie.
    """
    outcomes = _quiet(metadata=speaks("metadata", 0.9, deterministic_flag="generator_parameters"))
    rel = Reliability(score=0.9, reasons=(), coverage={"ai": 0, "edited": 0})
    result = decide(_fusion(outcomes), rel, outcomes)

    assert result.verdict == "ai_generated"
    assert any("deterministic flag" in r for r in result.reasons), result.reasons


def test_every_check_skipped_costs_reliability_and_produces_no_claim():
    """The deployed failure, reproduced from its outcome shapes rather than its log.

    Asserted end to end because the two halves were each defensible on their own: the penalties
    described real reasons, the noisy-OR of no terms is legitimately 0.0, and the verdict layer
    faithfully reported what a 0.0 at 0.31 reliability meant under the old rules. What none of them
    asked was how many checks actually ran.
    """
    skipped = {name: outcome(name, None, {}, status="skipped") for name in technique_names()}
    skipped["cnn"] = outcome("cnn", None, {"available": False, "reason": "no weights"},
                            status="skipped")
    rel = compute_reliability(_ctx(outcomes=skipped))

    assert sum(rel.coverage.values()) == 0
    assert "almost_nothing_ran" in [r["id"] for r in rel.reasons]
    assert rel.score < MIN_RELIABILITY_FOR_FINDING, rel.score

    result = decide(_fusion(skipped), rel, skipped)
    assert result.verdict == "inconclusive"
    assert result.overall == 0.0
