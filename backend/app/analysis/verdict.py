"""From two sub-scores and a reliability to a verdict a person can act on.

The spec asks for three outcomes plus "inconclusive", and the interesting part is not the three —
it is making "inconclusive" mean something. A tool that returns it only when two numbers are
exactly equal never shows it, and the user learns it is decoration. This layer returns it for
three separate, *named* reasons:

1. **Attribution failure.** Both sub-scores are elevated and close together, so the file is
   clearly not an untouched photograph but the evidence does not say whether it was generated or
   edited. The distinction matters to a reviewer and we refuse to guess it.
2. **Margin failure.** Neither sub-score leads the other by enough to be reported over it.
3. **Reliability failure.** The likelihoods are low *and* too little could be measured to make
   "low" meaningful. This is the case a naive implementation gets badly wrong.

Case 3 is the one worth defending. Heavily compressed input silences ELA, JPEG-ghost, block-grid
and the spectral analyser all at once. Their scores are NaN, the noisy-OR over an empty set is
0.0, and the verdict would come back "Likely Authentic" — a confident clean bill of health
awarded precisely to files where inspection was impossible. Reliability gates that: below the
floor, silence is reported as silence.

**Evidence flags are the only overrides, and they are explainable.** The prototype patched the
fused number with bare ``if`` statements that contradicted the score shown beside them. Here a
strong finding raises a named flag, the flag's effect on the number is stated in the flag's own
text, and both the number and the flag appear in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.analysis.explainers import VERDICT_COPY
from app.analysis.fusion import FusionResult
from app.analysis.reliability import MIN_MEASURED_CHECKS, Reliability

__all__ = ["EvidenceFlag", "VerdictResult", "build_flags", "decide"]

VerdictLabel = Literal["authentic", "ai_generated", "edited", "inconclusive"]

#: A sub-score at or above this counts as an affirmative finding for its hypothesis.
FINDING_FLOOR = 0.45
#: Below this overall, and with adequate reliability, nothing affirmative was found.
AUTHENTIC_CEILING = 0.35
#: How far ahead one hypothesis must be to be reported instead of "inconclusive".
MARGIN = 0.25
#: Reliability under this cannot support an "authentic" claim, only a weaker one.
MIN_RELIABILITY_FOR_AUTHENTIC = 0.45
#: Reliability under this cannot support any affirmative claim.
MIN_RELIABILITY_FOR_FINDING = 0.30

#: The metadata analyser's machine codes, in the words a reviewer reads. A flag that names the
#: evidence is auditable; one that says "a generator marker" makes the user open the raw details.
_GENERATOR_CODES = {
    "generator_parameters": "its metadata carries a generation-parameter block (prompt, sampler, "
                            "CFG scale, seed — the A1111/ComfyUI shape)",
    "xmp_ai_source_type": "its XMP packet declares an AI source type",
    "generator_software": "its software tag names a generator rather than a camera or editor",
}


@dataclass(frozen=True, slots=True)
class EvidenceFlag:
    """A named, explainable rule that adjusts the probabilistic fusion."""

    id: str
    kind: Literal["deterministic", "statistical"]
    text: str
    #: Which sub-score it floors, and to what. Stated in the flag so the UI can show the number
    #: moving and say why, rather than silently disagreeing with itself.
    raises: tuple[str, float] | None = None
    #: Forces a verdict regardless of the margin, for the cases where the evidence is a fact
    #: about the file rather than an inference from its pixels.
    forces: VerdictLabel | None = None


@dataclass(slots=True)
class VerdictResult:
    verdict: VerdictLabel
    headline: str
    detail: str
    #: Why this verdict rather than the nearby alternative. Populated for every verdict, but
    #: load-bearing for "inconclusive".
    reasons: list[str] = field(default_factory=list)
    flags: list[EvidenceFlag] = field(default_factory=list)
    #: Sub-scores *after* flags were applied, so the number and the verdict cannot disagree.
    ai_score: float = 0.0
    edited_score: float = 0.0
    overall: float = 0.0
    #: True when a flag moved a number, so the UI can label the score as flag-adjusted.
    adjusted: bool = False


def build_flags(outcomes: dict[str, Any]) -> list[EvidenceFlag]:
    """Deterministic and strong-statistical findings, as nameable rules.

    Split by kind because they deserve different language. ``deterministic`` flags are facts about
    the file — a signed manifest, an embedded parameter block — that no pixel analysis can refute.
    ``statistical`` flags are the strongest inference-grade signals we have, and they are worded as
    inferences even at their most confident.
    """
    flags: list[EvidenceFlag] = []
    outcomes = outcomes or {}

    def det(name: str) -> dict[str, Any]:
        oc = outcomes.get(name)
        return (oc.details or {}) if oc else {}

    md = det("metadata")
    if md.get("deterministic_flag"):
        declared = _GENERATOR_CODES.get(str(md["deterministic_flag"]),
                                        f"its metadata carries a {md['deterministic_flag']} marker")
        flags.append(EvidenceFlag(
            "generator_metadata", "deterministic",
            f"The file itself declares it was generated: {declared}. This is a statement by the "
            "file, not an inference from its pixels — though metadata can also be added or removed "
            "by hand.",
            raises=("ai_score", 0.85), forces="ai_generated"))

    c2 = det("c2pa")
    if c2.get("status") == "verified":
        flags.append(EvidenceFlag(
            "credential_verified", "deterministic",
            "A C2PA Content Credential signed this file and its signature verifies. Provenance is "
            "attested for the history the manifest describes.",
            raises=("ai_score", 0.0)))
    elif c2.get("status") == "invalid":
        flags.append(EvidenceFlag(
            "credential_invalid", "deterministic",
            "A C2PA manifest is present but its signature does not verify, so either the "
            "manifest or the asset was altered after signing.",
            raises=("overall", 0.50)))

    cm = det("copy_move")
    # Keyed on the structured finding, not on the technique's score. A displacement consensus is a
    # fact about the match geometry; gating the override on the score it is meant to correct would
    # make the flag a restatement of the number rather than an independent check on it.
    if (cm.get("inliers", 0) >= 12 and cm.get("spatial_separation", 0.0) >= 0.5
            and cm.get("dominant_displacement_px")):
        flags.append(EvidenceFlag(
            "copy_move_consensus", "statistical",
            f"{cm['inliers']} keypoint pairs agree on one displacement of "
            f"{tuple(int(v) for v in cm['dominant_displacement_px'])} px with spatially separate "
            f"source and destination. Displacement consensus is hard to produce by accident, so "
            f"this raises the edited score to a floor of 0.70 on its own.",
            raises=("edited_score", 0.70)))

    jg = det("jpeg_ghost")
    if jg.get("ghost_regions", 0) >= 2 and float(jg.get("peak_contrast", 0)) > 0.4:
        flags.append(EvidenceFlag(
            "jpeg_ghost_regions", "statistical",
            f"{jg['ghost_regions']} regions carry a quantisation periodicity the rest of the frame "
            f"does not share, which is an artefact of a separate encode rather than of scene "
            f"content. Edited score floored at 0.70.",
            raises=("edited_score", 0.70)))

    ela_d = det("ela")
    if ela_d.get("content_confounded") or ela_d.get("negligible_error_suppressed"):
        flags.append(EvidenceFlag(
            "ela_inconclusive", "statistical",
            "Error Level Analysis could not separate compression history from ordinary image "
            "detail on this file, so it neither supports nor refutes editing.",
            None, None))

    ai_d = det("cnn")
    if ai_d.get("detector_agreement") and ai_d["detector_agreement"].get("concur") is False:
        flags.append(EvidenceFlag(
            "detectors_disagree", "statistical",
            f"The EfficientNet and CLIP detectors disagree by "
            f"{ai_d['detector_agreement']['score_gap']}, which usually means the input sits near "
            f"a decision boundary. The margin band is widened rather than the numbers averaged.",
            None, None))

    return flags


def _apply(flags: list[EvidenceFlag], ai: float, edited: float, overall: float) -> tuple[
        float, float, float, bool]:
    adjusted = False
    for f in flags:
        if not f.raises:
            continue
        which, floor = f.raises
        if which == "ai_score" and floor > ai:
            ai = floor
            adjusted = True
        elif which == "edited_score" and floor > edited:
            edited = floor
            adjusted = True
        elif which == "overall" and floor > overall:
            overall = floor
            adjusted = True
    # overall is the probability that *either* failure mode holds, so it can never fall below the
    # larger sub-score once the sub-scores have been floored.
    overall = max(overall, ai, edited)
    return ai, edited, overall, adjusted


def decide(fusion: FusionResult, reliability: Reliability,
           outcomes: dict[str, Any] | None = None) -> VerdictResult:
    """Pick the verdict, and always say why."""
    flags = build_flags(outcomes or {})
    ai, edited, overall, adjusted = _apply(flags, fusion.ai_score, fusion.edited_score,
                                           fusion.overall)

    reasons: list[str] = []
    forced = next((f.forces for f in flags if f.forces), None)
    rel_low = reliability.score < MIN_RELIABILITY_FOR_FINDING
    rel_thin = reliability.score < MIN_RELIABILITY_FOR_AUTHENTIC
    disagree = any(f.id == "detectors_disagree" for f in flags)
    margin = MARGIN * (1.5 if disagree else 1.0)

    if forced:
        verdict = forced
        reasons.append("A deterministic flag about the file itself decides this, independently of "
                       "the pixel evidence.")
    elif sum(reliability.coverage.values()) < MIN_MEASURED_CHECKS:
        # Before the reliability comparison, not after it. Reliability is a product of penalties that
        # each describe a *reason* something was missing, and a job where nine checks never ran lands
        # at 0.31 -- a hair above the line where "authentic" becomes allowed. The count of measurements
        # is the direct fact, and it is the one that decides whether "nothing found" means anything.
        verdict = "inconclusive"
        reasons.append(
            f"Only {sum(reliability.coverage.values())} of the ten checks produced a usable "
            "measurement, so 'no evidence of manipulation' is not a claim this analysis can make. "
            "This is reported as inconclusive on purpose: the alternative is a clean-looking answer "
            "built on nothing.")
    elif rel_low and overall < AUTHENTIC_CEILING:
        verdict = "inconclusive"
        reasons.append(
            f"Nothing affirmative was found, but reliability is only "
            f"{reliability.score:.0%}: too many checks could not run for 'nothing found' to mean "
            f"'nothing there'. This is reported as inconclusive rather than authentic on purpose.")
    elif overall < AUTHENTIC_CEILING:
        verdict = "authentic"
        if rel_thin:
            reasons.append(f"No affirmative finding, at moderate reliability "
                           f"({reliability.score:.0%}).")
        else:
            reasons.append(f"Every check that could run came back quiet, at "
                           f"{reliability.score:.0%} reliability.")
    elif ai >= FINDING_FLOOR and edited >= FINDING_FLOOR and abs(ai - edited) < margin:
        verdict = "inconclusive"
        reasons.append(
            f"Both hypotheses are supported and too close to attribute (generated "
            f"{ai:.0%} vs edited {edited:.0%}; the margin must exceed {margin:.0%}). The file "
            f"shows signs of not being an untouched photograph, but the evidence does not say "
            f"whether it was synthesised or composited.")
    elif ai - edited >= margin:
        verdict = "ai_generated"
        reasons.append(f"Generation evidence leads manipulation evidence by {ai - edited:.0%}.")
    elif edited - ai >= margin:
        verdict = "edited"
        reasons.append(f"Manipulation evidence leads generation evidence by {edited - ai:.0%}.")
    else:
        verdict = "inconclusive"
        reasons.append(
            f"The leading sub-score is {max(ai, edited):.0%} but neither hypothesis clears the "
            f"other by the required {margin:.0%}, so the evidence supports 'altered, mechanism "
            f"unclear' rather than a specific claim.")

    if rel_thin and verdict != "inconclusive" and not forced:
        reasons.append(f"Reliability is {reliability.score:.0%}, which is low: "
                       f"{len(reliability.reasons)} documented reason(s) — see the reliability "
                       f"panel.")

    copy = VERDICT_COPY[verdict]
    return VerdictResult(
        verdict=verdict, headline=copy["headline"], detail=copy["detail"],
        reasons=reasons, flags=flags, ai_score=ai, edited_score=edited, overall=overall,
        adjusted=adjusted)
