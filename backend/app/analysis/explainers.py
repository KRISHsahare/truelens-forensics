"""Plain-English explanations for every technique result.

Each technique returns a ``details`` dict of machine-readable findings. That dict is what the
fusion model consumes; it is not what a person should read. This module turns it into three
sentences:

* ``why``    - why this measurement exists at all (static per technique)
* ``found``  - what *this particular image* showed (dynamic, from ``details``)
* ``weight`` - how much trust to place in that finding (dynamic)

``weight`` is the part that matters most and the part a demo usually omits. A forensic tool that
reports "ELA: 0.73 — highly suspicious" without also saying "and this number is confounded by
local image detail, which we measured at rho=0.41" is laundering a weak measurement into a
confident-sounding claim. Every branch below therefore states its own weakness explicitly.

Two rules enforced throughout:

1. **Never "fake" or "real".** The spec forbids absolute claims and the evidence does not
   support them. Vocabulary is "consistent with", "indicates", "no trace of".
2. **A skipped check is not a clean check.** Non-JPEG input, no face found, no model weights,
   too small to measure — all of these must read as *absence of evidence* and route to the
   reliability score, never as a low likelihood number. Conflating them is how a tool ends up
   confidently declaring a PNG authentic.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.analysis.registry import BY_NAME

__all__ = [
    "KNOWN_LIMITATIONS",
    "VERDICT_COPY",
    "Explanation",
    "Weight",
    "band",
    "explain",
]

#: Spec risk bands: green < 35%, amber 35-60%, red > 60%.
LOW_CEILING = 0.35
MODERATE_CEILING = 0.60

Band = Literal["low", "moderate", "high"]

#: How much evidential weight a single finding carries. ``unavailable`` means the check did not
#: run at all — which is a statement about our measurement, not about the image.
Weight = Literal["strong", "supporting", "weak", "unavailable"]

_WEIGHT_ORDER: dict[Weight, int] = {"strong": 3, "supporting": 2, "weak": 1, "unavailable": 0}


@dataclass(frozen=True, slots=True)
class Explanation:
    why: str
    found: str
    weight: Weight
    weight_note: str

    @staticmethod
    def unavailable(why: str, found: str, note: str) -> Explanation:
        return Explanation(why, found, "unavailable", note)


def band(score: float | None) -> Band:
    if score is None:
        return "low"
    if score < LOW_CEILING:
        return "low"
    if score < MODERATE_CEILING:
        return "moderate"
    return "high"


def _d(details: dict[str, Any] | None) -> dict[str, Any]:
    return details or {}


def _pct(x: float, places: int = 1) -> str:
    return f"{x * 100:.{places}f}%"


def _num(value: Any) -> str:
    """A measurement as a reader would say it: two decimals, three when the value is that small."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:.2f}" if abs(v) >= 0.01 else f"{round(v, 3):g}"


def _rng(value: Any, unit: str = "") -> str:
    """A ``[lo, hi]`` pair as the phrase it belongs in, not as the list it arrived as."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{_num(value[0])} to {_num(value[1])}{unit}"
    return _num(value)


def _list(value: Any) -> str:
    """Several short values inside a sentence, without the brackets of a Python list."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "none"
    return str(value)


def _agree(count: Any, plural: str, singular: str) -> str:
    """The verb the count demands: one region *carries*, two regions *carry*."""
    return singular if int(count or 0) == 1 else plural


#: A float that reached a sentence straight from ``details`` carries its full repr — ``5.4833`` and
#: ``-0.4457`` are the one visual tell that separates a lab instrument from a demo. Every number a
#: reader sees is rounded here, at the single place all prose leaves the module, so an explainer
#: cannot forget to format one.
_LONG_DECIMAL = re.compile(r"\d+\.\d{3,}")


def _tidy(text: str) -> str:
    return _LONG_DECIMAL.sub(lambda m: _num(m.group()), text)


def _tidied(e: Explanation) -> Explanation:
    return Explanation(_tidy(e.why), _tidy(e.found), e.weight, _tidy(e.weight_note))


# ------------------------------------------------------------------- per-technique
def _ela(score: float, d: dict[str, Any]) -> Explanation:
    why = ("JPEG is a lossy format: saving quantises each 8x8 block, discarding a predictable "
           "amount of detail. Re-saving at the *same* quality reproduces the file almost exactly, "
           "so error stays near zero everywhere. A region that was edited and re-compressed at a "
           "different quality no longer sits on that fixed point, and accumulates visibly more "
           "error than its surroundings.")
    if not d.get("applicable", True):
        return Explanation.unavailable(
            why, "This file is not a JPEG, so it has no prior compression history to re-encode "
                 "against.",
            "The check did not run. This is recorded as missing evidence and lowers the "
            "reliability score; it is not a sign the image is clean.")
    q = d.get("resave_quality")
    src = d.get("resave_quality_source", "")
    qtxt = f"quality {q}" if q else "an assumed quality"
    if src == "fallback_default":
        qtxt += " (the file's quantisation table was unreadable, so this is a guess)"

    if d.get("negligible_error_suppressed"):
        return Explanation(
            why,
            f"Re-encoding at this file's own JPEG {qtxt} reproduces it almost exactly. The 99th "
            f"percentile of per-block error is {d.get('p99_block_error')}, below the "
            f"{d.get('negligible_floor')} floor, so no region has a different compression history.",
            "weak",
            "A negative ELA result is weak evidence. It says the re-encode found nothing, not "
            "that nothing was done — an edit re-saved at the same quality as the original is "
            "invisible to this method by construction.")
    if d.get("content_confounded"):
        rho = d.get("detail_correlation_rho")
        return Explanation(
            why,
            f"Error is unevenly distributed (the brightest blocks reach "
            f"{d.get('p99_block_error')} against a floor of {d.get('negligible_floor')}), but it "
            f"tracks local image detail with rank correlation rho={rho}. Detailed regions resist "
            f"quantisation more than smooth ones, so this pattern is explained by the photograph's "
            f"content rather than by a second compression history. The score has been discounted "
            f"accordingly.",
            "weak",
            "This is the central weakness of ELA and it is measured, not assumed: in controlled "
            "testing, identical content with one fewer encode produced no signal, while different "
            "content with the *same* history produced a stronger signal than any real splice. ELA "
            "cannot by itself separate compression history from local detail.")
    b = band(score)
    if b == "high":
        found = (f"{d.get('hot_blocks')} of {d.get('total_blocks')} blocks {_agree(d.get('hot_blocks'), 'carry', 'carries')} anomalous "
                 f"re-encoding error, and the largest contiguous group accounts for "
                 f"{_pct(d.get('largest_hot_cluster_fraction', 0.0))} of them. Error is clustered "
                 f"in space rather than scattered, which is the shape a composited region makes. "
                 f"It is not correlated with image detail (rho="
                 f"{d.get('detail_correlation_rho')}).")
        weight, note = "supporting", (
            "Spatial clustering after removing the content confound is a real signal, but ELA is "
            "still a supporting measurement only. It is never sufficient on its own to reach a "
            "verdict, and a re-save of the whole image erases it entirely.")
    elif b == "moderate":
        found = (f"Some blocks carry more re-encoding error than others "
                 f"({_pct(d.get('hot_block_ratio', 0.0), 2)} of blocks, largest cluster "
                 f"{_pct(d.get('largest_hot_cluster_fraction', 0.0))}), but the pattern is not "
                 f"clearly separated from normal variation.")
        weight, note = "weak", "An ambiguous ELA result carries little weight either way."
    else:
        found = (f"Re-encoding error is uniform across the frame at this file's own "
                 f"{qtxt}. No region stands out.")
        weight, note = "weak", (
            "Uniform error is consistent with a single compression pass, but it is also what any "
            "whole-image re-save produces. Treat this as the absence of a finding, not as proof.")
    return Explanation(why, found, weight, note)


def _noise(score: float, d: dict[str, Any]) -> Explanation:
    why = ("A camera sensor adds photon shot and read noise at a characteristic level, and that "
           "level is essentially constant across the whole frame. A region pasted in from a "
           "different photograph carries a different sensor's noise floor, which survives editing "
           "long after visible seams have been retouched.")
    if not d.get("noise_floor_usable", True):
        return Explanation.unavailable(
            why,
            f"The residual noise level is {d.get('median_noise_sigma')} sigma, below the "
            f"{d.get('min_usable_sigma')} minimum needed to compare blocks meaningfully.",
            "There is not enough sensor noise left in this file to measure. That happens after "
            "heavy compression or aggressive noise reduction, and it lowers the reliability score "
            "rather than counting as evidence of authenticity.")
    if d.get("low_contrast_noise"):
        return Explanation(
            why,
            f"Noise levels are near-uniform across the frame: the spread between the quietest and "
            f"noisiest blocks is {d.get('absolute_sigma_spread')} sigma, below the 0.35 threshold.",
            "weak",
            "Uniform noise is consistent with a single capture, but uniformity is also what "
            "compression produces. Absence of a finding, not proof.")
    q = d.get("estimated_jpeg_quality")
    confounded = bool(d.get("compression_confounded"))
    b = band(score)
    if b == "high":
        found = (f"{d.get('outlier_blocks')} of {d.get('total_blocks')} blocks {_agree(d.get('outlier_blocks'), 'have', 'has')} a noise floor "
                 f"far from the frame median ({d.get('median_noise_sigma')} sigma), and "
                 f"{_pct(d.get('largest_outlier_cluster_fraction', 0.0))} of them "
                 f"{_agree(d.get('outlier_blocks'), 'form', 'forms')} one "
                 f"contiguous region. Noise range across the frame: "
                 f"{_rng(d.get('noise_sigma_range'), ' sigma')}.")
        weight = "supporting"
        note = ("A contiguous region with a different noise floor is one of the stronger splice "
                "indicators available, because noise is hard to forge deliberately. It is still "
                "not conclusive: heavy local retouching and denoising produce the same pattern.")
    elif b == "moderate":
        found = (f"{d.get('outlier_blocks')} blocks deviate from the frame's noise floor, but they "
                 f"do not form a clearly contiguous region.")
        weight, note = "weak", "Scattered outlier blocks are what texture and compression produce."
    else:
        found = (f"Noise floor is consistent across the frame "
                 f"(median {d.get('median_noise_sigma')} sigma).")
        weight, note = "weak", "Consistent noise is a mild point in favour of a single capture."
    if confounded:
        note += (f" Note: this file's JPEG quality is estimated at {q}, at or below the "
                 f"confounding threshold of 55. Quantisation is itself applied per 8x8 block at "
                 f"that level, so per-block noise estimates partly measure compression rather "
                 f"than sensor history, and the score has been discounted.")
        if weight == "supporting":
            weight = "weak"
    return Explanation(why, found, weight, note)


def _frequency(score: float, d: dict[str, Any]) -> Explanation:
    why = ("Natural photographs follow an approximate 1/f^alpha power law: their amplitude "
           "spectrum is close to a straight line in log-log space, with a steadily decaying tail. "
           "Generative models learn the low frequencies well and systematically mis-model the "
           "tail. Separately, upsampling by transposed convolution — used by every GAN generator "
           "and most diffusion decoders — multiplies the signal by a fixed kernel on a regular "
           "grid, which puts a *peak* at a specific radius along the horizontal and vertical "
           "axes. A camera has no periodic structure and cannot produce that peak.")
    ap = d.get("axis_periodicity") or {}
    hi = d.get("high_band_slope")
    crop = d.get("centre_crop_of")
    if d.get("smooth_tail"):
        found = (f"The high-frequency tail is almost flat (slope {hi} against a natural range of "
                 f"{_rng(d.get('natural_slope_range'))}, threshold {d.get('smooth_tail_threshold')}). "
                 f"Real optics and a real sensor always leave a decaying tail; a flat one means "
                 f"the fine detail was synthesised rather than captured.")
        weight = "strong"
        note = ("This is the most specific measurement in the frequency analyser, because a flat "
                "tail has no natural explanation. It is still one signal: aggressive resizing and "
                "heavy compression also flatten the tail, which is why the depletion case below "
                "is kept separate.")
    elif d.get("high_frequency_depleted"):
        return Explanation.unavailable(
            why,
            f"The high-frequency tail falls off far faster than a camera-produced image would "
            f"(slope {hi}, below the {d.get('depleted_threshold')} depletion threshold).",
            "This is NOT evidence the image is synthetic. Heavy JPEG compression, aggressive "
            "resizing and screenshots all destroy the tail. What it means is that the "
            "fine-detail traces this analyser depends on are largely gone, so the result is "
            "untrustworthy and the reliability score is reduced. One caveat: this measurement "
            "is global to the crop, so a single separately-compressed region inside it — a "
            "pasted JPEG, say — can flatten the average tail on its own. A depleted tail is "
            "therefore ambiguous between 'the whole file is compressed' and 'part of it is', "
            "and the other techniques are what disambiguate.")
    elif ap.get("score", 0.0) > 0.35:
        found = (f"A periodic peak sits at radius {ap.get('peak_radius')} — "
                 f"{_pct(ap.get('peak_as_fraction_of_nyquist', 0.0), 0)} of Nyquist — with "
                 f"excess energy {ap.get('peak_z')} sigma above the local trend "
                 f"(harmonic confirmation {ap.get('harmonic_confirmed')}). Periodic structure "
                 f"along the axes is what transposed-convolution upsampling leaves behind.")
        weight = "supporting"
        note = ("Periodic axis energy is a genuine generative artefact, but the same scan also "
                "picks up JPEG blocking and very sharp synthetic edges. It is corroborating "
                "evidence, not a verdict.")
    else:
        found = (f"The spectrum decays naturally: overall slope {d.get('spectral_slope')} inside "
                 f"the natural range {_rng(d.get('natural_slope_range'))}, high-band slope {hi}, and no "
                 f"periodic peak above threshold (strongest axis excess "
                 f"{ap.get('peak_z')} sigma).")
        weight, note = "weak", (
            "A natural-looking spectrum is a mild point against generation. It is weak evidence "
            "because newer generators are trained specifically to reproduce natural spectral "
            "statistics.")
    if crop:
        found += (f" Measured on a {d.get('fft_size')}px centre crop of the never-resampled "
                  f"{crop[0]}x{crop[1]} native region.")
    return Explanation(why, found, weight, note)


def _jpeg_ghost(score: float, d: dict[str, Any]) -> Explanation:
    why = ("Every JPEG encode leaves a fingerprint in the DCT coefficients: at a given quality, "
           "each frequency's quantisation step is a known integer, and the surviving coefficients "
           "cluster near multiples of it. That clustering — striation — is detectable per 8x8 "
           "block. A region compressed at a *different* quality before being pasted in striates at "
           "a different step than the rest of the frame, even when the final save has visually "
           "erased the seam.")
    if not d.get("applicable", True):
        return Explanation.unavailable(
            why, "This file is not a JPEG, so it carries no quantisation history to read.",
            "The check did not run. Recorded as missing evidence; it lowers reliability and is "
            "not a sign the image is clean.")
    if "dominant_quality" not in d:
        return Explanation.unavailable(
            why, d.get("note", "No informative quantisation step could be tested at this size."),
            "The image is too small, or too heavily compressed, for the striation test to have "
            "enough usable coefficients. No conclusion either way.")
    regions = d.get("per_region") or []
    if regions:
        qs = sorted({r["ghost_quality"] for r in regions})
        cells = ", ".join(f"({r['region'][0]},{r['region'][1]}) at q{r['ghost_quality']} "
                          f"[+{r['contrast']}]" for r in regions[:6])
        found = (f"{d.get('ghost_regions')} of {d.get('total_regions')} regions {_agree(d.get('ghost_regions'), 'carry', 'carries')} a secondary "
                 f"quantisation periodicity the rest of the frame does not share, at "
                 f"quality{'s' if len(qs) > 1 else ''} {' and '.join(str(q) for q in qs)}. "
                 f"The frame's own history reads as q{d.get('dominant_quality')}. "
                 f"Affected regions: {cells}.")
        weight = "strong"
        note = ("This is among the strongest findings available, because quantisation periodicity "
                "cannot be produced by image content — it is an artefact of a specific encoder "
                "run. Caveat: a region that has been re-encoded enough times converges onto the "
                "final lattice and stops being detectable, so a negative result means little.")
    elif d.get("dominant_quality") is None:
        return Explanation(
            why,
            f"No quantisation periodicity was detectable in any region (best support "
            f"{d.get('dominant_quality_support')}, below the {d.get('dominant_quality_floor')} "
            f"floor), so not even a single compression history could be established.",
            "unavailable",
            "The test found nothing to measure. This is absence of evidence, and it lowers the "
            "reliability score; it is not evidence of absence.")
    else:
        found = (f"Every region shares one quantisation periodicity, consistent with the whole "
                 f"frame having been saved once at approximately q{d.get('dominant_quality')} "
                 f"(support {d.get('dominant_quality_support')}). No second history was found.")
        weight, note = "weak", (
            "A single shared history is a mild point in favour of an uncomposited image, but "
            "repeated re-encoding erases prior periodicity, so this cannot rule out a splice.")
    return Explanation(why, found, weight, note)


def _block_grid(score: float, d: dict[str, Any]) -> Explanation:
    why = ("A JPEG encoder divides the image into 8x8 blocks starting at pixel (0,0). Every part "
           "of a singly-encoded file therefore shares one block phase. If a region was cropped "
           "from another JPEG and pasted at an offset that is not a multiple of 8, its blocks "
           "land on a *different* phase — a discontinuity that survives retouching and is invisible "
           "to the eye.")
    if not d.get("applicable", True):
        return Explanation.unavailable(
            why, "This file is not a JPEG, so it has no 8x8 DCT block grid to align.",
            "The check did not run; recorded as missing evidence.")
    if not d.get("phase_detected", True):
        return Explanation.unavailable(
            why,
            f"No dominant block phase could be established (prominence "
            f"{d.get('global_prominence')} below the {d.get('prominence_threshold')} threshold), "
            f"so there is no reference grid to compare regions against.",
            "This happens on non-JPEG-derived content, very small images, or after heavy "
            "processing. It lowers reliability rather than counting as clean.")
    disagree = d.get("disagreeing_mass", 0.0)
    if band(score) != "low":
        found = (f"The frame's dominant block phase is {d.get('global_phase')}, but "
                 f"{_pct(disagree)} of the analysed mass sits at a different phase "
                 f"({_list(d.get('disagreeing_phases'))}) across {d.get('distinct_strong_phases')} "
                 f"distinct strong phases. Regions disagreeing with the global grid are marked on "
                 f"the overlay.")
        weight = "supporting"
        note = ("Two block phases in one JPEG is a strong structural hint that content was "
                "composited from separate encodes. It is supporting rather than conclusive: some "
                "editors re-encode whole images on a shifted grid, and a crop-and-resave can "
                "produce the same discontinuity with no malicious intent.")
    else:
        found = (f"All {d.get('regions_analysed')} regions share block phase "
                 f"{d.get('global_phase')} (prominence {d.get('global_prominence')}). "
                 f"One consistent 8x8 grid across the frame.")
        weight, note = "weak", (
            "A single shared grid phase is consistent with one encode. It is weak evidence: an "
            "edit pasted at an 8-aligned offset is invisible to this test by construction.")
    return Explanation(why, found, weight, note)


def _copy_move(score: float, d: dict[str, Any]) -> Explanation:
    why = ("Clone-stamp forgery copies part of an image onto another part of the same image. "
           "SIFT keypoints are matched against each other rather than against a second image; "
           "genuine clones survive a Lowe ratio test and, crucially, agree on a common "
           "displacement vector. Random matches do not, so voting on displacement is what makes "
           "the finding specific.")
    inliers = d.get("inliers", 0)
    if inliers:
        found = (f"{inliers} keypoint pairs agree on a common displacement of at least "
                 f"{d.get('min_displacement_px')}px (from {d.get('matched_pairs')} candidate "
                 f"matches over {d.get('keypoints')} keypoints). Matched regions are linked on "
                 f"the overlay.")
        weight = "strong"
        note = ("Displacement-consistent keypoint clusters are difficult to produce by accident, "
                "which is why a strong copy-move finding raises an explainable evidence flag "
                "rather than merely adding to the score. It is not infallible: repeating "
                "structures like fence posts, windows and foliage generate genuine self-matches.")
    elif d.get("matched_pairs"):
        found = (f"{d.get('matched_pairs')} keypoint pairs matched, but none reached a consistent "
                 f"displacement, so they are not evidence of cloning.")
        weight, note = "weak", "Isolated matches without displacement consensus are normal."
    else:
        found = (f"No duplicated regions found ({d.get('keypoints')} keypoints detected, "
                 f"{d.get('matched_pairs')} candidate matches).")
        weight, note = "weak", (
            "This detector is deliberately conservative. It is tuned to avoid firing on "
            "repetitive natural texture, which means small or heavily compressed clones can be "
            "missed entirely — a negative result here is weak.")
    return Explanation(why, found, weight, note)


def _face(score: float, d: dict[str, Any]) -> Explanation:
    why = ("Faces are the most common deepfake target and the hardest to composite convincingly. "
           "Four checks run per detected face, using five-point landmarks rather than a Haar "
           "cascade: the sharpness transition at the blend boundary, seam gradient excess just "
           "inside that boundary, consistency of the specular reflection (catchlight) between the "
           "two eyes, and left/right lighting direction. A pasted face usually fails at least one.")
    if not d.get("available", True):
        return Explanation.unavailable(
            why, f"The face detector could not be loaded ({d.get('note', 'unavailable')}).",
            "Face-specific checks were skipped entirely. Recorded as missing evidence.")
    if not d.get("faces"):
        return Explanation.unavailable(
            why, "No face was detected in this image.",
            "Face checks were skipped, which lowers the reliability score. This is emphatically "
            "not evidence the image is authentic — most photographs contain no face, and a "
            "detector miss is not a clean result.")
    labels = {"sharpness": "blend-boundary sharpness", "seam": "seam gradient excess",
              "catchlight": "catchlight consistency", "lighting": "left/right lighting direction"}
    per_face = d.get("per_face") or []
    parts = []
    for i, f in enumerate(per_face[:4], start=1):
        dom = f.get("dominant_signal")
        parts.append(f"face {i}: score {f.get('score')} (strongest: {labels.get(dom, dom)})")
    found = f"{d.get('faces')} face(s) detected. " + "; ".join(parts)
    b = band(score)
    if b == "high":
        weight = "supporting"
        note = ("Face-specific inconsistencies are meaningful when several checks agree, but each "
                "one individually has benign explanations: a catchlight can be genuinely absent in "
                "one eye under diffuse lighting, and a retouched face has a real blend boundary. "
                "This contributes to the edited score; it does not decide it.")
    elif b == "moderate":
        weight, note = "weak", "A partial face signal is easily explained by retouching or lighting."
    else:
        weight, note = "weak", (
            "No face inconsistency found. Weak evidence: these checks only cover the four "
            "mechanisms they test, and a good compositing job defeats all four.")
    return Explanation(why, found, weight, note)


def _metadata(score: float, d: dict[str, Any]) -> Explanation:
    why = ("Camera files carry a dense, mutually consistent EXIF block: make, model, exposure "
           "time, aperture, ISO, focal length and three matching timestamps. Editing software "
           "rewrites or strips it. AI tools go further and embed their own generation parameters "
           "— prompt, sampler, CFG scale, seed, model hash — in PNG text chunks or XMP. Metadata "
           "is trivially forgeable, so it is treated as provenance context, never as proof.")
    findings = d.get("findings") or []
    hits = d.get("generator_hits") or []
    flagged = [f for f in findings if f.get("code") != "clean"]
    if hits:
        listed = "; ".join(f"“{h.get('keyword')}” in {h.get('chunk')}" for h in hits[:3])
        found = (f"Generator metadata found: {listed}. "
                 f"Camera identity present: {d.get('camera_make') or 'none'}.")
        weight = "strong"
        note = ("An embedded generation-parameter block is close to conclusive when the tool "
                "actually wrote it — but metadata is plain text that anyone can add or remove. "
                "It is strong corroboration, not proof, and a file with none is not thereby "
                "authentic.")
    elif flagged:
        found = "; ".join(f.get("text", f.get("code", "")) for f in flagged[:5])
        weight, note = "weak", (
            "Missing or inconsistent metadata is extremely common and mostly innocent. Every "
            "messaging app, social platform and browser download strips EXIF. These findings are "
            "reported for transparency and move the reliability score far more than the verdict.")
    else:
        found = (f"Metadata is internally consistent: {d.get('exif_tag_count')} EXIF tags, "
                 f"camera {d.get('camera_make') or 'unknown'} {d.get('camera_model') or ''}, "
                 f"software {d.get('software') or 'none recorded'}. No editing or generation "
                 f"markers.")
        weight, note = "weak", (
            "Consistent camera metadata is a mild point in favour of a direct-from-camera file. "
            "It is trivially copied from another photograph, so it proves little on its own.")
    if d.get("has_xmp") and d.get("xmp_snippets"):
        found += f" XMP present with {len(d['xmp_snippets'])} notable snippet(s)."
    return Explanation(why, found, weight, note)


def _c2pa(score: float, d: dict[str, Any]) -> Explanation:
    why = ("C2PA Content Credentials is a cryptographically signed manifest describing an asset's "
           "capture and edit history. Unlike EXIF it cannot be silently altered: the signature "
           "covers the assertions, so tampering invalidates it. When present and valid it is the "
           "only genuinely trustworthy provenance signal available — and when absent it says "
           "nothing at all, because adoption is still partial.")
    status = d.get("status", "absent")
    if status == "verified":
        return Explanation(
            why,
            f"A C2PA manifest is present and its signature verified. Signer: "
            f"{d.get('signer') or 'unknown'}. Assertions: {d.get('assertion_count')}.",
            "strong",
            "A verified manifest is strong provenance evidence for the *history it describes*. "
            "It does not certify that the pixels are unaltered beyond what the manifest claims, "
            "and a manifest can itself assert that AI generation took place.")
    if status == "invalid":
        return Explanation(
            why,
            f"A C2PA manifest is present but its signature did NOT verify "
            f"({d.get('error') or 'validation failed'}).",
            "strong",
            "An invalid signature means the manifest was altered after signing, or the asset was "
            "modified in a way the signature covers. This is a genuine red flag — though a "
            "broken credential chain from an over-eager platform re-encode can look identical.")
    if status == "absent":
        return Explanation.unavailable(
            why, "No C2PA manifest is present in this file.",
            "The overwhelming majority of photographs have no Content Credentials, including "
            "perfectly authentic ones. Absence carries no evidential weight in either direction "
            "and does not affect the verdict.")
    return Explanation.unavailable(
        why, f"Content Credentials could not be checked (status: {status}).",
        d.get("note") or "The check did not complete, so it contributed nothing.")


def _cnn(score: float, d: dict[str, Any]) -> Explanation:
    why = ("A learned detector sees patterns no hand-written measurement captures. This one is "
           "EfficientNet-B0 fine-tuned on labelled real/generated data with compression, resize "
           "and blur augmentation so it survives messenger-style recompression. Grad-CAM shows "
           "which pixels drove the decision. Optionally a frozen-CLIP linear probe (UnivFD-style) "
           "runs alongside: it has a completely different capacity and training set, so agreement "
           "between the two is worth more than either number alone.")
    if not d.get("available", True):
        return Explanation.unavailable(
            why, f"The learned detector is not loaded ({d.get('reason', 'no weights')}).",
            "This technique contributed nothing to the verdict. It is recorded as NaN in the "
            "fusion features — not as zero, because a missing measurement must not be mistaken "
            "for a clean one — and it lowers the reliability score.")
    parts = []
    if d.get("effnet_p_fake") is not None:
        parts.append(f"EfficientNet-B0 puts the probability of generation at "
                     f"{_pct(d['effnet_p_fake'])}")
    if d.get("clip_probe_p_fake") is not None:
        parts.append(f"the CLIP probe at {_pct(d['clip_probe_p_fake'])}")
    agree = d.get("detector_agreement") or {}
    if agree:
        parts.append(f"the two {'concur' if agree.get('concur') else 'DISAGREE'} "
                     f"(gap {agree.get('score_gap')})")
    tta = "with test-time augmentation" if d.get("tta") else "without augmentation"
    found = ", ".join(parts) + f". Inference ran {tta}."
    if not agree.get("concur", True) and agree:
        weight = "weak"
        note = ("The two detectors disagree, which is itself informative: it means the input sits "
                "near a decision boundary that models of very different capacity resolve "
                "differently. Disagreement should widen the inconclusive band rather than be "
                "averaged away.")
    elif band(score) == "high":
        weight = "strong"
        note = ("A confident learned detector is the strongest single AI-generation signal "
                "available here, but it generalises worst of all the techniques: it is trained on "
                "specific generators and degrades on ones it has never seen. Check the Grad-CAM "
                "overlay — if the highlighted regions are semantically meaningless, the "
                "confidence is likely spurious.")
    else:
        weight, note = "supporting", (
            "A low detector score means this model found no generation artefacts it recognises. "
            "It does not mean none are present.")
    return Explanation(why, found, weight, note)


_EXPLAINERS: dict[str, Callable[[float, dict[str, Any]], Explanation]] = {
    "ela": _ela,
    "noise": _noise,
    "frequency": _frequency,
    "jpeg_ghost": _jpeg_ghost,
    "block_grid": _block_grid,
    "copy_move": _copy_move,
    "face": _face,
    "metadata": _metadata,
    "c2pa": _c2pa,
    "cnn": _cnn,
}

#: Why the fallback exists: a technique that errors out still needs an honest sentence, and
#: silently dropping it from the report would hide the failure.
_FALLBACK_WHY = "This check did not produce a usable result."


def explain(name: str, score: float | None, details: dict[str, Any] | None,
            *, status: str = "ok") -> Explanation:
    """Build the three-sentence explanation for one technique result.

    Never raises. An unknown technique, a crashed one, or a missing ``details`` key all degrade to
    an honest "we could not measure this" rather than propagating a KeyError into the API layer.
    """
    return _tidied(_explain_raw(name, score, details, status=status))


def _explain_raw(name: str, score: float | None, details: dict[str, Any] | None,
                 *, status: str = "ok") -> Explanation:
    d = _d(details)
    fn = _EXPLAINERS.get(name)
    why = BY_NAME[name].summary if name in BY_NAME else _FALLBACK_WHY
    if fn is None:
        return Explanation.unavailable(
            why, f"No explanation is registered for '{name}'.",
            "This result was not interpreted; it contributed nothing to the verdict.")
    if status != "ok":
        label = {"error": "errored", "timeout": "timed out", "skipped": "was skipped"
                 }.get(status, status)
        spec_why = _EXPLAINER_WHY.get(name, why)
        return Explanation.unavailable(
            spec_why, f"This check {label}"
                      + (f": {d['error_msg']}" if d.get("error_msg") else "."),
            "A failed check is missing evidence, not a clean result. It lowers the reliability "
            "score and is excluded from the fusion features rather than being scored as zero.")
    try:
        return fn(float(score) if score is not None else 0.0, d)
    except Exception as exc:  # pragma: no cover - defensive, explanation must never break a job
        return Explanation.unavailable(
            _EXPLAINER_WHY.get(name, why),
            f"The result could not be interpreted ({type(exc).__name__}).",
            "The raw measurements are still shown in the details panel, but no plain-English "
            "conclusion is offered because one could not be derived safely.")


#: The full "why this matters" text is needed even when the check failed, so the user still learns
#: what they are missing. Extracted once at import rather than by calling the explainer.
_EXPLAINER_WHY: dict[str, str] = {}
for _name, _fn in _EXPLAINERS.items():
    try:
        _EXPLAINER_WHY[_name] = _fn(0.0, {"applicable": False, "available": False, "faces": 0,
                                          "status": "absent",
                                          "phase_detected": False}).why
    except Exception:  # pragma: no cover
        _EXPLAINER_WHY[_name] = BY_NAME[_name].summary if _name in BY_NAME else _FALLBACK_WHY


# ------------------------------------------------------------------- verdict copy
#: Deliberately avoids "real" and "fake". Every string states what the evidence supports and what
#: it does not, and every one is a probability statement.
VERDICT_COPY: dict[str, dict[str, str]] = {
    "authentic": {
        "label": "Likely Authentic",
        "headline": "No manipulation traces were found.",
        "detail": "Every technique that could run came back quiet, and the file's structure is "
                  "consistent with a single capture and encode. This is a probability, not a "
                  "proof: a sufficiently careful edit, or one followed by a whole-image re-save, "
                  "leaves no trace these methods can detect.",
    },
    "ai_generated": {
        "label": "Likely AI-Generated",
        "headline": "The evidence points to a synthetic image rather than an edited photograph.",
        "detail": "The signals that fired are specific to generation — spectral-tail shape, "
                  "periodic upsampling artefacts, generator metadata, and the learned detector — "
                  "rather than to compositing. Detectors of this kind generalise worst to "
                  "generators released after their training data, so confidence should be read "
                  "together with the reliability score.",
    },
    "edited": {
        "label": "Likely Edited / Manipulated",
        "headline": "The evidence points to a real photograph that has been altered.",
        "detail": "The signals that fired localise to regions of the frame — compression history, "
                  "noise floor, block-grid phase, cloned keypoints — which is the signature of "
                  "compositing rather than generation. Note that benign editing (retouching, "
                  "cropping, re-saving) produces some of the same traces.",
    },
    "inconclusive": {
        "label": "Inconclusive",
        "headline": "The available evidence does not support a confident conclusion.",
        "detail": "Either the two sub-scores are too close to separate, or too many checks could "
                  "not run for the result to be trusted. Reporting a guess here would be worse "
                  "than reporting nothing: read the reliability panel for what was missing.",
    },
}

#: Shown in the expandable "Known limitations" panel on the analyzer, and reproduced verbatim in
#: the exported PDF report. Written from measured behaviour, not from aspiration.
KNOWN_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "title": "Probability, not proof",
        "body": "No forensic tool can prove an image is authentic. Proving a negative would "
                "require ruling out every manipulation method that exists and every one that "
                "does not yet. What these techniques can do is detect the traces that known "
                "methods leave, and say how consistent the file is with a single capture.",
    },
    {
        "title": "ELA cannot separate compression history from image detail",
        "body": "Controlled testing showed that identical content with one fewer JPEG encode "
                "produces no ELA signal at all, while *different* content with the same "
                "compression history produces a stronger signal than any real splice. Detailed "
                "regions simply resist quantisation more than smooth ones. We measure that "
                "confound directly (rank correlation between block error and block edge energy) "
                "and discount the score when it is high — but ELA remains a supporting signal "
                "and never the sole basis for a verdict.",
    },
    {
        "title": "Re-encoding destroys evidence",
        "body": "A single aggressive JPEG re-save — what WhatsApp, Instagram and X all do — "
                "removes most of the fine-detail traces these methods read. Heavily compressed "
                "input is reported as low-reliability rather than as authentic. Absence of "
                "evidence is not evidence of absence.",
    },
    {
        "title": "Per-region quality recovery is impossible after a finer encode",
        "body": "It is not possible to ask a block 'what quality were you last saved at' and get "
                "a truthful answer once a coarser-quality encode has been applied over it: "
                "encoding projects onto the final quantisation lattice regardless of history. The "
                "JPEG-ghost analyser therefore does not attempt it. It measures DCT-coefficient "
                "*striation* instead, which does survive, and it reports 'no history detectable' "
                "rather than inventing one.",
    },
    {
        "title": "Learned detectors generalise worst",
        "body": "A CNN trained on a fixed set of generators degrades on generators released after "
                "its training data. This is an arms race with no stable end state. The published "
                "metrics describe performance on the evaluation split, not on images in the wild, "
                "and the CLIP probe is reported alongside precisely because its disagreement with "
                "the CNN is informative.",
    },
    {
        "title": "Missing metadata is normal",
        "body": "Social platforms, messaging apps and browsers strip EXIF as a matter of routine. "
                "Absence of camera metadata, absence of Content Credentials, and absence of a "
                "detected face all lower the *reliability* score. None of them counts as evidence "
                "of manipulation.",
    },
    {
        "title": "Synthetic evaluation numbers",
        "body": "Unless the metrics panel says otherwise, the published figures come from a "
                "synthetic fixture set whose forgery mechanism is known by construction. That set "
                "validates that each analyser responds to the artefact it claims to detect. It "
                "does NOT establish real-world accuracy, which requires CASIA v2, "
                "FaceForensics++, DFDC and the 140k Real and Fake Faces datasets — all gated "
                "multi-GB downloads. The training and calibration pipeline is complete; the "
                "weights and real-data metrics are not.",
    },
)
