"""Reliability: how much of this verdict can be trusted, independent of what it says.

The spec requires reliability to be a separate number from likelihood, and the reason is worth
stating plainly because it is the difference between a forensic tool and a coin flip. A score of
0.05 on Error Level Analysis means one of two completely different things:

* the check ran on a well-preserved JPEG and found nothing, or
* the check could not run at all — the file is a PNG, or 90 pixels wide, or so compressed that
  there is no signal left to read.

A likelihood score cannot tell those apart, because both produce zero. Collapsing them is exactly
how a tool ends up confidently declaring every heavily-compressed meme "authentic": compression
silences most detectors, silence reads as clean, and clean reads as trustworthy. Reliability is
the number that says "we could not see much here", whatever the verdict turned out to be.

Penalties compose **multiplicatively**, not additively. Each penalty claims to remove a fraction
of the remaining trust, so six modest penalties compound to a genuinely low score while two
cannot cancel each other out, and the total can never go below zero the way a sum of subtractions
invites a ``max(0, ...)`` clamp that hides how close it ran to the floor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.analysis.explainers import Explanation
from app.analysis.registry import TECHNIQUES
from app.analysis.types import JobArrays, TaskOutcome

__all__ = ["Penalty", "Reliability", "ReliabilityContext", "compute_reliability"]

#: Trust never falls below this. A verdict computed on almost nothing is still a verdict, and
#: showing 0.00 would imply the analysis produced literally no information — which it did not.
FLOOR = 0.05

#: Below this many *usable* measurements in a group, that group's score is barely supported.
STRONG_COVERAGE = 3
#: Below this many usable measurements across all groups, the analysis measured almost nothing at
#: all -- a different fact from "one group is thin", and the one that has to be priced separately.
MIN_MEASURED_CHECKS = 3
MIN_SIDE_FLOOR = 256
#: Bits per pixel under which messenger/social recompression has clearly taken detail away.
LOW_BPP = 0.35
#: JPEG quality under which quantisation itself dominates the per-block statistics.
HEAVY_QUALITY = 45
#: Analysis-track scale below which the native crop was doing a lot of work on a big file.
RESIZED_BELOW = 0.6


@dataclass(frozen=True, slots=True)
class Penalty:
    """One documented reason to trust the result less."""

    id: str
    #: Human-readable, shown in the reliability panel. Must say what is missing, not just that
    #: something is low.
    reason: str
    cost: float
    #: Which verdict the missing evidence bears on, so the UI can explain *which part* of the
    #: result is weak. ``None`` means it weakens everything.
    affects: str | None = None
    #: 'severe' penalties are those where the analysis genuinely could not look.
    severity: str = "moderate"


@dataclass(slots=True)
class ReliabilityContext:
    """Everything the penalty registry can consult."""

    arrays: JobArrays
    outcomes: dict[str, TaskOutcome] = field(default_factory=dict)
    explanations: dict[str, Explanation] = field(default_factory=dict)
    fusion_available: bool = False
    #: Group -> how many of that group's techniques produced a usable measurement.
    coverage: dict[str, int] = field(default_factory=dict)

    def details(self, name: str) -> dict[str, Any]:
        oc = self.outcomes.get(name)
        return (oc.details or {}) if oc else {}

    def weight(self, name: str) -> str:
        e = self.explanations.get(name)
        return e.weight if e else "unavailable"

    @property
    def errored(self) -> list[str]:
        return [n for n, o in self.outcomes.items() if o.status in ("error", "timeout")]


# ------------------------------------------------------------------ the registry
# Each predicate is written against a condition we can *name* for the user. A penalty that fires
# for a reason nobody can read out of a details dict is not a penalty, it is a bug.
def _small_side(ctx: ReliabilityContext) -> bool:
    return ctx.arrays.min_side < MIN_SIDE_FLOOR


def _heavy_bpp(ctx: ReliabilityContext) -> bool:
    return ctx.arrays.bpp < LOW_BPP


def _heavy_quality(ctx: ReliabilityContext) -> bool:
    q = ctx.arrays.est_jpeg_quality
    return q is not None and q < HEAVY_QUALITY


def _not_jpeg(ctx: ReliabilityContext) -> bool:
    return ctx.arrays.mime != "image/jpeg"


def _no_exif(ctx: ReliabilityContext) -> bool:
    return not ctx.details("metadata").get("has_exif", False)


def _no_face(ctx: ReliabilityContext) -> bool:
    d = ctx.details("face")
    return bool(d.get("available", False)) and not d.get("faces")


def _screenshot(ctx: ReliabilityContext) -> bool:
    return ctx.arrays.round_dimensions


def _resized(ctx: ReliabilityContext) -> bool:
    return ctx.arrays.analysis_scale < RESIZED_BELOW


def _spectra_destroyed(ctx: ReliabilityContext) -> bool:
    return bool(ctx.details("frequency").get("high_frequency_depleted"))


def _no_sensor_noise(ctx: ReliabilityContext) -> bool:
    return ctx.details("noise").get("noise_floor_usable") is False


def _cnn_missing(ctx: ReliabilityContext) -> bool:
    return ctx.weight("cnn") == "unavailable"


def _any_error(ctx: ReliabilityContext) -> bool:
    return bool(ctx.errored)


def _thin_edited(ctx: ReliabilityContext) -> bool:
    return ctx.coverage.get("edited", 0) < STRONG_COVERAGE


def _thin_ai(ctx: ReliabilityContext) -> bool:
    return ctx.coverage.get("ai", 0) < STRONG_COVERAGE


def _mostly_missing(ctx: ReliabilityContext) -> bool:
    """Fewer than `MIN_MEASURED_CHECKS` of the ten produced a usable measurement, in total.

    The two thin-coverage rules below are per group, and each costs 0.30, so a job where *nothing*
    ran still lands at 0.9 x 0.7 x 0.7 = 0.31 reliability — just above the line where the verdict
    layer stops calling it inconclusive. That is how a service whose worker pool never started
    answered "authentic" for a spliced photograph: the arithmetic was defensible and the conclusion
    was not. Total coverage is the fact that distinguishes "few checks fired" from "no check fired".
    """
    return _measured_count(ctx) < MIN_MEASURED_CHECKS


def _measured_count(ctx: ReliabilityContext) -> int:
    return sum(1 for spec in TECHNIQUES
               if spec.group != "context" and ctx.weight(spec.name) != "unavailable")


#: Declared in the order they should be shown to a reviewer: input-quality problems first, then
#: missing checks, then thin coverage.
_PENALTIES: tuple[tuple[Callable[[ReliabilityContext], bool], Penalty], ...] = (
    (_small_side, Penalty(
        "small", "Image is under 256px on its shortest side, so block-level statistics rest on "
        "very few samples.", 0.45, severity="severe")),
    (_heavy_bpp, Penalty(
        "low_bpp", "Very low bits-per-pixel — the file has been recompressed hard, which removes "
        "the fine-detail traces most checks read.", 0.35, severity="severe")),
    (_heavy_quality, Penalty(
        "heavy_jpeg", "JPEG quality estimated below 45. At that level quantisation is applied per "
        "8x8 block and dominates the noise and compression measurements.", 0.30,
        "edited", "severe")),
    (_spectra_destroyed, Penalty(
        "spectral_depletion", "The high-frequency tail is depleted, so the frequency analyser "
        "could not assess generation artefacts. This is not evidence either way about the image.",
        0.25, "ai", "severe")),
    (_no_sensor_noise, Penalty(
        "no_noise_floor", "Too little residual sensor noise to compare regions, so noise "
        "inconsistency could not contribute.", 0.25, "edited")),
    (_not_jpeg, Penalty(
        "not_jpeg", "The file is not a JPEG, so Error Level Analysis, JPEG-ghost and block-grid "
        "alignment all had nothing to read. Three of six manipulation checks are silent.",
        0.40, "edited", "severe")),
    (_resized, Penalty(
        "resized", "The image had to be downscaled for analysis, so it is either very large or "
        "was already resized before upload.", 0.15)),
    (_screenshot, Penalty(
        "screenshot_shape", "Both dimensions are round screen resolutions, which suggests a "
        "screenshot or an export rather than a camera file. Sensor-based checks assume a camera.",
        0.20)),
    (_no_exif, Penalty(
        "no_exif", "No EXIF present. Every messaging app and social platform strips it, so this "
        "is a note about missing context, not a suspicion about the image.", 0.10)),
    (_cnn_missing, Penalty(
        "cnn_missing", "The learned detector was unavailable, so no model-based evidence was "
        "available to corroborate the spectral analyser.", 0.30, "ai", "severe")),
    (_no_face, Penalty(
        "no_face", "No face was detected, so the face-consistency checks were skipped. Expected "
        "for most photographs; it only matters if a face was central to the question.", 0.12,
        "edited")),
    (_mostly_missing, Penalty(
        "almost_nothing_ran", "Fewer than three of the ten checks produced a usable measurement, "
        "so this analysis has almost nothing to report. Either the file is hostile to every method "
        "or the engine did not finish its work -- see the per-technique status below.",
        0.55, severity="severe")),
    (_thin_edited, Penalty(
        "thin_edited", "Fewer than three manipulation checks produced a usable measurement, so "
        "the edited score rests on thin evidence.", 0.30, "edited", "severe")),
    (_thin_ai, Penalty(
        "thin_ai", "Fewer than three generation checks produced a usable measurement, so the "
        "AI-generated score rests on thin evidence.", 0.30, "ai", "severe")),
    (_any_error, Penalty(
        "technique_error", "", 0.20, severity="moderate")),
)


@dataclass(frozen=True, slots=True)
class Reliability:
    score: float
    reasons: tuple[dict[str, Any], ...]
    coverage: dict[str, int]

    @property
    def band(self) -> str:
        s = self.score
        if s >= 0.75:
            return "high"
        if s >= 0.50:
            return "moderate"
        if s >= 0.30:
            return "low"
        return "very_low"


def compute_reliability(ctx: ReliabilityContext) -> Reliability:
    """Score how much of the verdict is supported by measurements that actually happened."""
    if not ctx.coverage:
        groups: dict[str, int] = {}
        for spec in TECHNIQUES:
            if spec.group == "context":
                continue
            groups[spec.group] = groups.get(spec.group, 0) + (
                1 if ctx.weight(spec.name) != "unavailable" else 0)
        ctx.coverage = groups

    score = 1.0
    reasons: list[dict[str, Any]] = []
    for predicate, penalty in _PENALTIES:
        try:
            fires = bool(predicate(ctx))
        except Exception:  # a malformed details dict must not fail the whole job
            fires = False
        if not fires:
            continue
        reason = penalty.reason
        if penalty.id == "technique_error":
            names = ", ".join(ctx.errored)
            reason = f"One or more checks failed to complete ({names}), so their evidence is missing."
        score *= (1.0 - penalty.cost)
        reasons.append({"id": penalty.id, "reason": reason, "cost": penalty.cost,
                        "affects": penalty.affects, "severity": penalty.severity})

    return Reliability(score=max(FLOOR, min(1.0, score)),
                       reasons=tuple(reasons), coverage=dict(ctx.coverage))
