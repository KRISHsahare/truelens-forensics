"""Fusion: ten per-technique scores -> two sub-scores and an overall likelihood.

The prototype fused with a hand-tuned ``WEIGHTS`` dict, then multiplied the result by 1.6 to make
the gauge look decisive, then overrode it with ``if`` statements. All three of those are problems:
the weights were guesses, the stretch was theatre, and the overrides silently contradicted the
number the user had just been shown.

This module replaces them with two paths that produce the *same schema*:

**Learned** — two classifiers over the 48-column feature vector (``features.py``), one for
"generated", one for "edited", each probability-calibrated. Used when ``fusion.joblib`` exists.

**Expert prior** — the documented noisy-OR below. Used when it does not, which is every fresh
checkout because the training datasets are gated multi-GB downloads nobody has agreed to here.

The expert prior is not a placeholder pretending to be a model. It is an explicit, auditable
statement of how much each analyser is trusted, and it is *labelled as such* in the result
(``source="expert"``) so the UI can say so too. Pretending otherwise would be the dishonest move.

Why noisy-OR rather than a weighted mean. A mean assumes the techniques are measuring the same
underlying quantity and averaging out noise. They are not: each is an independent test for a
different mechanism, and one of them screaming is not the same as six of them murmuring. A splice
that lights up JPEG-ghost, block-grid and ELA together should be more suspicious than any single
one, and a weighted mean actively suppresses that by dividing by six. Noisy-OR —
``1 - prod(1 - w_i * s_i)`` — accumulates agreement and, because a technique that could not run
contributes no term at all, degrades gracefully instead of averaging a silent check in as a zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.analysis.features import FEATURES, extract
from app.analysis.registry import TECHNIQUES

__all__ = ["EXPERT_WEIGHTS", "FusionResult", "fuse", "load_learned", "together"]

#: Per-technique trust ceiling for the expert prior: the probability the *whole* verdict moves to
#: when this analyser reports its maximum score and nothing else does.
#:
#: Calibrated from the controlled experiments recorded in the analyser docstrings, not from
#: intuition. The ordering encodes three measured facts:
#:
#: * JPEG-ghost and copy-move sit highest because their signals cannot be produced by image
#:   content — a quantisation periodicity and a displacement-consistent keypoint cluster are
#:   artefacts of a process, not of a scene.
#: * ELA sits lowest of the editing checks despite being the most famous, because testing showed
#:   it cannot separate compression history from local detail; a different-content/same-history
#:   control produced a stronger signal than any real splice.
#: * The CNN sits highest overall but only runs when weights exist, and generalises worst to
#:   generators it has not met.
EXPERT_WEIGHTS: dict[str, float] = {
    "jpeg_ghost": 0.75,
    "copy_move": 0.75,
    "block_grid": 0.60,
    "noise": 0.60,
    "face": 0.50,
    "ela": 0.45,
    "cnn": 0.85,
    "frequency": 0.70,
    # Metadata and C2PA contribute through flags, not scores. See ``_contribution``.
    "metadata": 0.00,
    "c2pa": 0.00,
}

#: A file that *states* it was generated is not a probabilistic finding. This is the probability
#: the expert prior assigns when metadata carries a generator-parameter block or an AI source type.
DETERMINISTIC_AI_P = 0.90

#: Same for a C2PA manifest whose signature fails to verify.
BAD_CREDENTIAL_P = 0.55

#: ...and much lower when one verifies cleanly.
GOOD_CREDENTIAL_P = 0.05


@dataclass(slots=True)
class FusionResult:
    ai_score: float
    edited_score: float
    overall: float
    #: "learned" when trained classifiers produced these, "expert" for the documented prior.
    source: str
    #: Whether the learned model's outputs were probability-calibrated. An uncalibrated gradient
    #: boosting score is a *ranking*, not a probability, and must not be shown as a percentage.
    calibrated: bool = False
    #: Technique -> the probability term it contributed, for the "why" panel and the PDF.
    contributions: dict[str, float] = field(default_factory=dict)
    #: Which group each score was thin on, so the verdict layer can widen its uncertainty.
    support: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ expert prior
def _contribution(name: str, details: dict[str, Any], score: float) -> tuple[float, str]:
    """Return ``(probability_term, why)`` for one technique under the expert prior.

    ``0.0`` with a reason of ``""`` means "contributes nothing" — including the case where the
    check could not run, which must never be mistaken for a clean result.
    """
    if details.get("applicable") is False or details.get("available") is False:
        return 0.0, ""
    if details.get("error_msg") or details.get("status") in ("error", "timeout"):
        return 0.0, ""

    if name == "metadata":
        # Only a fact about the file moves this. "No EXIF" is what Instagram produces.
        if details.get("deterministic_flag"):
            return DETERMINISTIC_AI_P, "generator metadata embedded in the file"
        return 0.0, ""

    if name == "c2pa":
        status = details.get("status")
        if status == "invalid":
            return BAD_CREDENTIAL_P, "C2PA signature failed to verify"
        if status == "verified":
            return GOOD_CREDENTIAL_P, "C2PA manifest verified"
        return 0.0, ""

    # Every technique that did run but reported "nothing to measure" is excluded by the caller
    # having set score to NaN, so a real 0.0 here genuinely means "looked, found nothing".
    w = EXPERT_WEIGHTS.get(name, 0.0)
    return (w * score, f"{name} at {score:.2f}" if score > 0 else "")


def _noisy_or(terms: list[float]) -> float:
    """``1 - prod(1 - p_i)``: the probability that at least one independent test fired."""
    if not terms:
        return 0.0
    keep = 1.0
    for p in terms:
        keep *= (1.0 - max(0.0, min(1.0, p)))
    return 1.0 - keep


def expert_fuse(outcomes: dict[str, Any], features: dict[str, float]) -> FusionResult:
    """Combine technique scores with the documented prior. Used when no trained model is present."""
    terms: dict[str, list[float]] = {"ai": [], "edited": []}
    contributions: dict[str, float] = {}
    notes: list[str] = []
    support = {"ai": 0, "edited": 0}

    for spec in TECHNIQUES:
        oc = outcomes.get(spec.name)
        if oc is None or oc.status not in ("ok",):
            continue
        raw = features.get(f"{spec.name}.score", math.nan)
        measured = features.get(f"{spec.name}.measured", 0.0) >= 0.5
        score = math.nan if raw is None else float(raw)
        term, why = _contribution(spec.name, oc.details or {},
                                  0.0 if math.isnan(score) else score)
        contributions[spec.name] = term
        if spec.group in terms:
            # Metadata/C2PA contribute their flag even though their *score* is unmeasured context,
            # so coverage is counted from the flag path too.
            if term > 0 or measured:
                terms[spec.group].append(term)
                support[spec.group] += 1
            if term > 0:
                notes.append(why)

    ai = _noisy_or(terms["ai"])
    edited = _noisy_or(terms["edited"])
    # Noisy-OR across the two hypotheses: "generated" and "edited" are different failure modes,
    # but either one makes the file "not an untouched photograph", which is what overall measures.
    overall = _noisy_or([ai, edited])
    return FusionResult(ai_score=ai, edited_score=edited, overall=overall,
                        source="expert", calibrated=False,
                        contributions=contributions, support=support,
                        notes=notes or ["no technique contributed positive evidence"])


# ------------------------------------------------------------------ learned path
class _Learned:
    """Two calibrated binary classifiers, loaded once per process.

    Cached at module scope because a spawn child that scores jobs must not re-read the joblib
    on every call — the same mistake the prototype made calling ``torch.load`` per request.
    """

    _cache: dict[str, _Learned | None] = {}

    def __init__(self, ai: Any, edited: Any, feature_names: Any, calibrated: bool) -> None:
        self.ai, self.edited = ai, edited
        self.feature_names = list(feature_names or [])
        self.calibrated = bool(calibrated)

    @classmethod
    def load(cls, path: str) -> _Learned | None:
        if path in cls._cache:
            return cls._cache[path]
        model = None
        try:
            import joblib

            blob = joblib.load(path)
            names = blob.get("feature_names") if isinstance(blob, dict) else None
            # A model whose columns no longer match the code is worse than no model: it would
            # return confident numbers computed against a different feature definition. Refusing
            # to load and falling back to the labelled expert prior is the honest failure.
            if names is None or list(names) != list(FEATURES):
                cls._cache[path] = None
                return None
            model = cls(blob["ai"], blob["edited"], names, blob.get("calibrated", False))
        except Exception:
            model = None
        cls._cache[path] = model
        return model


def load_learned(path: str | None) -> _Learned | None:
    return _Learned.load(path) if path else None


def learned_fuse(model: _Learned, features: dict[str, float]) -> FusionResult:
    """Score the feature vector with the trained classifiers.

    The linear fallback path inside ``calibrate_fusion.py`` cannot consume NaN, so ``impute=True``
    fills it with 0.0 and relies on the ``.measured`` columns to carry the distinction — which is
    why those columns are structural rather than decorative.
    """
    from app.analysis.features import vectorize

    row = vectorize(features, impute=True).reshape(1, -1)
    ai = float(model.ai.predict_proba(row)[0][1])
    edited = float(model.edited.predict_proba(row)[0][1])
    overall = _noisy_or([ai, edited])
    return FusionResult(ai_score=ai, edited_score=edited, overall=overall,
                        source="learned", calibrated=model.calibrated)


def fuse(
    outcomes: dict[str, Any],
    arrays: Any,
    explanations: dict[str, Any] | None = None,
    *,
    model_path: str | None = None,
) -> tuple[FusionResult, dict[str, float]]:
    """Fuse one job's technique outcomes. Returns the result and the feature vector used."""
    features = extract(outcomes, arrays, explanations)
    model = load_learned(model_path)
    if model is not None:
        return learned_fuse(model, features), features
    return expert_fuse(outcomes, features), features


def together(*probs: float) -> float:
    """Expose noisy-OR for the verdict layer, which composes scores the same way."""
    return _noisy_or(list(probs))
