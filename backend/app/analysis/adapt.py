"""The one place the ``(score, visual, details)`` contract meets the wire schema.

Every technique returns numpy scalars, tuples, and — deliberately — ``NaN`` for a measurement
that could not be taken. None of that is valid JSON. This module is the single conversion point,
so no technique has to think about serialisation and no route has to think about numpy.

Two conversions matter more than they look:

**NaN -> null.** The fusion engine uses ``NaN`` to mean "this check did not run", because
``HistGradientBoostingClassifier`` handles NaN natively and a missing measurement must not be
mistaken for a zero. But JSON has no NaN: Python's encoder emits the bare token ``NaN``, which
``JSON.parse`` in the browser rejects outright, taking the whole result page with it. So NaN
stays NaN in-process and becomes ``null`` here. The two representations are both correct for
their side of the boundary; conflating them is the bug.

**Arrays are capped.** ``details`` is built by our own code, but a technique that accidentally
drops a full-resolution array in there would serialise to hundreds of megabytes and OOM the
worker on the way out. The caps below turn that from an outage into a truncated field.
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.analysis.explainers import Explanation, Weight, band
from app.analysis.registry import BY_NAME, OverlayKind
from app.analysis.types import OverlayRef, TaskOutcome, TaskStatus

__all__ = ["ExplanationView", "TechniqueView", "build_technique_view", "jsonify"]

#: Longest list/array forwarded. Per-region tables are the largest legitimate payload and top
#: out around 25 entries; this only ever bites on a bug.
MAX_ITEMS = 4096
#: Recursion guard. Real ``details`` nests three or four levels (per-face -> checks -> eye).
MAX_DEPTH = 12
#: Longest string forwarded, so a multi-megabyte XMP blob cannot ride along.
MAX_STR = 4000

_BINARY_PLACEHOLDER = "<{n} bytes of binary data, not forwarded>"


def jsonify(value: Any, *, _depth: int = 0) -> Any:
    """Recursively coerce a ``details`` value into JSON-safe Python primitives.

    Returns ``None`` for NaN/Inf, ``bytes`` for binary payloads (as a placeholder, never as
    data), and lists for every sequence type. Dict keys are stringified. Never raises.
    """
    if value is None or isinstance(value, (bool, str)):
        return _clip_str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # JSON has no NaN or Infinity. Emitting them produces a response the browser cannot parse.
        return value if math.isfinite(value) else None
    if _depth >= MAX_DEPTH:
        return None

    # numpy scalars before ndarray: np.float64 is not an ndarray but np.bool_ etc. need int/float
    # coercion, and PIL's IFDRational needs float().
    if isinstance(value, Enum):
        return jsonify(value.value, _depth=_depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonify(dataclasses.asdict(value), _depth=_depth + 1)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _BINARY_PLACEHOLDER.format(n=len(value))
    if isinstance(value, dict):
        return {str(k): jsonify(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonify(v, _depth=_depth + 1) for v in list(value)[:MAX_ITEMS]]

    # Everything below is a third-party scalar/array. Imported lazily and duck-typed so this
    # module stays importable without numpy present (it is not, but the wire layer should not
    # be the thing that hard-fails on a missing optional dep).
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return _clip_str(str(value))

    if isinstance(value, np.ndarray):
        if value.size > MAX_ITEMS:
            return _BINARY_PLACEHOLDER.format(n=value.size).replace(
                "binary data", "array elements")
        return jsonify(value.tolist(), _depth=_depth + 1)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.generic):
        return jsonify(value.item(), _depth=_depth + 1)

    # PIL IFDRational, decimal.Decimal, fractions.Fraction: anything with a lossless float form.
    for attr in ("__float__", "__int__"):
        if hasattr(value, attr):
            try:
                return jsonify(getattr(value, attr)(), _depth=_depth + 1)
            except (TypeError, ValueError, OverflowError):
                break
    return _clip_str(str(value))


def _clip_str(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value if len(value) <= MAX_STR else value[:MAX_STR] + f"... [truncated, {len(value)} chars]"


class ExplanationView(BaseModel):
    """The three sentences a person reads, plus how much to trust them."""

    model_config = ConfigDict(frozen=True)

    why: str = Field(description="Why this measurement exists at all.")
    found: str = Field(description="What this particular image showed.")
    weight: Weight = Field(description="Evidential weight of the finding.")
    weight_note: str = Field(description="Why it carries that weight, including its weakness.")


class TechniqueView(BaseModel):
    """One technique's result, ready for the API response."""

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    group: str = Field(description="'ai', 'edited' or 'context' — which sub-score it feeds.")
    status: TaskStatus
    score: float | None = Field(
        None,
        description=(
            "0..1, higher = more suspicious. ``null`` whenever the check has nothing to report — it "
            "errored, it did not run, or it ran and could not measure this file. The last of those is "
            "the one a client gets wrong: a numeric 0.0 means \"this test ran and the image passed\", "
            "and rendering a silent check as a green zero is the exact false reassurance this product "
            "refuses to make. ``measured`` is the same fact as a boolean."))
    band: str = Field(description=(
        "'low' (<0.35), 'moderate' (<0.60), 'high', or 'not_measured' when there is no score to band. "
        "Never colour a 'not_measured' card as green."))
    #: The technique executed without error. Distinct from ``measured``: a check can run
    #: successfully and still have nothing to say, e.g. face analysis on a photo with no face.
    ran: bool
    #: The technique produced a finding that carries evidential weight. False when it was
    #: inapplicable, when it ran but could not measure (no detectable quantisation periodicity,
    #: no face, a spectral tail destroyed by compression), or when it errored.
    #:
    #: **The UI must render ``measured == False`` as a greyed-out "nothing to measure" card, never
    #: as a green zero.** A score of None or 0.0 here means the check is silent, not that the
    #: image passed it.
    measured: bool
    applicable: bool | None = Field(
        None, description="Whether the technique declares itself applicable to this file at all.")
    overlay_url: str | None = None
    overlay_kind: OverlayKind = "none"
    explanation: ExplanationView
    details: dict[str, Any] = Field(default_factory=dict)
    ms: int = 0
    error: str | None = None


def build_technique_view(
    outcome: TaskOutcome,
    explanation: Explanation,
    *,
    overlay_url: str | None = None,
) -> TechniqueView:
    """Adapt a worker's ``TaskOutcome`` into the wire schema.

    ``overlay_url`` is resolved by the caller (it needs the signing key and the request's base
    URL); this function only knows whether an overlay exists and what kind it is.
    """
    spec = BY_NAME.get(outcome.name)
    details = jsonify(outcome.details)
    ran = outcome.status == "ok"
    measured = ran and explanation.weight != "unavailable"
    # A technique that reports `applicable: False` or `available: False` is telling us it could
    # not measure this file. Surface that explicitly so the UI can grey the card out instead of
    # showing a reassuring green zero.
    applicable = None
    for key in ("applicable", "available"):
        if key in details:
            applicable = bool(details[key])
            break
    # Null the number, not just the flag. A technique that ran and had nothing to measure still
    # produced a float internally — 0.0 from a CNN that never loaded, a near-zero FFT slope from a
    # spectrum the encoder flattened — and every downstream colour scale reads a low float as "clean".
    # The raw value stays visible in ``details``, where a user can see what the check actually saw.
    shown_score = None if not measured or outcome.score is None else float(outcome.score)
    return TechniqueView(
        name=outcome.name,
        label=spec.label if spec else outcome.name,
        group=spec.group if spec else "context",
        status=outcome.status,
        score=shown_score,
        band="not_measured" if shown_score is None else band(shown_score),
        ran=ran,
        measured=measured,
        applicable=applicable,
        overlay_url=overlay_url if outcome.overlay is not None else None,
        overlay_kind=spec.overlay if spec else "none",
        explanation=ExplanationView(
            why=explanation.why, found=explanation.found,
            weight=explanation.weight, weight_note=explanation.weight_note),
        details=details,
        ms=int(outcome.ms),
        error=outcome.error_msg,
    )


def overlay_ref_to_dict(ref: OverlayRef | None) -> dict[str, Any] | None:
    """Serialise an ``OverlayRef`` for the internal job record (not the public response)."""
    return None if ref is None else jsonify(dataclasses.asdict(ref))
