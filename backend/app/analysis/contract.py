"""The analyser contract, ported verbatim from the Streamlit prototype.

Every technique in ``app/analysis/techniques/`` exposes::

    run(...) -> tuple[float, np.ndarray | None, dict]

where

* ``score``   - float in [0, 1], higher = more suspicious
* ``visual``  - uint8 **RGB** image for display, or ``None`` when the technique has
                nothing to draw
* ``details`` - dict of machine-readable findings (JSON-coerced later by ``adapt.jsonify``)

Two rules that are easy to get wrong:

1. **RGB, not BGR.** OpenCV is BGR and PIL is RGB. Techniques return RGB so the overlay
   sink performs exactly one ``cvtColor`` before ``cv2.imwrite``. Get this wrong and every
   face box is blue.
2. **Never raise.** ``worker.run_technique`` wraps every call, so an exception degrades one
   technique instead of failing the job — but the technique should still return a usable
   ``details`` dict explaining what it could not do.

This contract is adapted to the wire schema in exactly one place: ``analysis/adapt.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

__all__ = ["AnalyserFn", "TechniqueResult", "clip01"]

AnalyserFn = Callable[..., "TechniqueResult"]


class TechniqueResult(NamedTuple):
    """The ``(score, visual, details)`` triple, named for readability at call sites."""

    score: float
    visual: np.ndarray | None
    details: dict


def clip01(x: object) -> float:
    """Clamp to [0, 1] and coerce numpy scalars to a plain float.

    Returning ``np.float32`` here breaks Pydantic serialisation downstream, so every
    technique must funnel its score through this.

    Negative zero is normalised away: a product like ``-expm1(-0.0)`` is ``-0.0``, which compares
    equal to zero but serialises as ``-0.0`` and renders in the UI as "-0.00" on a clean image.
    """
    out = float(np.clip(float(x), 0.0, 1.0))
    return 0.0 if out == 0.0 else out
