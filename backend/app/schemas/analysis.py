"""The analysis result schema.

The spec fixes a flat shape — ``verdict, ai_score, edited_score, overall, reliability, reasons[],
techniques{}, metadata, c2pa, sha256, timings`` — and the UI needs more than that: headline copy,
evidence flags, reliability penalties, limitations. Both are present. The duplication is on purpose
and is *not* a modelling mistake: the flat fields are the published contract, stable across versions
and cheap to assert against in tests, while the nested blocks carry the detail the pages render. An
API that grows a new display need should add a nested block rather than break a documented field.

Every numeric field is ``float`` in 0..1 or ``None``. ``None`` means "not measured" and is the only
correct rendering of a check that could not run; a ``0.0`` here would tell the user their image passed
a test that never executed. That distinction is the product's central honesty claim, so it is stated
in the field descriptions and the frontend has to honour it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analysis.adapt import TechniqueView

__all__ = [
    "AnalysisResult",
    "C2paBlock",
    "EvidenceFlagView",
    "MetadataBlock",
    "ReliabilityBlock",
    "Timings",
    "VerdictBlock",
]

VerdictLabel = Literal["authentic", "ai_generated", "edited", "inconclusive"]
RiskBand = Literal["low", "moderate", "high"]

#: The C2PA states ``techniques/c2pa_verify.py`` can report. Kept here as data rather than as a
#: ``Literal`` so the set can grow at the analyser without breaking a client, with anything outside
#: it normalised to ``"unknown"`` at the boundary.
_CREDENTIAL_STATES = frozenset({
    "verified", "invalid", "absent", "unreadable", "unavailable", "malformed_json", "error",
    "unknown",
})


class EvidenceFlagView(BaseModel):
    """One rule that moved a number, stated so the UI can show *why* the score changed."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["deterministic", "statistical"]
    text: str
    target: str | None = Field(
        None, description="Which sub-score this floors: 'ai', 'edited' or 'overall'.")
    floor: float | None = Field(None, description="The value it was raised to.")
    forces: VerdictLabel | None = None

    @classmethod
    def from_flag(cls, flag: Any) -> EvidenceFlagView:
        """Adapt an ``analysis.verdict.EvidenceFlag``.

        ``raises`` is a ``(target, floor)`` tuple in the engine and two named fields here: a tuple
        makes a client unpack positionally, and a client that unpacks positionally will get it wrong
        the first time a third element is added.
        """
        raises = getattr(flag, "raises", None)
        target = floor = None
        if raises:
            target, floor = str(raises[0]), float(raises[1])
        return cls(
            id=flag.id, kind=flag.kind, text=flag.text,
            target=target, floor=floor, forces=getattr(flag, "forces", None),
        )


class VerdictBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: VerdictLabel
    headline: str
    detail: str
    reasons: list[str] = Field(default_factory=list)
    flags: list[EvidenceFlagView] = Field(default_factory=list)
    #: True when an evidence flag altered a fused score, so the UI can label the number honestly
    #: instead of presenting a flag-adjusted value as a pure model output.
    adjusted: bool = False
    risk_band: RiskBand


class ReliabilityPenalty(BaseModel):
    """One documented reason to trust the result less.

    Mirrors ``analysis.reliability.Penalty``. ``affects`` names which sub-score the missing evidence
    bears on — ``"ai"``, ``"edited"``, or ``None`` when it weakens everything — so the reliability
    panel can say *which part* of the answer is weak rather than just "low confidence".
    """

    model_config = ConfigDict(frozen=True)

    id: str
    reason: str
    cost: float = Field(description="Multiplicative factor applied, 0..1.")
    affects: str | None = None
    severity: Literal["moderate", "severe"] = "moderate"

    @field_validator("severity", mode="before")
    @classmethod
    def _default_severity(cls, v: object) -> str:
        # A penalty registry that gains a new severity label must not fail a job at the moment it is
        # assembled. "moderate" is the honest default for a cost we have not classified.
        return v if v in ("moderate", "severe") else "moderate"


class ReliabilityBlock(BaseModel):
    """How much the *checks* deserved to be believed — separate from how likely the image is tampered.

    A low score here with a low ``overall`` is not a clean image; it is an unreadable one. The whole
    point of keeping this next to ``overall`` rather than folding it in is that the UI must be able to
    say "inconclusive, because most checks could not run" out loud.
    """

    model_config = ConfigDict(frozen=True)

    #: A block that was never filled in says *zero* trust, not full trust: the other default would
    #: let an unpopulated panel read as "highly reliable". ``note`` is defaulted for the same reason
    #: — a bare 0.0 with no sentence next it is the one rendering a reader cannot interpret.
    score: float = Field(0.0, ge=0, le=1)
    reasons: list[ReliabilityPenalty] = Field(default_factory=list)
    coverage: dict[str, int] = Field(
        default_factory=dict, description="Measured checks per group (ai / edited).")
    note: str = "Reliability was not measured for this result."


class MetadataFinding(BaseModel):
    """One sentence the metadata analyser can defend, plus the weight it carried.

    ``text`` is written for a reader and ``severity`` is what the reader should do with it, so both
    travel. Flattening this to a string is what put a Python dict repr on the report page.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    text: str
    #: ``ok`` | ``info`` | ``weak`` | ``moderate`` | ``strong``.
    severity: str = "info"
    points: float = 0.0


class MetadataBlock(BaseModel):
    """The metadata technique's findings, lifted to the top level because the UI panel wants them.

    A pass-through of ``details`` rather than a re-modelled copy: the analyser is the authority on
    what it found, and mirroring its keys here would mean two places to update every time a field is
    added — and one of them would be wrong. ``raw`` is that pass-through and the named fields are
    conveniences over it.
    """

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    present: bool = False
    exif: dict[str, Any] = Field(default_factory=dict)
    software: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    findings: list[MetadataFinding] = Field(default_factory=list)
    #: ``chunk`` is where it was found (a PNG text chunk name, an XMP field) and ``keyword`` is the
    #: fingerprint that matched there.
    generator_hits: list[dict[str, Any]] = Field(default_factory=list)
    #: True when the EXIF table itself had PII redacted. Generator parameters are deliberately
    #: outside that scope — see ``core.security.scan_generator_fingerprints``.
    pii_redacted: bool = True
    deterministic_flag: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class C2paBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = "unknown"
    present: bool = False
    #: ``absent`` is the overwhelmingly common case and is not a red flag; social platforms strip
    #: manifests on re-encode, so "no credentials" mostly means "someone re-saved this".
    credential: str = "unknown"
    issuer: str | None = None
    claim_generator: str | None = None
    assertions: list[dict[str, Any]] = Field(default_factory=list)
    note: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("credential")
    @classmethod
    def _known_credential(cls, v: str) -> str:
        # The analyser's vocabulary is allowed to grow; the client's is not. An unrecognised status
        # becomes "unknown" rather than a validation error, because a new C2PA state in the field must
        # not turn a completed analysis into a 500.
        return v if v in _CREDENTIAL_STATES else "unknown"


class Timings(BaseModel):
    """Where the milliseconds went, because "under five seconds" is a claim this app makes."""

    model_config = ConfigDict(frozen=True)

    total_ms: int = 0
    decode_ms: int = Field(0, description="Upload validation and array construction, in the parent.")
    queue_ms: int = Field(0, description="Time waiting on the concurrency semaphore.")
    analysis_ms: int = Field(0, description="Wall time from first group submitted to last returned.")
    ipc_ms: int = Field(0, description="Pickling payload across the process boundary.")
    assembly_ms: int = Field(0, description="Fusion, reliability, verdict, serialisation.")
    per_technique_ms: dict[str, int] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """The complete result for one image."""

    model_config = ConfigDict(frozen=True)

    # --- the flat contract ---
    verdict: VerdictLabel
    ai_score: float = Field(ge=0, le=1, description="Likelihood the image is synthetic.")
    edited_score: float = Field(ge=0, le=1, description="Likelihood a real photo was manipulated.")
    overall: float = Field(ge=0, le=1, description="max(ai, edited): the number the gauge shows.")
    reliability: float = Field(ge=0, le=1, description="Confidence in the measurements themselves.")
    reasons: list[str] = Field(default_factory=list, description="Why this verdict.")
    techniques: dict[str, TechniqueView] = Field(default_factory=dict)
    metadata: MetadataBlock = Field(default_factory=MetadataBlock)
    c2pa: C2paBlock = Field(default_factory=C2paBlock)
    sha256: str = ""
    timings: Timings = Field(default_factory=Timings)

    # --- what the pages additionally render ---
    band: RiskBand = "low"
    headline: str = ""
    detail: str = ""
    flags: list[EvidenceFlagView] = Field(default_factory=list)
    reliability_detail: ReliabilityBlock = Field(default_factory=ReliabilityBlock)
    fusion: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    limitations: list[dict[str, str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    #: Served from the SHA-256 result cache rather than recomputed. The scores are identical; the
    #: timings are not, and pretending otherwise would make the latency numbers a lie.
    from_cache: bool = False
    #: File facts, never the filename or any client-supplied string that could carry content.
    image: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if v and (len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower())):
            raise ValueError("sha256 must be a 64-character hex digest")
        return v.lower()
