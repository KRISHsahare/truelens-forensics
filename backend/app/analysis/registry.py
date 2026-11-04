"""The technique catalogue: what exists, what it feeds, and how to call it.

**This module imports no technique module.** That is a hard constraint, not tidiness. The
analysis worker runs in a ``spawn`` child, which re-imports the worker module and everything it
transitively touches at call time. If the registry pulled in ``techniques/cnn.py``, every spawn
child would import torch before it had decided whether it needed to — seconds of startup and
hundreds of MB per worker, on machines where most jobs never touch the CNN at all.

So the registry holds *declarations*: strings, numbers, and binding descriptors. Technique
modules are imported by name inside ``analysis/worker.py``, in the child, only for the tasks
that child is actually given.

The second job of this module is to be the single source of truth for the three questions the
rest of the system keeps re-asking:

* which techniques contribute to ``ai_generated_score`` and which to ``edited_score``
  (``group``);
* which resolution track a technique must run on (``track``) — getting this wrong silently
  destroys the measurement, see ``core/pipeline.py``;
* how to render the returned visual (``overlay``), so the UI can label it honestly. A bare FFT
  spectrum presented as "heatmap over your photo" is a lie about what the user is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "AI_GROUP",
    "BY_NAME",
    "EDITED_GROUP",
    "TECHNIQUES",
    "FromArrays",
    "FromConfig",
    "Lit",
    "TechniqueSpec",
    "ai_techniques",
    "edited_techniques",
    "names",
    "spec",
]

#: Which verdict sub-score a technique feeds. Getting this right matters more than it looks:
#: metadata is the clearest trap. Its *score* is mostly "no EXIF present", and every social
#: platform strips EXIF, so letting that score vote would flag most honest photographs as
#: AI-generated. Metadata therefore feeds the AI sub-score only through its generator-parameter
#: and C2PA *flags* — facts about the file — never through its score. That rule lives in
#: ``fusion.py``, which is why the taxonomy here can stay about contribution rather than caution.
Group = Literal["ai", "edited", "context"]

#: Which resolution track the technique consumes. See the ``JobArrays`` docstring for why this is
#: a correctness requirement rather than an optimisation.
Track = Literal["native", "analysis", "container"]

#: How the returned ``visual`` should be presented.
#:
#: * ``heatmap``   - a per-pixel or per-block score composited over the photograph
#: * ``vector``    - boxes, landmarks or match lines drawn on the photograph
#: * ``spectrum``  - the bare FFT magnitude image; its axes ARE frequency, not position
#: * ``none``      - the technique never returns a visual
OverlayKind = Literal["heatmap", "vector", "spectrum", "none"]


# --------------------------------------------------------------------- binding
@dataclass(frozen=True, slots=True)
class FromArrays:
    """Bind a call argument from a ``JobArrays`` field of the same name."""

    field: str


@dataclass(frozen=True, slots=True)
class FromConfig:
    """Bind a call argument from the runtime config (model paths, feature switches).

    Resolved in the worker against a plain dict, so the registry itself needs no config import
    and cannot drag ``pydantic-settings`` into a spawn child either.
    """

    field: str


@dataclass(frozen=True, slots=True)
class Lit:
    """Bind a call argument to a constant."""

    value: Any


Arg = FromArrays | FromConfig | Lit


@dataclass(frozen=True, slots=True)
class TechniqueSpec:
    name: str
    label: str
    group: Group
    track: Track
    overlay: OverlayKind
    #: Dotless module name under ``app.analysis.techniques``. Differs from ``name`` for
    #: ``c2pa``, whose module is ``c2pa_verify`` because ``c2pa`` would shadow the library.
    module: str
    #: Positional arguments, in order.
    args: tuple[Arg, ...] = ()
    #: Keyword arguments.
    kwargs: MappingProxyType[str, Arg] = MappingProxyType({})
    #: One sentence for the technique grid on the landing page.
    summary: str = ""
    #: Display order in the analyzer's progress stepper and result cards.
    order: int = 0
    #: Techniques that need a heavyweight runtime (torch). Used to route them to the single
    #: torch-bearing worker instead of the general pool.
    heavy: bool = False


def _kw(**items: Arg) -> MappingProxyType[str, Arg]:
    return MappingProxyType(items)


#: Every technique, in display order. Order is deliberate: cheap and universal checks first, so
#: the progress stepper shows movement immediately, and the heavyweight learned detector last.
_SPECS: tuple[TechniqueSpec, ...] = (
    TechniqueSpec(
        name="metadata", label="Metadata & Provenance", group="ai", track="container",
        overlay="none", module="metadata", order=10,
        args=(FromArrays("orig_bytes"), FromArrays("mime")),
        kwargs=_kw(redact_pii=FromConfig("redact_pii")),
        summary="Reads EXIF, XMP and PNG text chunks for camera provenance and for the "
                "parameter blocks AI tools leave behind.",
    ),
    TechniqueSpec(
        name="c2pa", label="Content Credentials (C2PA)", group="ai", track="container",
        overlay="none", module="c2pa_verify", order=20,
        args=(FromArrays("orig_bytes"), FromArrays("mime")),
        summary="Verifies a cryptographically signed C2PA manifest, if the file carries one.",
    ),
    TechniqueSpec(
        name="ela", label="Error Level Analysis", group="edited", track="native",
        overlay="heatmap", module="ela", order=30,
        args=(FromArrays("native_rgb_u8"), FromArrays("mime"), FromArrays("est_jpeg_quality")),
        summary="Re-saves at the file's own JPEG quality and looks for regions whose "
                "quantisation error does not match the rest of the frame.",
    ),
    TechniqueSpec(
        name="jpeg_ghost", label="JPEG Ghost", group="edited", track="native",
        overlay="heatmap", module="jpeg_ghost", order=40,
        args=(FromArrays("native_rgb_u8"), FromArrays("mime"), FromArrays("native_box")),
        summary="Reads DCT-coefficient striation to find regions that carry a second, "
                "different compression history.",
    ),
    TechniqueSpec(
        name="block_grid", label="Block-Grid Alignment", group="edited", track="native",
        overlay="heatmap", module="block_grid", order=50,
        args=(FromArrays("native_gray_u8"), FromArrays("mime")),
        summary="Checks whether every part of the image shares one 8x8 JPEG block phase, or "
                "whether a pasted region landed on a different grid.",
    ),
    TechniqueSpec(
        name="noise", label="Noise Inconsistency", group="edited", track="analysis",
        overlay="heatmap", module="noise", order=60,
        args=(FromArrays("gray_u8"), FromArrays("analysis_scale"),
              FromArrays("est_jpeg_quality")),
        summary="Estimates the sensor noise floor per block; a spliced region carries a "
                "different camera's noise.",
    ),
    TechniqueSpec(
        name="copy_move", label="Copy-Move Cloning", group="edited", track="analysis",
        overlay="vector", module="copy_move", order=70,
        args=(FromArrays("rgb_u8"), FromArrays("gray_u8"), FromArrays("analysis_scale")),
        kwargs=_kw(max_features=FromConfig("copy_move_features")),
        summary="Matches SIFT keypoints against each other and votes on displacement, to find "
                "regions cloned from elsewhere in the same photo.",
    ),
    TechniqueSpec(
        name="face", label="Face Consistency", group="edited", track="analysis",
        overlay="vector", module="face", order=80,
        args=(FromArrays("rgb_u8"), FromArrays("gray_u8")),
        kwargs=_kw(model_path=FromConfig("face_model_path")),
        summary="Detects faces and landmarks, then tests blending boundaries, lighting "
                "symmetry and specular-reflection consistency.",
    ),
    TechniqueSpec(
        name="frequency", label="Frequency Spectrum", group="ai", track="native",
        overlay="spectrum", module="frequency", order=90,
        args=(FromArrays("native_gray_u8"),),
        summary="Fits the amplitude-spectrum decay and scans the axes for the periodic "
                "checkerboard upsampling leaves behind.",
    ),
    TechniqueSpec(
        name="cnn", label="Learned Detector", group="ai", track="analysis",
        overlay="heatmap", module="cnn", order=100, heavy=True,
        args=(FromArrays("rgb_u8"),),
        kwargs=_kw(tta=FromConfig("cnn_tta"), want_gradcam=Lit(True),
                   gradcam_threshold=FromConfig("gradcam_min_ai_score"),
                   weights_path=FromConfig("cnn_weights_path"),
                   clip_probe_path=FromConfig("clip_probe_path"),
                   use_clip=FromConfig("clip_probe_enabled")),
        summary="EfficientNet-B0 fine-tuned on labelled real/generated data, optionally "
                "corroborated by a frozen-CLIP linear probe, with Grad-CAM localisation.",
    ),
)

TECHNIQUES: tuple[TechniqueSpec, ...] = tuple(sorted(_SPECS, key=lambda s: s.order))
BY_NAME: MappingProxyType[str, TechniqueSpec] = MappingProxyType({s.name: s for s in TECHNIQUES})

AI_GROUP: tuple[str, ...] = tuple(s.name for s in TECHNIQUES if s.group == "ai")
EDITED_GROUP: tuple[str, ...] = tuple(s.name for s in TECHNIQUES if s.group == "edited")


def names() -> tuple[str, ...]:
    return tuple(s.name for s in TECHNIQUES)


def spec(name: str) -> TechniqueSpec:
    return BY_NAME[name]


def ai_techniques() -> tuple[TechniqueSpec, ...]:
    return tuple(s for s in TECHNIQUES if s.group == "ai")


def edited_techniques() -> tuple[TechniqueSpec, ...]:
    return tuple(s for s in TECHNIQUES if s.group == "edited")
