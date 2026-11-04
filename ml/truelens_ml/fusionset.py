"""The labelled corpus the fusion classifiers train on, and the features pulled from it.

``app/analysis/features.py`` defines the columns; this module fills them. The pipeline is the one
the server runs — ``build_arrays``, then every registered analyser, then ``extract`` — because a
fusion model trained on features computed any other way is calibrated against a vector the
deployed code never produces. ``backend/scripts/smoke_fusion.py`` uses the same in-process pattern
for the same reason.

**Two labels, not one.** The CNN answers "generated or captured". Fusion answers two separate
questions — P(generated) and P(manipulated) — and an image can be any of the four combinations, so
a sample carries ``(ai, edited)`` instead of an index into four classes:

    authentic       (0, 0)
    ai_generated    (1, 0)
    edited          (0, 1)
    ai_and_edited   (1, 1)

Each classifier trains over the whole corpus on its own column, so a class does not need a row in
the other label: ``edited`` images are ``ai=0`` data for the AI head and ``ai_generated`` images
are ``edited=0`` data for the manipulation head. ``ai_and_edited`` is the case a single-label
corpus cannot express and a deployed product definitely meets — a generated picture with a face
swapped in, or a photograph with a region replaced by a generated one.

**Splitting is by group.** ``corpus.py`` explains why for the CNN; it applies harder here. A
fusion classifier sees 48 columns and one row per image, so two near-duplicate frames give it two
chances to memorise one scene and a held-out score that measures recall of the training rows. A
group is the directory one level above the file, and a whole group lands on one side.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from truelens_ml.corpus import _SKIP_DIRS, _group_for, _walk

__all__ = [
    "CLASSES",
    "FusionSample",
    "build_generated_corpus",
    "discover",
    "extract_features",
    "labels",
    "populated_columns",
    "split_by_group",
    "tally",
]

#: Directory name -> ``(ai, edited)``. The vocabulary is closed on purpose: an unrecognised
#: directory is an error rather than a silently ignored class, because a corpus that quietly trains
#: on half its own folders produces a model whose reported numbers describe a different dataset.
CLASSES: dict[str, tuple[int, int]] = {
    "authentic": (0, 0),
    "ai_generated": (1, 0),
    "edited": (0, 1),
    "ai_and_edited": (1, 1),
}

#: The manipulation constructions ``build_generated_corpus`` cycles through, so one class covers
#: three mechanisms rather than teaching the classifier that "edited" means "spliced".
_EDITED_MECHANISMS = ("spliced", "copy_move", "double_compressed")

#: Two re-encodes of the same scene per group. Re-sharing at another quality is the variation real
#: corpora contain; geometric jitter is not, and an invented seam would be a cue the model could
#: learn instead of the forgery.
_VARIANT_QUALITIES = (88, 74)


@dataclass(frozen=True, slots=True)
class FusionSample:
    path: str
    cls: str
    group: str
    ai: int
    edited: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.cls, self.group)


# ------------------------------------------------------------------- discovery
def discover(root: str | os.PathLike[str], *, limit_per_class: int | None = None) -> list[FusionSample]:
    """Read ``<root>/<class>/[...]/<image>`` into labelled samples. Raises when the layout is wrong.

    A missing class directory is fatal, not a warning. Training a manipulation head on a corpus
    with no ``edited/`` folder yields a classifier that predicts one class, a calibration pass that
    cannot fit a monotone map to a constant, and — if it wrote anything at all — a blob that scores
    uploads forever.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(
            f"corpus directory not found: {base}. See ml/README.md for the layout, or build a "
            "throwaway one with ml/scripts/calibrate_fusion.py --generated.")

    present = {p.name for p in base.iterdir() if p.is_dir() and p.name not in _SKIP_DIRS}
    unknown = sorted(present - set(CLASSES))
    if unknown:
        raise ValueError(
            f"unrecognised directories under {base}: {unknown}. Class roots are "
            f"{sorted(CLASSES)}; anything else is a mis-typed label, not extra data.")

    samples: list[FusionSample] = []
    for cls in CLASSES:
        class_root = base / cls
        if not class_root.is_dir():
            continue
        files = sorted(_walk(class_root), key=str)
        if limit_per_class:
            files = files[:limit_per_class]
        ai, edited = CLASSES[cls]
        samples.extend(FusionSample(str(p), cls, _group_for(p, class_root), ai, edited) for p in files)

    per_class = {c: sum(1 for s in samples if s.cls == c) for c in CLASSES}
    empty = sorted(c for c, n in per_class.items() if n == 0)
    if empty:
        raise ValueError(
            f"corpus {base} has no images for {empty}. Fusion needs both hypotheses represented; "
            "the labelled expert prior stays in charge until it does.")
    return samples


# --------------------------------------------------------------------- split
def split_by_group(
    samples: Sequence[FusionSample],
    *,
    frac_train: float = 0.7,
    seed: int = 7,
) -> dict[str, list[FusionSample]]:
    """Group-atomic, class-stratified train/calibration split.

    The calibration half is not a validation set for tuning. It is what
    ``CalibratedClassifierCV(cv="prefit")`` is given so the *reported* probabilities are ones the
    classifier did not fit itself; calibrating on the training rows fits a monotone map to
    memorisation and prints an expected-calibration error that describes neither.
    """
    if not 0.0 < frac_train < 1.0:
        raise ValueError(f"frac_train must be in (0,1), got {frac_train}")
    out: dict[str, list[FusionSample]] = {"train": [], "calib": []}
    for cls in CLASSES:
        pool = [s for s in samples if s.cls == cls]
        if not pool:
            continue
        groups = _group_order(pool, seed)
        if len(groups) < 2:
            raise ValueError(
                f"class {cls} has {len(groups)} group(s) across {len(pool)} image(s). Every image "
                "of a class inside one group cannot be held back for calibration — add a "
                "sibling directory per scene, or pass more images.")
        n_train = max(1, min(len(groups) - 1, round(frac_train * len(groups))))
        for i, (_, members) in enumerate(groups):
            out["train" if i < n_train else "calib"].extend(members)
    if not out["calib"] or not out["train"]:
        raise ValueError(f"split produced an empty side: { {k: len(v) for k, v in out.items()} }")
    return out


def _group_order(pool: Sequence[FusionSample], seed: int) -> list[tuple[tuple[str, str], list[FusionSample]]]:
    """Groups in a sha1-keyed order — the same split for the same corpus and seed, forever."""
    by_group: dict[tuple[str, str], list[FusionSample]] = {}
    for s in pool:
        by_group.setdefault(s.key, []).append(s)
    return sorted(
        by_group.items(),
        key=lambda kv: hashlib.sha1(f"{seed}:{kv[0][0]}:{kv[0][1]}".encode("utf-8", "surrogateescape")).hexdigest(),
    )


def labels(samples: Iterable[FusionSample]) -> tuple[list[int], list[int]]:
    pool = list(samples)
    return [s.ai for s in pool], [s.edited for s in pool]


def tally(samples: Iterable[FusionSample]) -> dict[str, Any]:
    pool = list(samples)
    return {
        "images": len(pool),
        "groups": len({s.key for s in pool}),
        "per_class": {c: sum(1 for s in pool if s.cls == c) for c in CLASSES},
        "ai_positive": sum(s.ai for s in pool),
        "edited_positive": sum(s.edited for s in pool),
    }


# ----------------------------------------------------------------- extraction
def extract_features(
    samples: Sequence[FusionSample],
    *,
    config_overrides: dict[str, Any] | None = None,
    analysis_max_dim: int = 2048,
    native_max_dim: int = 2560,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Any, list[str], list[str]]:
    """Run the real pipeline over every sample: ``(matrix, feature_names, unreadable)``.

    Rows keep their NaN. A technique that could not run leaves NaN in its columns and 0 in its
    ``.measured`` companion, which the gradient-boosted trees consume natively — dropping the row
    instead would silently bias the corpus toward images that happen to be JPEGs.

    A sample whose *container* cannot be decoded is skipped and named in ``unreadable``. Its label
    would attach to a row of nothing but NaN, and the classifier would learn "everything missing"
    as a class.

    ``config_overrides`` patches the worker config — a specific ``cnn_weights_path``, say — so this
    can score with the checkpoint under test rather than whatever the environment points at.
    """
    import numpy as np

    from app.analysis import registry as R
    from app.analysis.explainers import explain
    from app.analysis.features import FEATURES, extract
    from app.analysis.types import TaskOutcome
    from app.config import get_settings
    from app.core.orchestrator import worker_config
    from app.core.pipeline import build_arrays
    from app.core.security import probe_header

    cfg = worker_config(get_settings())
    cfg.update(config_overrides or {})

    # Warm the process-wide runtime the same way the API's startup hook does. ``available`` is False
    # until ``load()`` has run, so without this the whole corpus scores with the CNN reported
    # unavailable — a silent, uniform NaN in five columns that a fusion model then learns as
    # "missing detector is a feature".
    weights = cfg.get("cnn_weights_path")
    if weights and Path(weights).is_file():
        try:
            from app.ml.cnn_runtime import get_runtime

            get_runtime(str(weights)).load()
        except Exception:  # noqa: BLE001 - torch missing here just means the cnn columns stay NaN
            pass

    def _bind(value: Any, arrays: Any) -> Any:
        if isinstance(value, R.FromArrays):
            return getattr(arrays, value.field)
        if isinstance(value, R.FromConfig):
            return cfg.get(value.field)
        return value.value

    rows: list[dict[str, float]] = []
    unreadable: list[str] = []
    for i, sample in enumerate(samples):
        if progress:
            progress(i, len(samples))
        try:
            data = Path(sample.path).read_bytes()
            arrays = build_arrays(data, probe_header(data),
                                  analysis_max_dim=analysis_max_dim,
                                  native_max_dim=native_max_dim)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is reported, not fatal
            unreadable.append(f"{sample.path}: {type(exc).__name__}: {str(exc)[:120]}")
            continue

        outcomes: dict[str, TaskOutcome] = {}
        explanations: dict[str, Any] = {}
        for spec in R.TECHNIQUES:
            args = [_bind(a, arrays) for a in spec.args]
            kwargs = {k: _bind(v, arrays) for k, v in spec.kwargs.items()}
            try:
                mod = __import__(f"app.analysis.techniques.{spec.module}", fromlist=["run"])
                res = mod.run(*args, **kwargs)
                oc = TaskOutcome(name=spec.name, status="ok", score=res.score,
                                 details=res.details, ms=0)
            except Exception as exc:  # noqa: BLE001 - the analyser failed; record and carry on
                oc = TaskOutcome(name=spec.name, status="error", score=None,
                                 details={"error_msg": f"{type(exc).__name__}: {str(exc)[:200]}"},
                                 ms=0)
            outcomes[spec.name] = oc
            explanations[spec.name] = explain(spec.name, oc.score, oc.details, status=oc.status)

        rows.append(extract(outcomes, arrays, explanations))

    if not rows:
        raise ValueError(f"none of {len(samples)} samples could be decoded")
    matrix = np.array([[r[name] for name in FEATURES] for r in rows], dtype=np.float64)
    return matrix, list(FEATURES), unreadable


def populated_columns(matrix: Any, names: Sequence[str]) -> dict[str, float]:
    """Share of rows where each technique actually produced a measurement, ascending.

    A blob whose ``cnn.*`` column is all NaN learned nothing about the learned detector and will
    keep scoring uploads as though it did. This is the number that says whether the model is worth
    shipping, and it belongs in the report next to the AUC rather than buried in a log.
    """
    import numpy as np

    out: dict[str, float] = {}
    for i, name in enumerate(names):
        if name.endswith(".measured"):
            out[name[: -len(".measured")]] = round(float(np.mean(matrix[:, i] > 0.5)), 3)
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


# ---------------------------------------------------------- generated corpus
def build_generated_corpus(
    out: str | os.PathLike[str],
    *,
    groups_per_class: int = 6,
    variants_per_group: int = 2,
    size: tuple[int, int] = (640, 800),
    seed: int = 20260103,
) -> dict[str, int]:
    """Materialise a labelled corpus from constructions whose truth is known by definition.

    This is the ``--generated`` path. It exists to exercise the trainer end to end on a machine
    that has no gated datasets, *not* to produce a shippable model: the labels are only as good as
    the construction, so an ``edited`` image here is a splice with a known offset rather than one
    someone annotated. ``calibrate_fusion.py`` therefore refuses to write into ``artifacts/`` from
    a generated corpus unless told, in words, that the corpus was synthetic.

    The constructions come from ``backend/tests/fixtures/make_fixtures.py``, driven at other seeds
    and saved as two JPEG qualities of one scene per group — which is what makes a group a group
    here, and what the splitter then keeps together.
    """
    from tests.fixtures import make_fixtures as M

    base = Path(out)
    if base.exists() and any(_walk(base)):
        raise ValueError(f"{base} already holds images; refusing to mix corpora — pass a fresh directory")

    h, w = size
    counts: dict[str, int] = dict.fromkeys(CLASSES, 0)
    for cls in CLASSES:
        (base / cls).mkdir(parents=True, exist_ok=True)

    for g in range(groups_per_class):
        for cls in CLASSES:
            scene_seed = seed + 1000 * (g + 1) + _CLASS_OFFSET[cls]
            plate = _scene(M, cls, h, w, scene_seed, variant=g)
            group_dir = base / cls / f"g{g:03d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for v in range(variants_per_group):
                if cls == "ai_generated" and v % 2 == 0:
                    # A generator's output usually reaches us as a PNG with no camera EXIF on it;
                    # the container itself is part of what the metadata and C2PA heads are trained
                    # to read, so both containers have to appear.
                    M.save_png(plate, str(group_dir / f"frame_{v}.png"))
                else:
                    M.save_jpeg(
                        plate, str(group_dir / f"frame_{v}.jpg"),
                        quality=_VARIANT_QUALITIES[v % len(_VARIANT_QUALITIES)],
                        exif=M.CAMERA_EXIF if cls == "authentic" else None)
                counts[cls] += 1
    return counts


#: Separate seed ranges per class, so two classes never build the same scene by accident.
_CLASS_OFFSET = {"authentic": 0, "ai_generated": 70_000, "edited": 30_000, "ai_and_edited": 50_000}


def _scene(M: Any, cls: str, h: int, w: int, scene_seed: int, *, variant: int) -> Any:
    if cls == "authentic":
        return M.make_authentic(h, w, seed=scene_seed)[0]
    if cls == "ai_generated":
        return M.make_ai_generated(h, w, seed=scene_seed)[0]
    if cls == "edited":
        return _edited(M, h, w, scene_seed, _EDITED_MECHANISMS[variant % len(_EDITED_MECHANISMS)])
    return _generated_then_spliced(M, h, w, scene_seed)


def _edited(M: Any, h: int, w: int, scene_seed: int, mechanism: str) -> Any:
    builders = {
        "spliced": M.make_spliced,
        "copy_move": M.make_copy_move,
        "double_compressed": M.make_double_compressed,
    }
    return builders[mechanism](h, w, seed=scene_seed)[0]


def _generated_then_spliced(M: Any, h: int, w: int, scene_seed: int) -> Any:
    """A generated plate with a region replaced from elsewhere: ``(1, 1)`` in one file.

    The paste is deliberately not 8-aligned, so the block-grid phase inside the box disagrees with
    the plate's the way a real selection does. Without a correlated case in the corpus the two
    heads never see the interaction, and the manipulation head learns to answer "no" whenever the
    AI head says "yes".
    """
    import numpy as np
    from PIL import Image

    plate = M.make_ai_generated(h, w, seed=scene_seed)[0]
    # The donor is built at the constructions' own minimum rather than at half the plate. Halving a
    # small image lands below the size ``make_spliced`` can place a box in, and the failure is a
    # broadcast error inside a backend test helper — which is not where anyone would think to look
    # for a corpus size limit.
    donor = _edited(M, max(h // 2, 240), max(w // 2, 320), scene_seed + 977, "spliced")
    dy, dx = h // 3 + 3, w // 2 + 5  # the +3/+5 keep the paste off the 8-pixel DCT grid
    box_h = max(16, min(h // 4, donor.shape[0], h - dy))
    box_w = max(16, min(w // 5, donor.shape[1], w - dx))
    patch = np.asarray(
        Image.fromarray(donor[:box_h, :box_w]).resize((box_w, box_h), Image.BILINEAR))
    plate[dy: dy + box_h, dx: dx + box_w] = patch
    return M.jpeg_roundtrip(plate, quality=86)
