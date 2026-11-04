"""Corpus layout, discovery, leak-free splitting, and the torch dataset.

**The label convention.** ``1`` means *not an untouched photograph from a camera* — generated,
rendered, or synthesised. ``0`` means a real camera capture that has not been through a generator.
This is the convention ``app.ml.cnn_runtime`` scores against: its single logit passed through
sigmoid is P(fake). Flip the labels here and the deployed detector inverts, silently, while every
number on the Methodology page still looks plausible.

**Layout.** A corpus is a directory tree, not a CSV:

    <root>/
      real/<group>/<image>          # class 0
      fake/<generator>/<group>/<image>   # class 1, one directory per generator family

``<generator>`` is load-bearing. A detector trained on one generator family and tested on another
generalises worse than its headline accuracy suggests, so evaluation reports per-family numbers
wherever the corpus has more than one family — which is the whole reason UnivFD-style reporting is
per-generator rather than pooled.

``<group>`` is what splitting respects. Frames of one video, crops of one scan, and the burst from
one prompt are near-duplicates of each other; putting frame 12 in train and frame 13 in val
measures memorisation and reports it as accuracy. So a split is assigned by group, and a group
that cannot be identified from the path is treated as one file per group rather than silently
pooling it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "CLASSES",
    "FAKE",
    "REAL",
    "Sample",
    "SplitName",
    "discover",
    "split_samples",
    "tally",
    "write_manifest",
]

REAL = 0
FAKE = 1
CLASSES: tuple[str, str] = ("real", "fake")

SplitName = str

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

#: Directories a corpus tree may contain that are not class roots.
_SKIP_DIRS = frozenset({".ipynb_checkpoints", ".DS_Store", "__pycache__"})


@dataclass(frozen=True, slots=True)
class Sample:
    """One image, with everything the split and the per-family report need."""

    path: str
    label: int
    #: ``real`` or the generator family directory name.
    source: str
    #: Near-duplicate group: files sharing it never straddle a split boundary.
    group: str

    @property
    def is_fake(self) -> bool:
        return self.label == FAKE

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "label": self.label, "source": self.source, "group": self.group}


def _stable_unit(text: str, buckets: int) -> int:
    """A deterministic bucket id.

    ``hash()`` is salted per process on Windows and Linux, so a split built with it would move
    files between train and val on the next run and quietly leak. ``sha1`` is not a randomiser,
    it is a stable id — which is exactly what a reproducible split needs.
    """
    digest = hashlib.sha1(text.encode("utf-8", "surrogateescape")).hexdigest()
    return int(digest[:12], 16) % max(1, buckets)


def discover(root: str | os.PathLike[str], *, limit: int | None = None) -> list[Sample]:
    """Read the documented tree into samples. Raises when the layout is wrong.

    An empty corpus is a hard error rather than an empty list: a training script that "succeeded"
    on zero images and wrote a checkpoint of untrained weights is the worst possible outcome,
    because the artifact then loads and scores confidently forever.
    """
    root = Path(root)
    real_root = root / "real"
    fake_root = root / "fake"
    if not real_root.is_dir() or not fake_root.is_dir():
        raise FileNotFoundError(
            f"{root} is not a corpus: expected {root / 'real'} and {root / 'fake'} to exist. "
            "See ml/README.md for the layout, or build one with ml/scripts/make_corpus.py."
        )

    samples: list[Sample] = []
    for path in sorted(_walk(real_root)):
        samples.append(Sample(str(path), REAL, "real", _group_for(path, real_root)))
    for family_dir in sorted(p for p in fake_root.iterdir() if p.is_dir()):
        if family_dir.name in _SKIP_DIRS:
            continue
        for path in sorted(_walk(family_dir)):
            samples.append(Sample(str(path), FAKE, family_dir.name, _group_for(path, family_dir)))

    if not samples:
        raise FileNotFoundError(f"{root} matched no images under real/ or fake/*/")
    counts = _counts(samples)
    if 0 in counts.values():
        raise ValueError(
            f"corpus {root} has only one class ({counts}); a single-class training run produces "
            "a constant predictor that reports 100% accuracy on the majority class."
        )
    if limit is not None:
        # Truncation must stay balanced or a --limit run teaches the model a class prior instead
        # of a visual concept.
        samples = _balanced_truncate(samples, limit)
    return samples


def _counts(samples: Sequence[Sample]) -> dict[str, int]:
    """Class sizes, keyed by the names the metrics file uses."""
    return {"real": sum(1 for s in samples if s.label == REAL),
            "fake": sum(1 for s in samples if s.label == FAKE)}


def _balanced_truncate(samples: Sequence[Sample], limit: int) -> list[Sample]:
    per_class = max(1, limit // 2)
    out: list[Sample] = []
    for label in (REAL, FAKE):
        pool = [s for s in samples if s.label == label]
        out.extend(pool[:per_class])
    return out


def _walk(directory: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def _group_for(path: Path, class_root: Path) -> str:
    """The directory one level above the file, or the file itself when it sits at the root.

    ``real/burst01/frame_0012.jpg`` groups by ``burst01``. A loose file at the class root has no
    siblings to protect against, so it is its own group.
    """
    parent = path.parent
    if parent == class_root:
        return f"file:{path.name}"
    return str(parent.relative_to(class_root)).replace(os.sep, "/")


def _group_order(pool: Sequence[Sample], seed: int) -> list[tuple[tuple[str, str], list[Sample]]]:
    """Groups of one class, in a stable order that does not depend on directory listing.

    The order is a hash of the group key, so adding images to the corpus cannot move an existing
    group across a split boundary — which would make two consecutive runs' validation sets
    incomparable — and taking a prefix of it is deterministic without a seeded RNG reaching into
    the dataset.
    """
    groups: dict[tuple[str, str], list[Sample]] = {}
    for sample in pool:
        groups.setdefault((sample.source, sample.group), []).append(sample)
    return sorted(groups.items(), key=lambda kv: _stable_unit(f"{seed}:{kv[0][0]}:{kv[0][1]}", 10 ** 9))


def split_samples(
    samples: Sequence[Sample],
    *,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 7,
) -> dict[SplitName, list[Sample]]:
    """Split by group, stratified by class.

    Two properties, and the naive implementations fail one each. Splitting *files* at random
    leaks: frames 12 and 13 of the same video land on opposite sides of the boundary and the
    reported accuracy measures memorisation. Splitting *groups* by a hash fixes the leak but, on a
    corpus with a dozen groups, can leave a split holding one class — at which point its metrics
    are undefined. So the split is computed per class, over an atomically-grouped, hash-ordered
    list, with a floor that guarantees each split receives at least one group of each class
    whenever the class has three or more groups to give.
    """
    if abs(sum(fractions) - 1.0) > 1e-6 or len(fractions) != 3:
        raise ValueError(f"fractions must be three weights summing to 1, got {tuple(fractions)}")
    out: dict[SplitName, list[Sample]] = {"train": [], "val": [], "test": []}
    for label in (REAL, FAKE):
        ordered = _group_order([s for s in samples if s.label == label], seed)
        n = len(ordered)
        if n < 3:
            raise ValueError(
                f"class {CLASSES[label]} has only {n} group(s) in this corpus; a split needs at "
                "least three groups per class or every per-class metric is computed on a subset "
                "of one group's near-duplicates."
            )
        cut_train = min(max(1, round(fractions[0] * n)), n - 2)
        cut_val = min(cut_train + max(1, round(fractions[1] * n)), n - 1)
        for index, (_, members) in enumerate(ordered):
            name = "train" if index < cut_train else "val" if index < cut_val else "test"
            out[name].extend(members)
    for name, part in out.items():
        counts = _counts(part)
        if 0 in counts.values():
            raise ValueError(
                f"split {name} contains only one class ({counts}); widen the fractions or use a "
                "corpus with more groups — per-class metrics on a single-class split are undefined."
            )
    return out


def tally(samples: Sequence[Sample]) -> dict[str, object]:
    """A JSON-safe summary for the metrics file: how much data, from where, in which split."""
    by_source: dict[str, int] = {}
    for sample in samples:
        by_source[sample.source] = by_source.get(sample.source, 0) + 1
    return {
        "images": len(samples),
        "real": sum(1 for s in samples if s.label == REAL),
        "fake": sum(1 for s in samples if s.label == FAKE),
        "groups": len({(s.source, s.group) for s in samples}),
        "by_source": dict(sorted(by_source.items())),
    }


def write_manifest(path: str | os.PathLike[str], splits: dict[SplitName, list[Sample]]) -> None:
    """Write the split assignment as JSON so an evaluation run can be reproduced exactly.

    Without this, ``metrics.json`` claims a number nobody can re-derive: the same corpus with a
    different directory listing order would produce a different test set and a different AUC, and
    the discrepancy would be invisible.
    """
    import json

    payload = {
        "splits": {name: [s.as_dict() for s in part] for name, part in splits.items()},
        "summary": {name: tally(part) for name, part in splits.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------- torch dataset
def build_dataset(
    samples: Sequence[Sample],
    *,
    img_size: int = 224,
    augment: bool = False,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
):
    """``(image tensor, label, index)`` for ``torch.utils.data.DataLoader``.

    Decoding goes through Pillow with ``Image.MAX_IMAGE_PIXELS`` respected, and every image is
    resized to ``img_size`` the way the *runtime* does it — ``cv2.INTER_AREA`` after a letterbox —
    so a model trained on a crop-and-squash pipeline is not deployed into a letterbox pipeline.
    Train/deploy preprocessing drift is a silent accuracy loss; it is cheapest to prevent by
    importing the resize the service actually uses.
    """
    from torch.utils.data import Dataset

    class _Corpus(Dataset):
        def __init__(self) -> None:
            self.samples = list(samples)
            self.img_size = img_size
            self.augment = augment
            self.mean = tuple(mean)
            self.std = tuple(std)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, i: int):
            import numpy as np
            import torch
            from PIL import Image

            from truelens_ml.augment import letterbox, to_tensor, transform_train

            sample = self.samples[i]
            with Image.open(sample.path) as im:
                im.load()
                rgb = np.asarray(im.convert("RGB"))
            if self.augment:
                rgb = transform_train(rgb, size=self.img_size)
            else:
                rgb = letterbox(rgb, self.img_size)
            return to_tensor(rgb, mean=self.mean, std=self.std), torch.tensor(
                float(sample.label), dtype=torch.float32
            ), i

    return _Corpus()
