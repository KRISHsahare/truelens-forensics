"""Pixel-space augmentation, and the preprocessing the deployed model actually sees.

The generalisation claim in the spec rests on these transforms, so they are not cosmetic:

* **JPEG re-encode at a random quality.** Every real photograph a user will upload has been
  compressed — by the camera, by WhatsApp, by a screenshot tool, by a web resizer — and each
  pass leaves a quantisation signature. A model trained on clean PNGs learns texture statistics
  that survive exactly zero of those round trips.
* **Resize round trip.** Down-then-up scales destroy the high-frequency detail a generator's
  checkerboard pattern lives in, which is the most common way a deployed detector goes quiet.
* **Blur.** A soft-focus photograph must not read as synthetic because it lacks sensor noise.
* **Random crop and flip.** Standard, and the flip is the same one ``cnn_runtime.score(tta=True)``
  averages over, so training and inference agree about which symmetries are real.

Everything happens on ``uint8`` arrays before normalisation, because JPEG compression of a
normalised float tensor is not JPEG compression — it is noise. ``letterbox`` mirrors
``CnnRuntime.encode_for_resize`` deliberately: a model trained on a squashed 3:2 face and deployed
on a letterboxed one is a model whose accuracy figure describes a different pipeline.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "letterbox", "to_tensor", "transform_train"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Pad value from ``CnnRuntime.encode_for_resize``. Kept as a constant on both sides of the
#: boundary so a change here has to be made there too, in the same commit.
_PAD = 114


def letterbox(rgb_u8: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize onto a ``size`` square, padded with ``_PAD``."""
    h, w = rgb_u8.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), _PAD, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    return canvas


def to_tensor(rgb_u8: np.ndarray,
              *,
              mean=IMAGENET_MEAN,
              std=IMAGENET_STD) -> "np.ndarray":
    """``uint8 HWC`` -> float32 CHW in ImageNet units. Returns numpy; the caller makes a tensor.

    Numpy rather than torch so this module stays importable — and testable — without torch in
    the process, which is what lets the analyser suite run in a spawn child without paying for
    the import.
    """
    arr = rgb_u8.astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def transform_train(rgb_u8: np.ndarray,
                    *,
                    size: int = 224,
                    rng: random.Random | None = None) -> np.ndarray:
    """One augmentation draw, ending in a letterboxed ``size x size x 3`` uint8 image."""
    r = rng or random
    img = rgb_u8

    if r.random() < 0.5:
        # Random-resized-crop, but with a floor of 0.65 of the frame: a generator's artefacts are
        # a global texture property, and a 10%-of-frame crop asks the model to infer it from a
        # patch that may contain none of them.
        h, w = img.shape[:2]
        scale = r.uniform(0.65, 1.0)
        ch, cw = max(8, int(h * scale)), max(8, int(w * scale))
        top, left = r.randint(0, h - ch), r.randint(0, w - cw)
        img = img[top:top + ch, left:left + cw]

    if r.random() < 0.5:
        img = img[:, ::-1]

    if r.random() < 0.35:
        img = _jpeg_roundtrip(img, quality=r.randint(45, 95))

    if r.random() < 0.30:
        # Down then up: removes the detail band a naive detector over-fits to.
        factor = r.uniform(0.35, 0.7)
        small = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    if r.random() < 0.25:
        k = int(r.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    if r.random() < 0.4:
        img = _photometric(img, r)

    return letterbox(np.ascontiguousarray(img), size)


def _jpeg_roundtrip(rgb_u8: np.ndarray, *, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:  # pragma: no cover - only a memory-exhausted encoder gets here
        return rgb_u8
    back = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(back, cv2.COLOR_BGR2RGB)


def _photometric(rgb_u8: np.ndarray, r: random.Random) -> np.ndarray:
    """Brightness/contrast in the range a real re-encode produces, and nothing wider.

    A colour-jitter range borrowed from an ImageNet classifier is too aggressive here: extreme
    hue shifts create the very texture statistics the detector is trying to measure, so the model
    learns to call saturation a forgery.
    """
    img = rgb_u8.astype(np.float32)
    img *= r.uniform(0.85, 1.15)                       # brightness
    img = (img - 127.5) * r.uniform(0.85, 1.15) + 127.5  # contrast about the mid-grey
    return np.clip(img, 0, 255).astype(np.uint8)
