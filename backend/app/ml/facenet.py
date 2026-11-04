"""YuNet face detection.

We use OpenCV's bundled ``cv2.FaceDetectorYN`` rather than MediaPipe. MediaPipe costs 1.5-3.5 s
to import and ~200 MB RSS *per process* — fatal with a pool of 8 workers — links ``libGL.so.1``
and ``libglib2.0-0`` which ``python:3.12-slim`` does not ship, and its Tasks API downloads a
2.7 MB ``face_landmarker.task`` from ``storage.googleapis.com`` *at request time*. YuNet needs
no extra dependency, and its 232 KB ONNX is vendored in this repository.

This module imports ``cv2`` and ``numpy`` only. Keeping torch out of its transitive imports is
a hard constraint: every spawn child re-imports whatever ``worker.py`` pulls in.

The detector is **not picklable**, so it is built lazily per process and cached in a module
global — never passed as a ``submit()`` argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "DEFAULT_MODEL",
    "Face",
    "FaceDetectorUnavailable",
    "detect_faces",
    "get_detector",
    "reset_detector",
]

DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"
)

# YuNet returns (N, 15): box(4) + 5 landmarks x 2 coords(10) + score(1).
_LANDMARK_NAMES = ("right_eye", "left_eye", "nose_tip", "right_mouth", "left_mouth")
_BOX, _LM_START, _SCORE = slice(0, 4), slice(4, 14), 14
_EXPECTED_COLS = 15

_DETECTORS: dict[str, cv2.FaceDetectorYN] = {}


class FaceDetectorUnavailable(RuntimeError):
    """Raised when the vendored ONNX is missing or OpenCV lacks the objdetect module."""


@dataclass(slots=True, frozen=True)
class Face:
    """One detected face with named landmarks.

    YuNet gives 5 landmarks (eye *centres*, not eye corners). That supports the lighting and
    catchlight consistency checks, but not sub-pixel gaze estimation — a real limitation, and
    the reason those checks contribute modestly rather than decisively.
    """

    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: dict[str, tuple[float, float]]

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def get_detector(model_path: str = DEFAULT_MODEL,
                 input_size: tuple[int, int] = (320, 320)) -> cv2.FaceDetectorYN:
    """Return this process's cached detector, creating it on first use."""
    key = f"{model_path}:{input_size}"
    det = _DETECTORS.get(key)
    if det is not None:
        return det
    if not hasattr(cv2, "FaceDetectorYN"):
        raise FaceDetectorUnavailable("This OpenCV build has no FaceDetectorYN (objdetect).")
    if not os.path.exists(model_path):
        raise FaceDetectorUnavailable(f"YuNet model not found at {model_path}")
    # score_threshold 0.7, nms_threshold 0.3, top_k 500
    det = cv2.FaceDetectorYN_create(model_path, "", input_size, 0.7, 0.3, 500)
    _DETECTORS[key] = det
    return det


def reset_detector() -> None:
    """Test helper."""
    _DETECTORS.clear()


def detect_faces(rgb_u8: np.ndarray, model_path: str = DEFAULT_MODEL) -> list[Face]:
    """Detect faces in an **RGB** uint8 array.

    Two things that fail silently rather than loudly if you get them wrong:

    * YuNet is a ``cv2.dnn`` network whose first convolution expects **3 channels in BGR** order
      (the ``cv2.imread`` convention it was trained under). Passing a grayscale array raises
      ``Number of input channels should be multiple of 3 but got 1``; passing RGB swaps the
      channels and quietly degrades detection instead of erroring.
    * ``setInputSize`` must be called before *every* detect on a new image size. Skipping it
      does not error — it returns boxes scaled to the previous image, which is worse.
    """
    det = get_detector(model_path)
    h, w = rgb_u8.shape[:2]
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR) if rgb_u8.ndim == 3 and rgb_u8.shape[2] == 3 \
        else cv2.cvtColor(rgb_u8, cv2.COLOR_GRAY2BGR)
    det.setInputSize((int(w), int(h)))
    retval, faces = det.detect(bgr)
    if faces is None or retval != 1 or len(faces) == 0:
        return []
    if faces.shape[1] != _EXPECTED_COLS:  # pragma: no cover - guards an OpenCV API change
        raise FaceDetectorUnavailable(
            f"YuNet returned {faces.shape[1]} columns, expected {_EXPECTED_COLS}. "
            "The landmark column mapping must be re-verified for this OpenCV version."
        )
    out: list[Face] = []
    for row in faces:
        lm = row[_LM_START].reshape(5, 2)
        out.append(Face(
            x=int(round(row[0])), y=int(round(row[1])),
            w=int(round(row[2])), h=int(round(row[3])),
            score=float(row[_SCORE]),
            landmarks={name: (float(lm[i][0]), float(lm[i][1]))
                       for i, name in enumerate(_LANDMARK_NAMES)},
        ))
    out.sort(key=lambda f: f.w * f.h, reverse=True)
    return out[:16]
