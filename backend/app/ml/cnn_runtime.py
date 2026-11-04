"""CNN inference runtime.

The prototype rebuilt the model and called ``torch.load`` **on every request** — ~400 ms and a
fresh 16 MB allocation per analysis, twice (once for the score, again for Grad-CAM). Here the
weights are loaded once at startup and the instance is shared.

Design constraints:

* ``encode()`` returns **numpy**, not a torch tensor. Scoring runs under
  ``torch.inference_mode``; Grad-CAM cannot. If a single tensor were shared between the two
  paths, Grad-CAM would raise ``Inference tensors cannot be saved for backward``. Building a
  fresh tensor inside each path makes that structurally impossible.
* Scoring and Grad-CAM are called from a single-threaded executor owned by the app, so there is
  no concurrent graph mutation on the shared module. Nothing here is locked, deliberately:
  adding a lock would serialise what the executor already serialises.
* If the weights are missing, ``available`` is False and the CNN technique reports
  ``unavailable`` — which becomes ``NaN`` in the fusion feature vector, *not* ``0.0``. A
  missing detector must never read as "no evidence of manipulation".
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CnnRuntime", "get_runtime", "reset_runtime"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224


class CnnRuntime:
    """Holds one EfficientNet-B0 with a single-logit head."""

    def __init__(self, weights_path: str, *, torch_threads: int = 4) -> None:
        self.weights_path = weights_path
        self.torch_threads = max(1, torch_threads)
        self._model = None
        self._load_error: str | None = None
        self._loaded = False
        self._lock = threading.Lock()
        self.input_size = INPUT_SIZE
        self.last_gradcam_ms: float = 0.0

    # ---------------------------------------------------------------- lifecycle
    def load(self) -> bool:
        """Load weights once. Idempotent and thread-safe. Never raises."""
        with self._lock:
            if self._loaded:
                return self.available
            self._loaded = True
            try:
                self._model = self._build_and_load()
                logger.info("CNN loaded from %s", self.weights_path)
            except Exception as exc:
                self._model = None
                self._load_error = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning("CNN unavailable: %s", self._load_error)
            return self.available

    def _build_and_load(self):
        import torch
        from torchvision import models

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"no weights at {self.weights_path}; train them with ml/scripts/train_cnn.py"
            )
        torch.set_num_threads(self.torch_threads)
        model = models.efficientnet_b0(weights=None)
        head = model.classifier[1]
        model.classifier[1] = torch.nn.Linear(head.in_features, 1)

        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Accept a checkpoint saved from a DataParallel wrapper or with a "model." prefix.
        cleaned = {}
        for k, v in state.items():
            nk = k.removeprefix("module.").removeprefix("model.")
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=True)

        final = model.classifier[1]
        if getattr(final, "out_features", 1) != 1:
            raise ValueError(
                f"head has {final.out_features} outputs, expected 1 (binary BCEWithLogits)"
            )
        model.eval()

        # Must happen after load: the in-place sweep mutates module attributes.
        from app.ml.gradcam import disable_inplace_activations, verify_gradcam_layer
        n = disable_inplace_activations(model)
        if n == 0:
            raise RuntimeError(
                "no in-place activations found to disable — torchvision's EfficientNet layout "
                "changed; Grad-CAM's SiLU fix must be re-verified."
            )
        verify_gradcam_layer(model, self.input_size)
        logger.info("disabled %d in-place activations for Grad-CAM", n)
        return model

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ------------------------------------------------------------------- encode
    def encode(self, rgb_u8: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> (1, 3, 224, 224) float32 numpy. No torch touched here."""
        s = self.input_size
        h, w = rgb_u8.shape[:2]
        if (h, w) != (s, s):
            resized = cv2.resize(rgb_u8, (s, s), interpolation=cv2.INTER_AREA)
        else:
            resized = rgb_u8
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return chw[None, ...]

    def encode_for_resize(self, rgb_u8: np.ndarray) -> np.ndarray:
        """Variant that preserves aspect ratio via letterbox padding.

        Squashing a 3:2 photo to a square distorts faces and hands — the regions the model was
        trained to scrutinise. Letterboxing keeps proportions at the cost of some resolution.
        """
        s = self.input_size
        h, w = rgb_u8.shape[:2]
        scale = s / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        small = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        top, left = (s - nh) // 2, (s - nw) // 2
        canvas[top:top + nh, left:left + nw] = small
        return self.encode(canvas)

    # -------------------------------------------------------------------- score
    def score(self, arr_f32: np.ndarray, *, tta: bool = False) -> float | None:
        """P(fake) in [0, 1], or None when the model is unavailable."""
        if not self.available:
            return None
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(arr_f32)
            if tta:
                # Horizontal flip is the one augmentation that is safe for this task: faces are
                # not chiral in a way that changes authenticity. Batched so it costs one pass.
                x = torch.cat([x, x.flip(-1)], dim=0)
            logits = self._model(x).squeeze(-1)
            probs = torch.sigmoid(logits)
            return float(probs.mean().item())

    def gradcam(self, arr_f32: np.ndarray) -> np.ndarray | None:
        """Normalised [0,1] CAM at 7x7, or None when unavailable."""
        if not self.available:
            return None
        import time

        from app.ml.gradcam import gradcam as _gradcam

        t0 = time.perf_counter()
        cam = _gradcam(self._model, arr_f32)
        self.last_gradcam_ms = (time.perf_counter() - t0) * 1000.0
        return cam

    # ------------------------------------------------------------------- warmup
    def warmup(self) -> dict[str, float]:
        """Run a dummy forward and a dummy backward before the service accepts traffic.

        First-inference cost is 3-10x steady state (oneDNN kernel selection, autograd graph
        setup). Without this the first real user request looks like a total failure.
        """
        import time

        if not self.available:
            return {"available": 0.0}
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        t0 = time.perf_counter()
        self.score(dummy, tta=True)
        forward_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        self.gradcam(dummy)
        backward_ms = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        self.score(dummy, tta=False)
        steady_ms = (time.perf_counter() - t2) * 1000.0
        return {"warm_forward_ms": round(forward_ms, 1),
                "warm_gradcam_ms": round(backward_ms, 1),
                "steady_forward_ms": round(steady_ms, 1)}


_RUNTIME: CnnRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def get_runtime(weights_path: str | None = None, *, torch_threads: int = 4) -> CnnRuntime:
    """Process-wide singleton. The whole point is that the model is loaded exactly once."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = CnnRuntime(weights_path or "artifacts/model.pt",
                                  torch_threads=torch_threads)
        elif weights_path and _RUNTIME.weights_path != weights_path:
            _RUNTIME = CnnRuntime(weights_path, torch_threads=torch_threads)
        return _RUNTIME


def reset_runtime() -> None:
    """Test helper: drop the singleton so a different weights path can be loaded."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = None
