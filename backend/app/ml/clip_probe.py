"""Optional CLIP-feature linear probe (UnivFD-style detector).

Rationale: a detector fine-tuned end-to-end on one generator family latches onto that family's
specific artifacts and generalises badly to the next one. A *linear probe on frozen CLIP
features* has almost no capacity, so it cannot memorise a generator's fingerprint — it only
learns the broad "synthetic vs photographic" direction that survives across architectures. That
is why it is reported **alongside** EfficientNet rather than instead of it, and why their
*agreement* is surfaced: when two detectors with very different inductive biases concur, the
verdict is worth more than either alone.

This is an optional extra (``pip install -e .[clip]``). It is lazy-imported and every failure
mode degrades to ``available = False`` rather than raising, so a missing dependency can never
take the API down. The ViT-L/14 weights are ~900 MB and are downloaded on first use — which is
why the whole path is opt-in and disabled by default.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ClipProbe", "get_probe", "reset_probe"]

ARCH = "ViT-L-14"
PRETRAINED = "openai"
INPUT_SIZE = 224


class ClipProbe:
    """Frozen CLIP image encoder + a trained linear head."""

    def __init__(self, probe_path: str, *, enabled: bool = False) -> None:
        self.probe_path = probe_path
        self.enabled = enabled
        self._model = None
        self._preprocess = None
        self._head = None
        self._load_error: str | None = None
        self._attempted = False
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self._attempted:
                return self.available
            self._attempted = True
            if not self.enabled:
                self._load_error = "disabled (set ENABLE_CLIP_PROBE=true and install .[clip])"
                return False
            try:
                self._load()
            except Exception as exc:
                self._model = self._head = None
                self._load_error = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning("CLIP probe unavailable: %s", self._load_error)
            return self.available

    def _load(self) -> None:
        import joblib
        import open_clip
        import torch

        if not os.path.exists(self.probe_path):
            raise FileNotFoundError(f"no linear probe at {self.probe_path}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            ARCH, pretrained=PRETRAINED
        )
        model.eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self._model = model
        self._preprocess = preprocess
        self._head = joblib.load(self.probe_path)
        logger.info("CLIP probe loaded (%s/%s)", ARCH, PRETRAINED)

    @property
    def available(self) -> bool:
        return self._model is not None and self._head is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def features(self, rgb_u8: np.ndarray) -> np.ndarray | None:
        """Frozen CLIP image embedding, L2-normalised. numpy in, numpy out."""
        if not self.available:
            return None
        import torch
        from PIL import Image

        try:
            pil = Image.fromarray(rgb_u8).convert("RGB")
            x = self._preprocess(pil).unsqueeze(0)
            with torch.inference_mode():
                feat = self._model.encode_image(x)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return feat.squeeze(0).float().cpu().numpy()
        except Exception as exc:
            logger.warning("CLIP feature extraction failed: %s", exc)
            return None

    def score(self, rgb_u8: np.ndarray) -> float | None:
        feat = self.features(rgb_u8)
        if feat is None:
            return None
        try:
            return float(np.clip(self._head.predict_proba(feat[None, :])[0, 1], 0.0, 1.0))
        except Exception as exc:
            logger.warning("CLIP probe head failed: %s", exc)
            return None

    def warmup(self) -> dict[str, float]:
        if not self.available:
            return {"available": 0.0}
        import time

        dummy = (np.random.rand(INPUT_SIZE, INPUT_SIZE, 3) * 255).astype(np.uint8)
        t0 = time.perf_counter()
        self.score(dummy)
        return {"warm_score_ms": round((time.perf_counter() - t0) * 1000.0, 1)}


_PROBE: ClipProbe | None = None
_PROBE_LOCK = threading.Lock()


def get_probe(probe_path: str | None = None, *, enabled: bool = False) -> ClipProbe:
    global _PROBE
    with _PROBE_LOCK:
        if _PROBE is None:
            _PROBE = ClipProbe(probe_path or "artifacts/clip_probe.joblib", enabled=enabled)
        return _PROBE


def reset_probe() -> None:
    global _PROBE
    with _PROBE_LOCK:
        _PROBE = None
