"""The detector architecture, defined once so the trainer and the runtime cannot disagree.

``app.ml.cnn_runtime.CnnRuntime._build_and_load`` constructs
``efficientnet_b0`` with ``classifier[1]`` replaced by a single-logit ``Linear`` and loads a
checkpoint into it with ``strict=True``. That is the deployment contract. If this module built a
slightly different head — two logits, a dropout change, a different backbone — training would
still succeed and produce a file, and the API would refuse to load it at startup, which is a
failure loud enough to notice but slow enough to waste a deploy.

So the shape lives here, ``cnn_runtime`` keeps its own copy for the inference path (it must not
import training code into a spawn worker), and ``check_artifacts.py`` asserts the two agree every
time an artifact is produced. The assertion is the thing that makes the duplication safe.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["ARCHITECTURE", "build_model", "load_checkpoint", "save_checkpoint"]

#: Written into every checkpoint and into metrics.json, so a number on the Methodology page can be
#: traced to the exact configuration that produced it.
ARCHITECTURE = "efficientnet_b0_adamw_bce_1logit"


def build_model(*, pretrained: bool = True, dropout: float = 0.2):
    """EfficientNet-B0 with a one-logit head.

    ``pretrained`` degrades rather than raising when the ImageNet weights cannot be fetched. On a
    machine with no egress that is the difference between "you can verify the pipeline" and "you
    cannot run anything"; the checkpoint metadata records which one happened, so a randomly
    initialised model can never be mistaken for an ImageNet-initialised one later.
    """
    import torch
    from torchvision import models

    weights = "DEFAULT" if pretrained else None
    try:
        model = models.efficientnet_b0(weights=weights)
        init = "imagenet1k"
    except Exception as exc:  # offline, or a torchvision layout change
        model = models.efficientnet_b0(weights=None)
        init = f"random ({type(exc).__name__})"
    if dropout != 0.2:
        model.classifier[0] = torch.nn.Dropout(p=dropout)
    head = model.classifier[1]
    model.classifier[1] = torch.nn.Linear(head.in_features, 1)
    model.truelens_init = init  # type: ignore[attr-defined]
    return model


def save_checkpoint(model: Any, path: str | os.PathLike[str], *, meta: dict[str, Any]) -> str:
    """Write ``{"state_dict": ..., "meta": {...}}``.

    The wrapper is what ``cnn_runtime`` accepts alongside a bare state dict, and the metadata is
    the only surviving record of what the file is: which corpus, which split, which initialisation.
    Weights without provenance are the artifact that makes a methodology page unanswerable a year
    later.
    """
    import torch

    target = os.fspath(path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    state = model.state_dict()
    torch.save({"state_dict": state, "meta": meta}, target)
    return target


def load_checkpoint(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(state_dict, meta)`` from either supported checkpoint shape.

    ``weights_only=True`` for the same reason the runtime uses it: a checkpoint is a file someone
    else produced, and unpickling one is arbitrary code execution. That constrains ``meta`` to
    JSON primitives, which is what it should have been limited to anyway.
    """
    import torch

    blob = torch.load(os.fspath(path), map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        return dict(blob["state_dict"]), dict(blob.get("meta") or {})
    return dict(blob), {}
