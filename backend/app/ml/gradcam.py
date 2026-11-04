"""Grad-CAM for EfficientNet-B0.

Three bugs in the prototype's ``gradcam()`` are fixed here.

**1. In-place SiLU corrupts the autograd graph.** ``torchvision``'s ``_ConvNormAct`` blocks are
built with ``nn.SiLU(inplace=True)``, and ``model.features[-1]`` ends in one. The in-place
activation overwrites the buffer ``SiluBackward0`` saved, so a backward hook on that module
raises ``RuntimeError: one of the variables needed for gradient computation has been modified
by an inplace operation``. We sweep every in-place activation off **once at load time**, and
assert the sweep changed something so a future torchvision refactor fails loudly instead of
silently reintroducing the bug.

**2. Hooks were never removed.** ``register_forward_hook`` returns a ``RemovableHandle`` that the
prototype discarded. In a long-lived server the module accumulates one hook per request — an
unbounded memory and CPU leak. Here the handle is removed in a ``finally``.

**3. ``register_full_backward_hook`` is the wrong tool.** We use a *forward* hook to capture the
activation and ``torch.autograd.grad`` to get its gradient. That is a tensor-level operation, so
module in-place semantics no longer matter, and it does not populate ``.grad`` on parameters —
no cross-request gradient contamination and no ``zero_grad()`` bookkeeping.

**4. Inference-mode tensors cannot be saved for backward.** ``torch.inference_mode`` and
Grad-CAM are mutually exclusive: building the input tensor under inference mode and then
differentiating through it raises ``Inference tensors cannot be saved for backward``. The
public ``CnnRuntime.encode()`` therefore returns **numpy**, and each path builds its own fresh
tensor. Grad-CAM additionally wraps itself in ``torch.enable_grad()`` to escape any ambient
inference mode.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["GRADCAM_LAYER", "disable_inplace_activations", "gradcam", "verify_gradcam_layer"]

#: Last conv block of EfficientNet-B0: Conv2dNormActivation(320 -> 1280, 1x1).
GRADCAM_LAYER = "features.8"
#: Expected activation shape for a 224x224 input. Asserted at startup.
EXPECTED_CHANNELS = 1280
EXPECTED_SPATIAL = (7, 7)

_INPLACE_TYPES = (nn.SiLU, nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardtanh, nn.ELU, nn.CELU)


def disable_inplace_activations(model: nn.Module) -> int:
    """Turn off every in-place activation. Returns the number modified."""
    n = 0
    for module in model.modules():
        if isinstance(module, _INPLACE_TYPES) and getattr(module, "inplace", False):
            module.inplace = False
            n += 1
    return n


def verify_gradcam_layer(model: nn.Module, input_size: int = 224) -> tuple[int, tuple[int, int]]:
    """Confirm the hooked layer still produces the activation shape Grad-CAM assumes.

    If torchvision ever reorders ``features``, the CAM resolution changes silently and the
    heatmap stops corresponding to the image. This makes that loud at startup.
    """
    layer = model.get_submodule(GRADCAM_LAYER)
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda m, i, o: captured.append(o))
    try:
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, input_size, input_size))
        model.train(was_training)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"Grad-CAM layer {GRADCAM_LAYER} produced no activation.")
    act = captured[0]
    channels = int(act.shape[1])
    spatial = (int(act.shape[2]), int(act.shape[3]))
    if channels != EXPECTED_CHANNELS or spatial != EXPECTED_SPATIAL:
        raise RuntimeError(
            f"Grad-CAM layer {GRADCAM_LAYER} produced {channels} channels at {spatial}, "
            f"expected {EXPECTED_CHANNELS} at {EXPECTED_SPATIAL}. torchvision's EfficientNet "
            "layout has changed; GRADCAM_LAYER must be re-pointed at the last conv block."
        )
    return channels, spatial


def gradcam(
    model: nn.Module, arr_f32: np.ndarray, layer: str = GRADCAM_LAYER
) -> np.ndarray:
    """Return a normalised [0,1] float32 CAM at the layer's spatial resolution.

    ``arr_f32`` is a (1, 3, H, W) **numpy** array. A fresh tensor is created from a copy so the
    result is never an inference tensor.
    """
    activations: list[torch.Tensor] = []
    target = model.get_submodule(layer)
    handle = target.register_forward_hook(lambda m, i, o: activations.append(o))
    try:
        with torch.enable_grad():
            x = torch.from_numpy(np.ascontiguousarray(arr_f32.copy()))
            logit = model(x).squeeze()
            if activations[0].requires_grad is False:
                raise RuntimeError(
                    "Grad-CAM target has no gradient path; the model is probably in "
                    "inference mode or its parameters are frozen without grad enabled."
                )
            grad = torch.autograd.grad(
                outputs=logit,
                inputs=activations[0],
                grad_outputs=torch.ones_like(logit),
                retain_graph=False,
                create_graph=False,
            )[0]
            act = activations[0]
            # Global-average-pooled gradient weights, then the standard ReLU on the weighted sum.
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1)).squeeze(0)
            cam = cam.detach().to(torch.float32).cpu().numpy()
    finally:
        handle.remove()

    cam -= float(cam.min())
    denom = float(cam.max()) + 1e-8
    return (cam / denom).astype(np.float32)
