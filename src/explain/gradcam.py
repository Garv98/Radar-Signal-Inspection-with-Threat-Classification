"""
gradcam.py
==========
Grad-CAM for the ML classifier's CNN branch.

Grad-CAM highlights which Range-Doppler regions drove a given class decision.
We hook the last convolutional block of :class:`RadarClassifier.cnn_branch`,
capture its activations and gradients, and produce a [D x R] heat-map upsampled
to the input RD-map resolution.

Pure PyTorch -- no external explainability dependency.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM on a target conv module of a multi-input model.

    The wrapped model is the radar classifier whose ``forward`` signature is
    ``(spectrogram, doppler_seq, env_features)``.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        if target_layer is None:
            target_layer = self._default_target(model)
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _default_target(model: nn.Module) -> nn.Module:
        """Last Conv2d inside the CNN branch."""
        last_conv = None
        cnn = getattr(model, "cnn_branch", model)
        for m in cnn.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found for Grad-CAM target")
        return last_conv

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(
        self,
        spectrogram: torch.Tensor,
        doppler_seq: torch.Tensor,
        env_features: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Returns (cam [Hin x Win] in [0,1], predicted/target class, class probs).
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(spectrogram, doppler_seq, env_features)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze(0)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        acts = self._activations          # [1, C, h, w]
        grads = self._gradients           # [1, C, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)         # GAP over spatial
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [1,1,h,w]

        in_hw = spectrogram.shape[-2:]
        cam = F.interpolate(cam, size=in_hw, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        denom = cam.max() if cam.max() > 0 else 1.0
        cam = (cam / denom).astype(np.float32)
        return cam, class_idx, probs.astype(np.float32)

    def close(self):
        self._fwd.remove()
        self._bwd.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def gradcam_on_rd_map(
    model: nn.Module,
    rd_map: np.ndarray,
    doppler: np.ndarray,
    env: np.ndarray,
    class_idx: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Convenience wrapper that runs Grad-CAM directly on numpy feature tensors."""
    dev = torch.device(device)
    spec = torch.from_numpy(np.asarray(rd_map, np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
    dop = torch.from_numpy(np.asarray(doppler, np.float32)).unsqueeze(0).to(dev)
    ev = torch.from_numpy(np.asarray(env, np.float32)).unsqueeze(0).to(dev)
    cam_obj = GradCAM(model.to(dev))
    try:
        return cam_obj(spec, dop, ev, class_idx)
    finally:
        cam_obj.close()
