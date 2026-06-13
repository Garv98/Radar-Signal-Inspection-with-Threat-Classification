"""
attribution.py
==============
Model-agnostic feature attribution in pure PyTorch (no SHAP/captum dependency).

Provides:
  * integrated_gradients  -- axiomatic attribution for any input tensor
  * occlusion_importance  -- perturbation-based importance map
  * ml_feature_importance -- which input branch (RD map / Doppler / env) drove
                             the ML classifier's decision

If the optional ``shap`` package is installed, :func:`shap_summary` exposes a
KernelExplainer over a feature-summary vector; otherwise it raises a clear
ImportError so callers can fall back to the built-in methods.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def integrated_gradients(
    forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 32,
) -> torch.Tensor:
    """
    Integrated Gradients (Sundararajan et al., 2017).

    ``forward`` maps an input tensor -> logits [B, K].  Returns an attribution
    tensor with the same shape as ``x`` (single sample, B=1 expected).
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    scaled = [baseline + (float(k) / steps) * (x - baseline) for k in range(steps + 1)]
    grads = []
    for s in scaled:
        s = s.clone().requires_grad_(True)
        logits = forward(s)
        score = logits[:, target].sum()
        grad = torch.autograd.grad(score, s)[0]
        grads.append(grad.detach())
    avg_grad = torch.stack(grads[:-1]).mean(dim=0)   # trapezoidal-ish average
    return (x - baseline) * avg_grad


def occlusion_importance(
    forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: int,
    patch: Tuple[int, int] = (4, 8),
    stride: Tuple[int, int] = (4, 8),
) -> np.ndarray:
    """
    Occlusion sensitivity for a [1,1,H,W] input: the drop in target-class
    probability when each patch is zeroed.  Returns an [H, W] importance map.
    """
    assert x.dim() == 4, "occlusion expects [1,1,H,W]"
    _, _, H, W = x.shape
    with torch.no_grad():
        base = F.softmax(forward(x), dim=1)[0, target].item()
    heat = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    ph, pw = patch
    sh, sw = stride
    for i in range(0, H, sh):
        for j in range(0, W, sw):
            xo = x.clone()
            xo[:, :, i:i + ph, j:j + pw] = 0.0
            with torch.no_grad():
                p = F.softmax(forward(xo), dim=1)[0, target].item()
            heat[i:i + ph, j:j + pw] += (base - p)
            counts[i:i + ph, j:j + pw] += 1.0
    counts[counts == 0] = 1.0
    return heat / counts


def ml_feature_importance(
    model: torch.nn.Module,
    rd_map: np.ndarray,
    doppler: np.ndarray,
    env: np.ndarray,
    target: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Branch-level importance for the multi-input radar classifier using the L1
    norm of integrated gradients on each input branch, normalised to sum to 1.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    spec = torch.from_numpy(np.asarray(rd_map, np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
    dop = torch.from_numpy(np.asarray(doppler, np.float32)).unsqueeze(0).to(dev)
    ev = torch.from_numpy(np.asarray(env, np.float32)).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(spec, dop, ev)
    if target is None:
        target = int(logits.argmax(dim=1).item())

    ig_spec = integrated_gradients(lambda s: model(s, dop, ev), spec, target)
    ig_dop = integrated_gradients(lambda d: model(spec, d, ev), dop, target)
    ig_env = integrated_gradients(lambda e: model(spec, dop, e), ev, target)

    raw = {
        "rd_map": float(ig_spec.abs().sum().item()),
        "doppler": float(ig_dop.abs().sum().item()),
        "env": float(ig_env.abs().sum().item()),
    }
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def shap_summary(predict_fn, background: np.ndarray, x: np.ndarray):
    """
    Optional SHAP KernelExplainer over a flat feature-summary vector.
    Raises ImportError if SHAP is not installed (use the built-in methods instead).
    """
    try:
        import shap  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "shap is not installed; use integrated_gradients / occlusion_importance "
            "or `pip install shap`."
        ) from e
    explainer = shap.KernelExplainer(predict_fn, background)
    return explainer.shap_values(x)
