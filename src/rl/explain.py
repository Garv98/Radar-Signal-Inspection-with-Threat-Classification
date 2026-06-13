"""
explain.py  (RL gatekeeper explainability)
==========================================
Explains *why* the gatekeeper chose DISCARD or FORWARD for a given signal.

Two complementary views:

1. State attribution -- Integrated Gradients of the advantage gap
   ``A(FORWARD) - A(DISCARD)`` with respect to the input feature tensor X.
   A positive RD-map saliency region means "this part of the Range-Doppler map
   pushed the agent toward FORWARD".

2. Action rationale -- a structured, human-readable summary combining the
   agent's Q-values (decision margin) with branch-level importance (RD map vs
   Doppler vs env) so an operator can see what drove the gate.

Pure PyTorch; reuses :func:`src.explain.attribution.integrated_gradients`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from src.explain.attribution import integrated_gradients
from src.rl.reward import ACTION_DISCARD, ACTION_FORWARD

_ACTION_NAME = {ACTION_DISCARD: "DISCARD", ACTION_FORWARD: "FORWARD"}


def _decision_score(qnet, rd, dp, ev) -> torch.Tensor:
    """Scalar 'forwarding preference' = Q(FORWARD) - Q(DISCARD), per sample."""
    q = qnet(rd, dp, ev)
    return q[:, ACTION_FORWARD] - q[:, ACTION_DISCARD]


def gatekeeper_attribution(
    qnet: torch.nn.Module,
    rd_map: np.ndarray,
    doppler: np.ndarray,
    env: np.ndarray,
    device: str = "cpu",
) -> Dict[str, object]:
    """
    Attribute the FORWARD-vs-DISCARD decision to the input feature tensor.

    Returns a dict with:
      action, q_values, decision_margin,
      rd_saliency [D, R], doppler_saliency [D], env_saliency [3],
      branch_importance {rd_map, doppler, env}  (normalised to sum 1).
    """
    dev = torch.device(device)
    qnet = qnet.to(dev).eval()

    rd = torch.from_numpy(np.asarray(rd_map, np.float32)).unsqueeze(0).to(dev)   # [1,D,R]
    dp = torch.from_numpy(np.asarray(doppler, np.float32)).unsqueeze(0).to(dev)  # [1,D]
    ev = torch.from_numpy(np.asarray(env, np.float32)).unsqueeze(0).to(dev)      # [1,3]

    with torch.no_grad():
        q = qnet(rd, dp, ev).squeeze(0).cpu().numpy()
    action = int(np.argmax(q))
    margin = float(q[ACTION_FORWARD] - q[ACTION_DISCARD])

    # Integrated gradients of the decision score wrt each branch.
    ig_rd = integrated_gradients(lambda t: _decision_score(qnet, t, dp, ev).unsqueeze(1),
                                 rd, target=0)
    ig_dp = integrated_gradients(lambda t: _decision_score(qnet, rd, t, ev).unsqueeze(1),
                                 dp, target=0)
    ig_ev = integrated_gradients(lambda t: _decision_score(qnet, rd, dp, t).unsqueeze(1),
                                 ev, target=0)

    rd_sal = ig_rd.squeeze(0).detach().cpu().numpy()
    dp_sal = ig_dp.squeeze(0).detach().cpu().numpy()
    ev_sal = ig_ev.squeeze(0).detach().cpu().numpy()

    raw = {
        "rd_map": float(np.abs(rd_sal).sum()),
        "doppler": float(np.abs(dp_sal).sum()),
        "env": float(np.abs(ev_sal).sum()),
    }
    total = sum(raw.values()) or 1.0
    branch_importance = {k: v / total for k, v in raw.items()}

    return {
        "action": _ACTION_NAME[action],
        "action_idx": action,
        "q_values": {"DISCARD": float(q[ACTION_DISCARD]), "FORWARD": float(q[ACTION_FORWARD])},
        "decision_margin": margin,
        "rd_saliency": rd_sal,
        "doppler_saliency": dp_sal,
        "env_saliency": ev_sal,
        "branch_importance": branch_importance,
    }


def action_rationale(attr: Dict[str, object]) -> str:
    """Human-readable one-paragraph rationale from a gatekeeper_attribution dict."""
    a = attr["action"]
    margin = attr["decision_margin"]
    bi = attr["branch_importance"]
    dom = max(bi, key=bi.get)
    conf = "decisively" if abs(margin) > 1.0 else "marginally"
    driver = {"rd_map": "the Range-Doppler structure",
              "doppler": "the Doppler profile",
              "env": "the environmental context"}[dom]
    verb = ("forwarded for deep ML inspection" if a == "FORWARD"
            else "discarded without ML inspection")
    return (f"The gatekeeper {conf} {verb} "
            f"(margin Q_forward-Q_discard = {margin:+.2f}), driven mainly by "
            f"{driver} ({bi[dom]*100:.0f}% of attribution).")
