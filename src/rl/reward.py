"""
reward.py
=========
Value-of-Information (VoI) reward for the RL gatekeeper.

Design goal
-----------
The gatekeeper decides, from the feature tensor X alone, whether to pay the
cost of a deep ML inspection (FORWARD) or to drop the signal (DISCARD).  A
principled objective is the *value of information*: forward a signal only when
the expected decision value of the ML readout exceeds the cost of obtaining it.

Notation (per signal with true label ``y`` and ML class posterior ``p``)
------------------------------------------------------------------------
    c          = max_k p_k                         (confidence / top-1 prob)
    margin     = p_(1) - p_(2)                     (decisiveness)
    H_norm     = -sum_k p_k log p_k / log K        (normalised entropy, 0..1)
    correct    = 1[argmax p == y]                  (0 or 1)
    certainty  = alpha * c + (1 - alpha) * (1 - H_norm)
    v[y]       = data-derived class value (rare / threat classes weigh more)

ML decision utility (the benefit of having run the classifier on this signal):

    U_ml(X) = v[y] * (2*correct - 1) * certainty

    * correct & certain on a valuable class  ->  large positive
    * wrong   & certain                       ->  large negative (ML misleads)
    * uncertain (high entropy / low conf)     ->  near zero (ML adds little)

Reward
------
    R(FORWARD) = U_ml(X) - kappa - eta * max(0, rho - rho*)
    R(DISCARD) = - beta * max(0, U_ml(X))

where
    kappa  = compute cost of one ML inference (utility units),
    beta   = miss-aversion (penalty multiplier for discarding a signal whose
             inspection would have been valuable -- the false-negative term),
    rho    = running forward rate, rho* = target forward budget,
    eta    = budget-pressure coefficient (prevents "forward everything").

Optimality / justification
--------------------------
With gamma = 0 (each signal is an independent contextual-bandit decision) the
Q-values the agent learns are exactly the conditional expectations

    Q(X, FORWARD) = E[ U_ml - kappa - budget | X ]
    Q(X, DISCARD) = E[ -beta * U_ml^+        | X ]

so the greedy policy forwards iff

    E[U_ml | X] + beta * E[U_ml^+ | X] > kappa + budget_pressure.

When U_ml is dominated by its positive part this reduces to the classic VoI
rule "acquire information iff expected value > cost", with the asymmetric
factor ``beta`` implementing a Neyman-Pearson-style preference that false
negatives on valuable signals are costlier than wasted compute.  Every
coefficient is configurable and ``v[y]`` is estimated from the data, so no
threshold or magnitude is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Class-value derivation (from the data, never hardcoded)
# ──────────────────────────────────────────────────────────────────────────────

def derive_class_values(
    label_counts: np.ndarray,
    gamma: float = 0.5,
    value_clip: Tuple[float, float] = (0.1, 3.0),
    threat_indices: Optional[List[int]] = None,
    threat_value: float = 1.5,
    nonthreat_value: float = 0.25,
) -> np.ndarray:
    """
    Threat-centric class value v[y], modulated by the data frequency.

    Operational stakes dominate: a (confirmable) threat detection is valuable
    (``threat_value``), while inspecting a non-threat adds little operational
    value (``nonthreat_value``, intentionally placed in the sub-cost region so
    correctly-dismissable background is discarded).  The inverse class frequency
    (raised to ``gamma``) gives rarer classes a modest extra weight -- but it is
    normalised *within each group* (threats / non-threats) so that frequency
    only re-weights members of a group and never pushes a (possibly majority)
    threat down into the non-threat value band.

        inv[y] = (mean_g n / n_y) ** gamma   normalised to mean 1 within group g
        base[y]= threat_value if y in threats else nonthreat_value
        v[y]   = clip( base[y] * inv[y] )

    Classes absent from the data keep their group base value (inv = 1).
    """
    counts = np.asarray(label_counts, dtype=np.float64)
    K = len(counts)
    threats = set(threat_indices or [])

    inv = np.ones(K, dtype=np.float64)
    # Inverse-frequency weight, normalised to mean 1 *within each group* so the
    # threat/non-threat value separation is preserved regardless of which class
    # happens to dominate the dataset.
    for group in (threats, set(range(K)) - threats):
        members = [y for y in group if counts[y] > 0]
        if len(members) < 2:
            continue  # 0 or 1 present member -> no within-group modulation
        c = counts[members]
        w = (c.mean() / c) ** gamma
        w = w / w.mean()
        for y, wy in zip(members, w):
            inv[y] = wy

    base = np.array([threat_value if y in threats else nonthreat_value
                     for y in range(K)], dtype=np.float64)

    v = base * inv
    lo, hi = value_clip
    return np.clip(v, lo, hi).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Reward configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    compute_cost: float = 0.35          # kappa
    miss_aversion: float = 3.0          # beta
    certainty_blend: float = 0.5        # alpha (confidence vs 1-entropy)
    value_gamma: float = 0.5
    value_clip: Tuple[float, float] = (0.1, 3.0)
    threat_classes: List[str] = field(default_factory=lambda: ["Drone", "Aircraft"])
    threat_value: float = 1.5
    nonthreat_value: float = 0.25
    forward_budget: Optional[float] = None   # rho*; None -> derive from data
    budget_penalty: float = 1.0              # eta

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RewardConfig":
        d = dict(d or {})
        vc = d.get("value_clip", (0.1, 3.0))
        return cls(
            compute_cost=float(d.get("compute_cost", 0.35)),
            miss_aversion=float(d.get("miss_aversion", 3.0)),
            certainty_blend=float(d.get("certainty_blend", 0.5)),
            value_gamma=float(d.get("value_gamma", 0.5)),
            value_clip=(float(vc[0]), float(vc[1])),
            threat_classes=list(d.get("threat_classes", ["Drone", "Aircraft"])),
            threat_value=float(d.get("threat_value", 1.5)),
            nonthreat_value=float(d.get("nonthreat_value", 0.25)),
            forward_budget=(None if d.get("forward_budget", None) is None
                            else float(d["forward_budget"])),
            budget_penalty=float(d.get("budget_penalty", 1.0)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Reward function
# ──────────────────────────────────────────────────────────────────────────────

ACTION_DISCARD = 0
ACTION_FORWARD = 1


def entropy_normalized(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p)))


class RewardFunction:
    """
    Computes the VoI reward and exposes the per-signal ML utility ``U_ml`` that
    drives the oracle teacher.  ``class_values`` are derived from the data pool.
    """

    def __init__(
        self,
        config: RewardConfig,
        class_values: np.ndarray,
        forward_budget: Optional[float] = None,
    ):
        self.cfg = config
        self.class_values = np.asarray(class_values, dtype=np.float32)
        # Effective forward budget: explicit config wins, else provided estimate.
        self.forward_budget = (
            config.forward_budget if config.forward_budget is not None
            else forward_budget
        )

    # ── core quantities ────────────────────────────────────────────────────
    def ml_utility(self, probs: np.ndarray, true_label: int) -> Tuple[float, Dict]:
        """U_ml(X) and the diagnostic components that compose it."""
        probs = np.asarray(probs, dtype=np.float64)
        conf = float(probs.max())
        pred = int(probs.argmax())
        correct = 1.0 if pred == int(true_label) else 0.0

        srt = np.sort(probs)[::-1]
        margin = float(srt[0] - srt[1]) if probs.size > 1 else float(srt[0])
        h_norm = entropy_normalized(probs)

        alpha = self.cfg.certainty_blend
        certainty = alpha * conf + (1.0 - alpha) * (1.0 - h_norm)

        v = float(self.class_values[int(true_label)])
        u = v * (2.0 * correct - 1.0) * certainty
        return u, {
            "confidence": conf,
            "margin": margin,
            "entropy": h_norm,
            "certainty": certainty,
            "correct": correct,
            "pred": pred,
            "class_value": v,
            "u_ml": u,
        }

    def teacher_action(self, u_ml: float) -> int:
        """
        Oracle teacher: the Bayes-greedy action under the reward below.  Since
        R(FORWARD)=U_surplus and R(DISCARD)=-beta*max(U_surplus,0) with
        U_surplus=U_ml-kappa, the greedy rule is exactly the value-of-information
        threshold "forward iff U_ml >= kappa" (no hardcoded probability cutoff).
        """
        return ACTION_FORWARD if (u_ml - self.cfg.compute_cost) >= 0.0 else ACTION_DISCARD

    def reward(
        self,
        action: int,
        probs: np.ndarray,
        true_label: int,
        forward_rate: float = 0.0,
    ) -> Tuple[float, Dict]:
        """Reward for taking ``action`` on a signal with ML posterior ``probs``."""
        u, comp = self.ml_utility(probs, true_label)
        surplus = u - self.cfg.compute_cost            # net value of inspecting

        budget = 0.0
        if self.forward_budget is not None:
            over = max(0.0, forward_rate - self.forward_budget)
            budget = self.cfg.budget_penalty * over

        if action == ACTION_FORWARD:
            r = surplus - budget
        else:
            # Penalise discarding only when the signal was worth forwarding
            # (surplus > 0), in proportion to how worth-forwarding it was.
            r = -self.cfg.miss_aversion * max(surplus, 0.0)

        comp.update({
            "action": int(action),
            "u_surplus": float(surplus),
            "budget_penalty": budget,
            "reward": float(r),
        })
        return float(r), comp
