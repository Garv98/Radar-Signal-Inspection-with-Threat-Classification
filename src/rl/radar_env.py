"""
radar_env.py
============
Gatekeeper environment (Gymnasium-compatible).

Data flow enforced by this environment::

    Raw IQ -> preprocessing -> feature tensor X = (rd_map, doppler, env)
        X -> RL gatekeeper -> action {0 = DISCARD, 1 = FORWARD}
              FORWARD -> X -> ML classifier  (the expensive deep inspection)
              DISCARD -> stop                (ML inference skipped)

Crucially the agent's **observation is X itself** -- the very same tensor the
ML classifier consumes -- and never the ML output.  That is what makes the
agent a genuine *gatekeeper*: at deployment the ML model is only invoked on
FORWARD decisions, so DISCARD decisions save compute.

The ML classifier acts as the **teacher**: it is run here to score the reward
(Value-of-Information, see :mod:`src.rl.reward`) and to provide oracle
demonstrations during curriculum warm-up.  During training the ML model is
evaluated every step so the reward is defined for both actions; the number of
*policy-induced* ML invocations (FORWARD count) is tracked separately and is
the quantity that deployment actually pays.

State (per signal):
    rd_map  : float32 [D, R]   normalised Range-Doppler map
    doppler : float32 [D]      Doppler profile
    env     : float32 [3]      normalised environmental vector

Action space: Discrete(2)  ->  0 DISCARD, 1 FORWARD
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from src.rl.reward import RewardFunction, ACTION_DISCARD, ACTION_FORWARD
from src.rl.data_source import RadarSignalSource, Signal, CLASS_NAMES


class GatekeeperEnv(gym.Env):
    """RL environment for the computational gatekeeper."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        ml_classifier: torch.nn.Module,
        data_source: RadarSignalSource,
        reward_fn: RewardFunction,
        max_steps: int = 400,
        device: str = "cpu",
        balanced_sampling: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if ml_classifier is None or data_source is None or reward_fn is None:
            raise ValueError("GatekeeperEnv requires ml_classifier, data_source and reward_fn")

        self.device = torch.device(device)
        self.ml = ml_classifier.to(self.device).eval()
        self.source = data_source
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.balanced = balanced_sampling
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            self.source.reseed(seed)

        D = self.source.doppler_fft
        R = self.source.range_fft
        self.observation_space = spaces.Tuple((
            spaces.Box(0.0, 1.0, shape=(D, R), dtype=np.float32),   # rd_map
            spaces.Box(0.0, 1.0, shape=(D,),  dtype=np.float32),    # doppler
            spaces.Box(0.0, 1.0, shape=(3,),  dtype=np.float32),    # env
        ))
        self.action_space = spaces.Discrete(2)

        self._step = 0
        self._signal: Optional[Signal] = None
        self._probs: Optional[np.ndarray] = None
        self._stats = self._blank_stats()

    # ── ML teacher inference ────────────────────────────────────────────────
    @torch.no_grad()
    def _ml_probs(self, sig: Signal) -> np.ndarray:
        spec = torch.from_numpy(sig.rd_map).unsqueeze(0).unsqueeze(0).to(self.device)
        dop = torch.from_numpy(sig.doppler).unsqueeze(0).to(self.device)
        env = torch.from_numpy(sig.env).unsqueeze(0).to(self.device)
        logits = self.ml(spec, dop, env)
        return F.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    # ── observation helpers ─────────────────────────────────────────────────
    @staticmethod
    def _obs(sig: Signal) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return sig.rd_map, sig.doppler, sig.env

    def _draw(self):
        self._signal = self.source.sample(balanced=self.balanced)
        # The teacher (ML) scores the reward.  Because the teacher is fixed and
        # the signal pool is reused every episode, its output for a given signal
        # never changes -> compute once and cache on the Signal (exact, not an
        # approximation).  This removes the per-step ML forward pass that
        # otherwise dominates training time.  Deployment still only pays ML on
        # FORWARD (tracked separately via ml_invocations).
        if self._signal.probs is None:
            self._signal.probs = self._ml_probs(self._signal)
        self._probs = self._signal.probs
        return self._obs(self._signal)

    def teacher_action(self) -> int:
        """Oracle demonstration action for the current signal."""
        u, _ = self.reward_fn.ml_utility(self._probs, self._signal.label)
        return self.reward_fn.teacher_action(u)

    # ── Gym API ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._stats = self._blank_stats()
        obs = self._draw()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"invalid action {action}"
        sig = self._signal
        forwards = self._stats["forward"]
        forward_rate = forwards / max(self._step, 1)

        reward, comp = self.reward_fn.reward(
            action, self._probs, sig.label, forward_rate=forward_rate
        )

        # ── bookkeeping ─────────────────────────────────────────────────────
        self._step += 1
        self._stats["total"] += 1
        if action == ACTION_FORWARD:
            self._stats["forward"] += 1            # one real ML invocation
            self._stats["ml_invocations"] += 1
        else:
            self._stats["discard"] += 1

        # Confusion vs. the "should we have forwarded?" oracle (teacher_action).
        oracle = self.reward_fn.teacher_action(comp["u_ml"])
        if action == ACTION_FORWARD and oracle == ACTION_FORWARD:
            self._stats["tp"] += 1
        elif action == ACTION_DISCARD and oracle == ACTION_DISCARD:
            self._stats["tn"] += 1
        elif action == ACTION_FORWARD and oracle == ACTION_DISCARD:
            self._stats["fp"] += 1
        else:
            self._stats["fn"] += 1
        self._stats["reward_sum"] += reward

        terminated = False
        truncated = self._step >= self.max_steps
        next_obs = self._draw()

        info = {
            "signal_class": CLASS_NAMES[sig.label],
            "true_label": sig.label,
            "source": sig.source,
            "action": "forward" if action == ACTION_FORWARD else "discard",
            "ml_pred": CLASS_NAMES[comp["pred"]],
            "ml_confidence": comp["confidence"],
            "ml_entropy": comp["entropy"],
            "ml_correct": comp["correct"],
            "u_ml": comp["u_ml"],
            "oracle_action": "forward" if oracle == ACTION_FORWARD else "discard",
            "reward": reward,
            "forward_rate": (self._stats["forward"] / self._stats["total"]),
            "stats": dict(self._stats),
        }
        return next_obs, reward, terminated, truncated, info

    # ── stats / render ──────────────────────────────────────────────────────
    @staticmethod
    def _blank_stats() -> Dict:
        return {"total": 0, "forward": 0, "discard": 0, "ml_invocations": 0,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0, "reward_sum": 0.0}

    def episode_metrics(self) -> Dict:
        s = self._stats
        total = max(s["total"], 1)
        tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]
        return {
            "forward_rate": s["forward"] / total,
            "discard_rate": s["discard"] / total,
            "workload_reduction": s["discard"] / total,   # fraction of ML calls avoided
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "accuracy": (tp + tn) / total,
            "miss_rate": fn / max(tp + fn, 1),
            "avg_reward": s["reward_sum"] / total,
        }

    def render(self, mode: str = "human") -> None:
        m = self.episode_metrics()
        print(f"[step {self._step:04d}] fwd={m['forward_rate']:.2f} "
              f"save={m['workload_reduction']:.2f} f1={m['f1']:.2f} "
              f"miss={m['miss_rate']:.2f} R={m['avg_reward']:.2f}")

    def close(self) -> None:
        pass
