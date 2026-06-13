"""
dqn_agent.py
============
Deep Q-Network (DQN) agent for radar signal prioritization.

Architecture:
  - Dueling DQN (separate value + advantage streams)
  - Prioritized Experience Replay (PER)
  - Double DQN target update
  - Epsilon-greedy exploration with decay schedule
  - Gradient clipping for training stability
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from typing import Dict, List, Tuple, Optional

from src.rl.encoder import GatekeeperQNet, count_parameters

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DUELING DQN
# ─────────────────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
    Improves stability when many actions share similar Q-values.
    """

    def __init__(
        self,
        n_features: int,
        n_actions:  int,
        hidden_dims: List[int] = [256, 256, 128],
    ):
        super().__init__()

        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PRIORITIZED EXPERIENCE REPLAY
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Proportional PER: P(i) proportional to |TD_error_i|^alpha.
    Importance-sampling weights correct the bias from non-uniform sampling.
    """

    def __init__(
        self,
        capacity:    int   = 20_000,
        alpha:       float = 0.6,
        beta_start:  float = 0.4,
        beta_frames: int   = 100_000,
        epsilon:     float = 1e-5,
    ):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.epsilon     = epsilon
        self.frame       = 1

        self.buffer    : List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *args) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        n     = len(self.buffer)
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.frame += 1
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(float(err)) + self.epsilon

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DQN AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double Dueling DQN + Prioritized Experience Replay.

    The agent operates on the 10-dim state derived from the ML classifier.
    During curriculum warm-up, the replay buffer is pre-filled with
    teacher (ML) demonstrations before the agent starts self-training.
    """

    def __init__(
        self,
        n_features:      int,
        n_actions:       int,
        lr:              float = 1e-3,
        gamma:           float = 0.99,
        eps_start:       float = 1.0,
        eps_end:         float = 0.05,
        eps_decay:       float = 0.995,
        batch_size:      int   = 64,
        target_update:   int   = 200,
        tau:             float = 1.0,
        grad_clip:       float = 10.0,
        buffer_capacity: int   = 20_000,
        device: Optional[str]  = None,
    ):
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.eps          = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.batch_size   = batch_size
        self.target_update= target_update
        self.tau          = tau
        self.grad_clip    = grad_clip
        self.steps_done   = 0

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[DQNAgent] Device: {self.device}")

        self.policy_net = DuelingDQN(n_features, n_actions).to(self.device)
        self.target_net = DuelingDQN(n_features, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        self.memory    = PrioritizedReplayBuffer(capacity=buffer_capacity)

        self.loss_history:   List[float] = []
        self.reward_history: List[float] = []

    # ── Action selection ────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(s).argmax(dim=1).item())

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    # ── Learning step ───────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        samples, indices, weights = self.memory.sample(self.batch_size)
        weights = weights.to(self.device)

        states      = torch.FloatTensor(np.array([t.state      for t in samples])).to(self.device)
        actions     = torch.LongTensor( np.array([t.action     for t in samples])).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(np.array([t.reward     for t in samples])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in samples])).to(self.device)
        dones       = torch.FloatTensor(np.array([t.done       for t in samples], dtype=np.float32)).unsqueeze(1).to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_actions  = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q        = self.target_net(next_states).gather(1, next_actions)
            target_q      = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.policy_net(states).gather(1, actions)

        td_errors  = (current_q - target_q).detach().cpu().numpy().squeeze()
        loss_elem  = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss       = (weights.unsqueeze(1) * loss_elem).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        # Epsilon decay
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.steps_done += 1

        # Target network sync
        if self.steps_done % self.target_update == 0:
            if self.tau == 1.0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    # ── Metrics ─────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        return {
            "epsilon":    round(self.eps, 4),
            "steps_done": self.steps_done,
            "buffer_len": len(self.memory),
            "mean_loss_last100": float(np.mean(self.loss_history[-100:])) if self.loss_history else 0.0,
        }

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "epsilon":           self.eps,
            "steps_done":        self.steps_done,
            "loss_history":      self.loss_history,
            "reward_history":    self.reward_history,
        }, path)
        print(f"[DQNAgent] Saved -> {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.eps         = ckpt["epsilon"]
        self.steps_done  = ckpt["steps_done"]
        self.loss_history    = ckpt.get("loss_history", [])
        self.reward_history  = ckpt.get("reward_history", [])
        print(f"[DQNAgent] Loaded <- {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GATEKEEPER AGENT  (operates on the shared feature tensor X)
# ─────────────────────────────────────────────────────────────────────────────

State = Tuple[np.ndarray, np.ndarray, np.ndarray]   # (rd_map, doppler, env)


def _stack_states(states: List[State], device: torch.device):
    rd = torch.from_numpy(np.stack([s[0] for s in states])).float().to(device)
    dp = torch.from_numpy(np.stack([s[1] for s in states])).float().to(device)
    ev = torch.from_numpy(np.stack([s[2] for s in states])).float().to(device)
    return rd, dp, ev


class GatekeeperAgent:
    """
    Double Dueling DQN + Prioritized Experience Replay for the gatekeeper.

    Unlike :class:`DQNAgent` (flat vector states) this agent consumes the same
    structured feature tensor the ML model sees -- ``(rd_map, doppler, env)`` --
    via a lightweight :class:`GatekeeperQNet` encoder.  With ``gamma = 0`` the
    problem is an i.i.d. contextual bandit (each signal is independent), which
    is the correct model for per-signal gatekeeping and removes bootstrapping
    instability; ``gamma > 0`` is still supported for experimentation.
    """

    def __init__(
        self,
        doppler_dim: int,
        env_dim: int = 3,
        n_actions: int = 2,
        encoder_cfg: Optional[dict] = None,
        head_hidden: Tuple[int, ...] = (128, 64),
        stream_hidden: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.0,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_decay: float = 0.997,
        batch_size: int = 64,
        target_update: int = 250,
        tau: float = 1.0,
        grad_clip: float = 10.0,
        buffer_capacity: int = 20_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 80_000,
        device: Optional[str] = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau
        self.grad_clip = grad_clip
        self.steps_done = 0

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._net_kwargs = dict(
            doppler_dim=doppler_dim, env_dim=env_dim, n_actions=n_actions,
            encoder_cfg=encoder_cfg, head_hidden=list(head_hidden),
            stream_hidden=stream_hidden,
        )
        self.policy_net = GatekeeperQNet(**self._net_kwargs).to(self.device)
        self.target_net = GatekeeperQNet(**self._net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1e-5)
        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_capacity, alpha=per_alpha,
            beta_start=per_beta_start, beta_frames=per_beta_frames,
        )

        self.loss_history: List[float] = []
        self.reward_history: List[float] = []
        print(f"[GatekeeperAgent] device={self.device}  "
              f"params={count_parameters(self.policy_net):,}  gamma={gamma}")

    # ── Action selection ────────────────────────────────────────────────────
    def select_action(self, state: State, training: bool = True) -> int:
        if training and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            rd, dp, ev = _stack_states([state], self.device)
            return int(self.policy_net(rd, dp, ev).argmax(dim=1).item())

    def q_values(self, state: State) -> np.ndarray:
        with torch.no_grad():
            rd, dp, ev = _stack_states([state], self.device)
            return self.policy_net(rd, dp, ev).squeeze(0).cpu().numpy()

    def store(self, state: State, action: int, reward: float,
              next_state: State, done: bool) -> None:
        self.memory.push(state, action, reward, next_state, done)

    # ── Learning step ───────────────────────────────────────────────────────
    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        samples, indices, weights = self.memory.sample(self.batch_size)
        weights = weights.to(self.device)

        states = _stack_states([t.state for t in samples], self.device)
        next_states = _stack_states([t.next_state for t in samples], self.device)
        actions = torch.LongTensor([t.action for t in samples]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in samples]).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in samples]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            if self.gamma > 0.0:
                next_actions = self.policy_net(*next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(*next_states).gather(1, next_actions)
                target_q = rewards + self.gamma * next_q * (1.0 - dones)
            else:
                target_q = rewards  # bandit: reward is the full return

        current_q = self.policy_net(*states).gather(1, actions)

        td_errors = (current_q - target_q).detach().cpu().numpy().squeeze()
        loss_elem = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights.unsqueeze(1) * loss_elem).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.memory.update_priorities(indices, np.atleast_1d(td_errors))

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            if self.tau >= 1.0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        return loss_val

    def get_metrics(self) -> dict:
        return {
            "epsilon": round(self.eps, 4),
            "steps_done": self.steps_done,
            "buffer_len": len(self.memory),
            "mean_loss_last100": float(np.mean(self.loss_history[-100:]))
            if self.loss_history else 0.0,
        }

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "net_kwargs": self._net_kwargs,
            "epsilon": self.eps,
            "steps_done": self.steps_done,
            "gamma": self.gamma,
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[GatekeeperAgent] saved -> {path}")

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.eps = ckpt.get("epsilon", self.eps_end)
        self.steps_done = ckpt.get("steps_done", 0)
        self.loss_history = ckpt.get("loss_history", [])
        self.reward_history = ckpt.get("reward_history", [])
        print(f"[GatekeeperAgent] loaded <- {path}")
        return ckpt

    @staticmethod
    def from_checkpoint(path: str, device: Optional[str] = None) -> "GatekeeperAgent":
        """Reconstruct an agent (architecture + weights) from a checkpoint."""
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        nk = ckpt["net_kwargs"]
        agent = GatekeeperAgent(
            doppler_dim=nk["doppler_dim"], env_dim=nk["env_dim"],
            n_actions=nk["n_actions"], encoder_cfg=nk["encoder_cfg"],
            head_hidden=tuple(nk["head_hidden"]), stream_hidden=nk["stream_hidden"],
            device=str(dev),
        )
        agent.load(path)
        agent.eps = agent.eps_end
        return agent
