"""
encoder.py
==========
Lightweight feature encoder + dueling Q-network for the RL gatekeeper.

The gatekeeper consumes the *same* feature tensor X as the ML classifier
(RD map + Doppler profile + environmental vector) but must be dramatically
cheaper to evaluate, otherwise running it would not save compute.  The encoder
is therefore a tiny CNN (default channels [8, 16] vs the classifier's
[32, 64, 128]) followed by a small fusion MLP -> shared latent -> dueling head.

    rd_map  [B,1,D,R] -> tiny CNN -> adaptive pool -> flatten ┐
    doppler [B,D]     ─┐                                       ├-> latent -> Q[2]
    env     [B,3]     ─┴-> side MLP ──────────────────────────┘

Action space: {0 = DISCARD, 1 = FORWARD}.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """Tiny CNN over the RD map fused with Doppler + env side features."""

    def __init__(
        self,
        doppler_dim: int,
        env_dim: int = 3,
        conv_channels: List[int] = (8, 16),
        kernel_size: int = 3,
        pool_size: int = 2,
        adaptive_pool: Tuple[int, int] = (2, 2),
        side_hidden: int = 16,
        latent_dim: int = 64,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(adaptive_pool)
        conv_out = conv_channels[-1] * adaptive_pool[0] * adaptive_pool[1]

        self.side = nn.Sequential(
            nn.Linear(doppler_dim + env_dim, side_hidden),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Linear(conv_out + side_hidden, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.latent_dim = latent_dim

    def forward(self, rd_map: torch.Tensor, doppler: torch.Tensor,
                env: torch.Tensor) -> torch.Tensor:
        if rd_map.dim() == 3:                       # [B, D, R] -> [B, 1, D, R]
            rd_map = rd_map.unsqueeze(1)
        c = self.gap(self.conv(rd_map)).flatten(1)  # [B, conv_out]
        s = self.side(torch.cat([doppler, env], dim=1))
        return self.fuse(torch.cat([c, s], dim=1))  # [B, latent_dim]


class GatekeeperQNet(nn.Module):
    """
    Encoder + Dueling DQN head:  Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).

    Accepts the raw feature tensor X as three tensors so the network truly
    consumes the same representation as the ML model.
    """

    def __init__(
        self,
        doppler_dim: int,
        env_dim: int = 3,
        n_actions: int = 2,
        encoder_cfg: Dict = None,
        head_hidden: List[int] = (128, 64),
        stream_hidden: int = 64,
    ):
        super().__init__()
        encoder_cfg = dict(encoder_cfg or {})
        self.encoder = FeatureEncoder(
            doppler_dim=doppler_dim,
            env_dim=env_dim,
            conv_channels=tuple(encoder_cfg.get("conv_channels", (8, 16))),
            kernel_size=encoder_cfg.get("kernel_size", 3),
            pool_size=encoder_cfg.get("pool_size", 2),
            adaptive_pool=tuple(encoder_cfg.get("adaptive_pool", (2, 2))),
            side_hidden=encoder_cfg.get("side_hidden", 16),
            latent_dim=encoder_cfg.get("latent_dim", 64),
        )

        dims: List[int] = []
        in_dim = self.encoder.latent_dim
        body: List[nn.Module] = []
        for h in head_hidden:
            body += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            in_dim = h
            dims.append(h)
        self.body = nn.Sequential(*body)

        self.value = nn.Sequential(
            nn.Linear(in_dim, stream_hidden), nn.ReLU(inplace=True),
            nn.Linear(stream_hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_dim, stream_hidden), nn.ReLU(inplace=True),
            nn.Linear(stream_hidden, n_actions),
        )
        self.n_actions = n_actions
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def latent(self, rd_map, doppler, env) -> torch.Tensor:
        return self.body(self.encoder(rd_map, doppler, env))

    def forward(self, rd_map, doppler, env) -> torch.Tensor:
        feat = self.latent(rd_map, doppler, env)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
