"""
CNN-LSTM Hybrid Model for Radar Threat Classification

Architecture:
  • CNN branch   — Range-Doppler map [1×32×128]
                   3× (Conv2d → BN → GELU → MaxPool) with SE-attention
  • LSTM branch  — Doppler profile [32] processed as sequence
                   Bidirectional 2-layer LSTM
  • Env branch   — Environmental features [3] → MLP
  • Classifier   — Concatenated features → residual FC head → softmax

Key improvements over baseline:
  • Squeeze-and-Excitation (SE) channel attention in each CNN block
  • Bidirectional LSTM captures both forward and backward spectral context
  • GELU activations for smoother gradients
  • Residual skip in classifier head
  • Label-smoothing-compatible architecture (logits only, no softmax here)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Channel attention: scale feature maps by learned channel importances."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ConvBlock(nn.Module):
    """Conv2d → BN → GELU → SE-attention → MaxPool."""

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        kernel:  int = 3,
        pool:    int = 2,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()
        self.se   = SqueezeExcitation(out_ch, se_reduction)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        x = self.se(x)
        x = self.pool(x)
        return x


class CNNBranch(nn.Module):
    """
    CNN branch for Range-Doppler spectrogram processing.

    Each block doubles the number of channels and halves spatial size.
    A final AdaptiveAvgPool2d(4, 4) ensures fixed output regardless of
    input spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        layers = []
        prev = in_channels
        for out_ch in channels:
            layers.append(ConvBlock(prev, out_ch, kernel_size, pool_size))
            prev = out_ch
        self.blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.output_size = channels[-1] * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.global_pool(x)
        return x.flatten(1)


class LSTMBranch(nn.Module):
    """
    Bidirectional LSTM branch for Doppler sequence processing.

    Treats the 1-D Doppler profile as a temporal sequence of scalar
    observations so the LSTM learns to capture spectral shapes (e.g.
    the pattern of JEM harmonics, rotor sidebands, wing-beat modulation).
    """

    def __init__(
        self,
        input_size:  int  = 1,
        hidden_size: int  = 64,
        num_layers:  int  = 2,
        bidirectional: bool = True,
        dropout:     float = 0.20,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        self.output_size   = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len] or [batch, seq_len, features]"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)                       # [B, T, 1]
        _, (h, _) = self.lstm(x)
        if self.bidirectional:
            # Concatenate last layer's forward and backward hidden states
            h_out = torch.cat([h[-2], h[-1]], dim=1)  # [B, 2*H]
        else:
            h_out = h[-1]                              # [B, H]
        return h_out


class EnvironmentalBranch(nn.Module):
    """Small MLP for rain / temperature / pressure features."""

    def __init__(self, input_size: int = 3, hidden_size: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main classifier
# ──────────────────────────────────────────────────────────────────────────────

class RadarClassifier(nn.Module):
    """
    Multi-modal radar threat classifier.

    Inputs:
        spectrogram  [B, 1, D, R]  — Range-Doppler map
        doppler_seq  [B, D]         — 1-D Doppler profile
        env_features [B, 3]         — Environmental scalars

    Output:
        logits       [B, num_classes]   (use CrossEntropy, NOT softmax here)
    """

    def __init__(
        self,
        num_classes:    int   = 5,
        cnn_channels:   List[int] = [32, 64, 128],
        lstm_hidden_size: int = 64,
        lstm_num_layers:  int = 2,
        lstm_bidirectional: bool = True,
        env_feature_dim:  int = 3,
        env_hidden_size:  int = 16,
        fc_hidden:        int = 256,
        dropout:          float = 0.35,
    ):
        super().__init__()

        self.cnn_branch = CNNBranch(in_channels=1, channels=cnn_channels)
        self.lstm_branch = LSTMBranch(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
        )
        self.env_branch = EnvironmentalBranch(env_feature_dim, env_hidden_size)

        combined = (
            self.cnn_branch.output_size +
            self.lstm_branch.output_size +
            self.env_branch.output_size
        )

        # Classifier head with residual skip
        self.fc1   = nn.Linear(combined, fc_hidden)
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(fc_hidden, fc_hidden // 2)
        self.drop2 = nn.Dropout(dropout)
        self.out   = nn.Linear(fc_hidden // 2, num_classes)

        # Residual projection (combined → fc_hidden//2)
        self.skip_proj = nn.Linear(combined, fc_hidden // 2, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)

    def forward(
        self,
        spectrogram:  torch.Tensor,
        doppler_seq:  torch.Tensor,
        env_features: torch.Tensor,
    ) -> torch.Tensor:
        cnn_out  = self.cnn_branch(spectrogram)     # [B, cnn_size]
        lstm_out = self.lstm_branch(doppler_seq)    # [B, lstm_size]
        env_out  = self.env_branch(env_features)    # [B, env_size]

        combined = torch.cat([cnn_out, lstm_out, env_out], dim=1)

        # Residual FC head
        h = self.drop1(F.gelu(self.fc1(combined)))  # [B, fc_hidden]
        h = self.drop2(F.gelu(self.fc2(h)))          # [B, fc_hidden//2]
        h = h + self.skip_proj(combined)             # residual
        return self.out(h)                           # [B, num_classes]

    def get_feature_vector(
        self,
        spectrogram:  torch.Tensor,
        doppler_seq:  torch.Tensor,
        env_features: torch.Tensor,
    ) -> torch.Tensor:
        """Combined feature vector before the classifier head."""
        return torch.cat([
            self.cnn_branch(spectrogram),
            self.lstm_branch(doppler_seq),
            self.env_branch(env_features),
        ], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline (CNN-only, for ablation)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleCNNClassifier(nn.Module):
    """Spectrogram-only baseline for ablation studies."""

    def __init__(
        self,
        num_classes: int = 5,
        channels: List[int] = [32, 64, 128],
        fc_hidden: int = 256,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.cnn = CNNBranch(channels=channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn.output_size, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cnn(spectrogram))


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> RadarClassifier:
    """Instantiate RadarClassifier from a config dict."""
    mc   = config.get('model', {})
    cnn  = mc.get('cnn',  {})
    lstm = mc.get('lstm', {})

    return RadarClassifier(
        num_classes=config.get('dataset', {}).get('num_classes', 5),
        cnn_channels=cnn.get('channels', [32, 64, 128]),
        lstm_hidden_size=lstm.get('hidden_size', 64),
        lstm_num_layers=lstm.get('num_layers', 2),
        lstm_bidirectional=lstm.get('bidirectional', True),
        env_feature_dim=mc.get('env_feature_dim', 3),
        env_hidden_size=mc.get('env_hidden_size', 16),
        fc_hidden=mc.get('fc_hidden', 256),
        dropout=mc.get('dropout', 0.35),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = RadarClassifier()
    print(f"Parameters: {count_parameters(model):,}")

    B = 4
    spec  = torch.randn(B, 1, 32, 128)
    dop   = torch.randn(B, 32)
    env   = torch.randn(B, 3)
    logits = model(spec, dop, env)
    print(f"Input:  spec={spec.shape}  dop={dop.shape}  env={env.shape}")
    print(f"Output: {logits.shape}")
    feats = model.get_feature_vector(spec, dop, env)
    print(f"Feature vector: {feats.shape}")
