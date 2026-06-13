"""Shared pytest fixtures for the radar inspection framework."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo root is importable (so `import src...` works under pytest).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn_lstm import build_model
from src.rl.data_source import RadarSignalSource, CLASS_NAMES
from src.rl.reward import RewardConfig, RewardFunction, derive_class_values


@pytest.fixture(scope="session")
def ml_model():
    """An untrained classifier of the correct architecture (shape/logic tests)."""
    return build_model({"dataset": {"num_classes": len(CLASS_NAMES)}}).eval()


@pytest.fixture(scope="session")
def synth_source():
    """Small synthetic-only signal pool (no Zenodo load)."""
    return RadarSignalSource(mode="synthetic", synthetic_per_class=8,
                             range_fft=128, doppler_fft=32, seed=0, verbose=False)


@pytest.fixture
def reward_fn(synth_source):
    cfg = RewardConfig()
    cv = derive_class_values(synth_source.label_counts(), gamma=cfg.value_gamma,
                             value_clip=cfg.value_clip)
    return RewardFunction(cfg, cv, forward_budget=0.4)
