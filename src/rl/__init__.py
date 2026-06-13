"""RL gatekeeper package: shared-tensor environment, encoder, agent, reward."""

from .radar_env import GatekeeperEnv
from .dqn_agent import (
    GatekeeperAgent, DQNAgent, DuelingDQN, PrioritizedReplayBuffer,
)
from .encoder import FeatureEncoder, GatekeeperQNet, count_parameters
from .reward import (
    RewardFunction, RewardConfig, derive_class_values,
    ACTION_DISCARD, ACTION_FORWARD,
)
from .data_source import (
    RadarSignalSource, Signal, CLASS_NAMES, CLASS_TO_IDX,
    load_zenodo_signals, generate_synthetic_signals,
)

__all__ = [
    "GatekeeperEnv",
    "GatekeeperAgent", "DQNAgent", "DuelingDQN", "PrioritizedReplayBuffer",
    "FeatureEncoder", "GatekeeperQNet", "count_parameters",
    "RewardFunction", "RewardConfig", "derive_class_values",
    "ACTION_DISCARD", "ACTION_FORWARD",
    "RadarSignalSource", "Signal", "CLASS_NAMES", "CLASS_TO_IDX",
    "load_zenodo_signals", "generate_synthetic_signals",
]
