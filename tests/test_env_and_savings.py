"""GatekeeperEnv data-flow + the compute-savings accounting that defines the project."""

import numpy as np

from src.rl.data_source import RadarSignalSource, CLASS_NAMES, Signal
from src.rl.radar_env import GatekeeperEnv
from src.rl.reward import ACTION_DISCARD, ACTION_FORWARD


def test_data_source_pool(synth_source):
    assert len(synth_source) > 0
    counts = synth_source.label_counts()
    assert counts.sum() == len(synth_source)
    s = synth_source.sample()
    assert isinstance(s, Signal)
    assert s.rd_map.shape == (32, 128)
    assert s.doppler.shape == (32,)
    assert s.env.shape == (3,)


def test_env_obs_matches_space(ml_model, synth_source, reward_fn):
    env = GatekeeperEnv(ml_model, synth_source, reward_fn, max_steps=10, device="cpu")
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    nobs, r, term, trunc, info = env.step(ACTION_FORWARD)
    assert env.observation_space.contains(nobs)
    assert np.isfinite(r)
    assert {"ml_pred", "ml_confidence", "u_ml", "forward_rate"} <= set(info)


def test_discard_skips_ml_invocation(ml_model, synth_source, reward_fn):
    """The core savings property: ML invocations == number of FORWARD actions."""
    env = GatekeeperEnv(ml_model, synth_source, reward_fn, max_steps=50, device="cpu")
    env.reset()
    forwards = 0
    for i in range(50):
        action = ACTION_FORWARD if i % 3 == 0 else ACTION_DISCARD
        _, _, term, trunc, info = env.step(action)
        forwards += int(action == ACTION_FORWARD)
        if term or trunc:
            break
    assert env._stats["ml_invocations"] == forwards


def test_always_discard_maximizes_savings(ml_model, synth_source, reward_fn):
    env = GatekeeperEnv(ml_model, synth_source, reward_fn, max_steps=30, device="cpu")
    env.reset()
    for _ in range(30):
        _, _, term, trunc, _ = env.step(ACTION_DISCARD)
        if term or trunc:
            break
    m = env.episode_metrics()
    assert m["workload_reduction"] == 1.0     # zero ML calls
    assert m["forward_rate"] == 0.0


def test_always_forward_no_savings(ml_model, synth_source, reward_fn):
    env = GatekeeperEnv(ml_model, synth_source, reward_fn, max_steps=30, device="cpu")
    env.reset()
    for _ in range(30):
        _, _, term, trunc, _ = env.step(ACTION_FORWARD)
        if term or trunc:
            break
    m = env.episode_metrics()
    assert m["workload_reduction"] == 0.0
    assert m["forward_rate"] == 1.0
