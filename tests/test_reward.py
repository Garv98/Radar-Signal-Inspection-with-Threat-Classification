"""Value-of-Information reward: properties that must hold by construction."""

import numpy as np
import pytest

from src.rl.reward import (
    RewardConfig, RewardFunction, derive_class_values, entropy_normalized,
    ACTION_DISCARD, ACTION_FORWARD,
)


def test_class_values_inverse_frequency():
    # within the non-threat group, rarer class 0 outweighs common class 2
    counts = np.array([10, 0, 1000, 100, 50])
    v = derive_class_values(counts, gamma=0.5, value_clip=(0.01, 5.0),
                            threat_indices=None)
    assert v[0] > v[2]
    assert np.all(v > 0)


def test_threats_outvalue_nonthreats():
    counts = np.array([100, 100, 100, 100, 100])
    v = derive_class_values(counts, threat_indices=[0, 1],
                            threat_value=1.5, nonthreat_value=0.25)
    # threat classes valued well above non-threats; non-threats below kappa region
    assert min(v[0], v[1]) > max(v[2], v[3], v[4])
    assert max(v[2], v[3], v[4]) < 0.35   # below default compute cost


def test_entropy_bounds():
    uniform = np.ones(5) / 5
    onehot = np.array([1.0, 0, 0, 0, 0])
    assert entropy_normalized(uniform) == pytest.approx(1.0, abs=1e-6)
    assert entropy_normalized(onehot) == pytest.approx(0.0, abs=1e-6)


def _rf(values=None):
    cfg = RewardConfig(compute_cost=0.35, miss_aversion=3.0)
    if values is None:
        # threat-centric values (classes 0,1 are threats), as in production
        values = derive_class_values(np.array([100, 100, 100, 100, 100]),
                                     threat_indices=[0, 1],
                                     threat_value=1.5, nonthreat_value=0.25)
    return RewardFunction(cfg, np.asarray(values, np.float32), forward_budget=None)


def test_utility_sign():
    rf = _rf()
    confident_correct = np.array([0.9, 0.04, 0.03, 0.02, 0.01], np.float32)
    confident_wrong = np.array([0.9, 0.04, 0.03, 0.02, 0.01], np.float32)
    u_correct, _ = rf.ml_utility(confident_correct, true_label=0)
    u_wrong, _ = rf.ml_utility(confident_wrong, true_label=2)
    assert u_correct > 0          # correct + confident -> positive
    assert u_wrong < 0            # wrong + confident -> negative
    # uncertain -> near zero
    uncertain = np.ones(5, np.float32) / 5
    u_unc, _ = rf.ml_utility(uncertain, true_label=0)
    assert abs(u_unc) < abs(u_correct)


def test_discard_penalizes_valuable_signal():
    rf = _rf()
    probs = np.array([0.92, 0.03, 0.02, 0.02, 0.01], np.float32)  # correct, confident
    r_fwd, _ = rf.reward(ACTION_FORWARD, probs, 0, forward_rate=0.0)
    r_dis, _ = rf.reward(ACTION_DISCARD, probs, 0, forward_rate=0.0)
    assert r_fwd > r_dis                 # forwarding a valuable signal beats discarding
    assert r_dis < 0                     # missing it is penalised


def test_forward_noise_is_wasteful():
    rf = _rf()
    # ML correctly, confidently dismisses obvious noise (class 4)
    probs = np.array([0.02, 0.02, 0.02, 0.04, 0.90], np.float32)
    r_fwd, _ = rf.reward(ACTION_FORWARD, probs, 4, forward_rate=0.0)
    r_dis, _ = rf.reward(ACTION_DISCARD, probs, 4, forward_rate=0.0)
    # discarding obvious noise should not be worse than forwarding it
    assert r_dis >= r_fwd - 1e-6


def test_teacher_action_threshold():
    rf = _rf()
    high = np.array([0.95, 0.02, 0.01, 0.01, 0.01], np.float32)
    low = np.ones(5, np.float32) / 5
    u_hi, _ = rf.ml_utility(high, 0)
    u_lo, _ = rf.ml_utility(low, 0)
    assert rf.teacher_action(u_hi) == ACTION_FORWARD
    assert rf.teacher_action(u_lo) == ACTION_DISCARD
