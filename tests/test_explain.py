"""Explainability: Grad-CAM, ML branch importance, gatekeeper attribution."""

import numpy as np

from src.explain import gradcam_on_rd_map, ml_feature_importance
from src.rl.encoder import GatekeeperQNet
from src.rl.explain import gatekeeper_attribution, action_rationale


def _x():
    return (np.random.rand(32, 128).astype("float32"),
            np.random.rand(32).astype("float32"),
            np.random.rand(3).astype("float32"))


def test_gradcam_shape_and_probs(ml_model):
    rd, dp, ev = _x()
    cam, cls, probs = gradcam_on_rd_map(ml_model, rd, dp, ev)
    assert cam.shape == (32, 128)
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert probs.shape == (5,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_ml_branch_importance_sums_to_one(ml_model):
    rd, dp, ev = _x()
    imp = ml_feature_importance(ml_model, rd, dp, ev)
    assert set(imp) == {"rd_map", "doppler", "env"}
    assert abs(sum(imp.values()) - 1.0) < 1e-5


def test_gatekeeper_attribution():
    qnet = GatekeeperQNet(doppler_dim=32)
    rd, dp, ev = _x()
    attr = gatekeeper_attribution(qnet, rd, dp, ev)
    assert attr["action"] in ("DISCARD", "FORWARD")
    assert attr["rd_saliency"].shape == (32, 128)
    assert abs(sum(attr["branch_importance"].values()) - 1.0) < 1e-5
    assert isinstance(action_rationale(attr), str)
