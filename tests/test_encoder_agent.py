"""Encoder, Q-net and gatekeeper agent: shapes, cheapness, persistence."""

import numpy as np
import torch

from src.rl.encoder import GatekeeperQNet, count_parameters
from src.rl.dqn_agent import GatekeeperAgent
from src.models.cnn_lstm import build_model


def test_qnet_output_shape_and_consumes_X():
    qnet = GatekeeperQNet(doppler_dim=32, env_dim=3)
    rd = torch.rand(4, 32, 128)      # same RD-map tensor the ML model consumes
    dp = torch.rand(4, 32)
    ev = torch.rand(4, 3)
    out = qnet(rd, dp, ev)
    assert out.shape == (4, 2)


def test_gatekeeper_cheaper_than_ml():
    qnet = GatekeeperQNet(doppler_dim=32)
    ml = build_model({"dataset": {"num_classes": 5}})
    assert count_parameters(qnet) < count_parameters(ml) / 10  # >10x cheaper


def test_agent_action_space():
    agent = GatekeeperAgent(doppler_dim=32, device="cpu", eps_start=0.0)
    state = (np.random.rand(32, 128).astype("float32"),
             np.random.rand(32).astype("float32"),
             np.random.rand(3).astype("float32"))
    a = agent.select_action(state, training=False)
    assert a in (0, 1)
    q = agent.q_values(state)
    assert q.shape == (2,)


def test_agent_learns_and_roundtrips(tmp_path):
    agent = GatekeeperAgent(doppler_dim=32, device="cpu", batch_size=8,
                            buffer_capacity=64)

    def rnd_state():
        return (np.random.rand(32, 128).astype("float32"),
                np.random.rand(32).astype("float32"),
                np.random.rand(3).astype("float32"))

    for _ in range(40):
        s, ns = rnd_state(), rnd_state()
        agent.store(s, np.random.randint(2), float(np.random.randn()), ns, True)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)

    path = tmp_path / "gk.pt"
    agent.save(str(path))
    restored = GatekeeperAgent.from_checkpoint(str(path), device="cpu")
    s = rnd_state()
    assert np.allclose(agent.q_values(s), restored.q_values(s), atol=1e-5)
