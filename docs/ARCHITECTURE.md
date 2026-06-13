# Architecture — Hierarchical Radar Signal Inspection Framework

A two-stage **computational gatekeeper** that reduces ML inference cost while
preserving threat-detection performance on 77 GHz FMCW radar.

```
                        ┌──────────────────────────────────────────────┐
   Raw IQ  ───────────▶ │  Preprocessing (shared)                      │
  [P × N] complex       │  iq_to_range_doppler → RD map, Doppler, env   │
                        └───────────────────────┬──────────────────────┘
                                                │  Feature tensor X
                                                │  = (rd_map[32×128], doppler[32], env[3])
                                                ▼
                        ┌──────────────────────────────────────────────┐
                        │  RL GATEKEEPER  (student, ~33K params)        │
                        │  GatekeeperQNet(X) → Q[DISCARD, FORWARD]      │
                        └───────────┬───────────────────┬──────────────┘
                          action=0  │                   │ action=1
                          DISCARD   ▼                   ▼ FORWARD
                        ┌──────────────┐   ┌────────────────────────────┐
                        │  drop signal │   │  ML CLASSIFIER (teacher,    │
                        │  (no ML run) │   │  ~1.1M params) RadarClassifier│
                        │  COMPUTE     │   │  X → class posterior p[5]   │
                        │  SAVED       │   └─────────────┬──────────────┘
                        └──────────────┘                 ▼
                                                  Classification + threat level
```

## Core principle: the gatekeeper sees **X**, never the ML output

The agent's observation **is the same feature tensor X** that the ML model
consumes. It decides **before** the expensive model runs. On `DISCARD`, the ML
classifier is never invoked → real compute is saved. This is what makes the
agent a gatekeeper rather than a post-hoc filter.

> An earlier design derived the RL state from the ML softmax output. That is
> logically circular for a cost-reduction goal: the expensive model has already
> run by the time the agent "decides" to discard, so nothing is saved. The
> current design fixes this.

## Teacher–Student

| Role    | Component                        | Function                                   |
|---------|----------------------------------|--------------------------------------------|
| Teacher | `RadarClassifier` (CNN-LSTM)     | Scores the reward (Value-of-Information) and gives oracle demonstrations during curriculum warm-up. |
| Student | `GatekeeperQNet` (tiny CNN+DQN)  | Learns, from X alone, to predict whether deep inspection is worth its cost. |

The teacher is run during **training** to define the reward for both actions.
At **deployment** the teacher only runs on `FORWARD` decisions.

## Why DQN (not PPO/SAC/Actor-Critic)

- **Binary discrete action** (`{DISCARD, FORWARD}`) → value-based methods are a
  natural fit; no need for a policy-gradient continuous actor.
- **Sample efficiency** — off-policy replay lets us reuse teacher demonstrations
  (curriculum Phase 1), which a strictly on-policy method (PPO) cannot.
- **Stability** — with `gamma = 0` the problem is an i.i.d. **contextual bandit**
  (each signal is independent; the action taken on one signal does not change
  the next signal). Myopic targets remove bootstrap divergence entirely.
- **Real-time** — a single forward pass of a ~33K-param net per signal.

Enhancements: **Double DQN** (decouples action selection/evaluation),
**Dueling** streams (separate state-value/advantage), **Prioritized Experience
Replay** (focuses on high-TD-error transitions, e.g. rare confirmable threats).

## Shared feature representation

`src/data/dataset.py::iq_to_range_doppler` is the single feature path:

1. Range FFT (fast-time) → Doppler FFT (slow-time) → `fftshift` → dB magnitude
2. Min-max normalise to `[0, 1]` → `rd_map [32 × 128]`
3. `doppler = rd_map.mean(axis=1)` → `[32]`; `env` normalised → `[3]`

Both the ML model and the gatekeeper consume exactly this `X`. There is no
separate feature-engineering path.

## Module map

| Path | Responsibility |
|------|----------------|
| `src/data/preprocessing.py`, `src/data/dataset.py` | Shared IQ → X preprocessing |
| `src/data/synthetic_generator.py` | Physics-based 77 GHz FMCW simulator (JEM, micro-Doppler) |
| `src/rl/data_source.py` | Unified real (Zenodo) + synthetic signal pool → X |
| `src/rl/encoder.py` | `FeatureEncoder` + `GatekeeperQNet` (dueling) |
| `src/rl/reward.py` | Value-of-Information reward + data-derived class values |
| `src/rl/radar_env.py` | `GatekeeperEnv` (state = X, ML only on FORWARD) |
| `src/rl/dqn_agent.py` | `GatekeeperAgent` (Double Dueling DQN + PER) |
| `src/models/cnn_lstm.py` | `RadarClassifier` teacher (CNN + BiLSTM + SE attention) |
| `src/explain/` | Grad-CAM + integrated-gradients for the ML model |
| `src/rl/explain.py` | Gatekeeper decision attribution + rationale |
| `src/utils/experiment_tracking.py` | MLflow tracking + model registry |
| `train_rl.py` | Curriculum training pipeline (the RL deliverable) |
| `train.py`, `finetune_zenodo.py` | ML training / 77 GHz fine-tuning pipelines |
| `simulator.py` | Streamlit dashboard / end-to-end simulator |
| `scripts/benchmark_gatekeeper.py` | Measured workload-reduction benchmark |

See `DATA_FLOW.md` for the runtime sequence and `REWARD_JUSTIFICATION.md` for the
reward mathematics.
