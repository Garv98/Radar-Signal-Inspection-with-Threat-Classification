# Radar Signal Inspection Framework (RL Gatekeeper + ML Classifier)

A production-oriented, hierarchical inspection pipeline for 77 GHz FMCW radar:

```
Raw IQ → Preprocessing → Feature tensor X → RL Gatekeeper(X) → {DISCARD | FORWARD}
                                                                   FORWARD → ML Classifier(X) → class + threat
```

The **RL gatekeeper** is a lightweight learned triage stage (~33K params) that
decides — from the *same* feature tensor `X` the ML model consumes — whether a
signal is worth the cost of deep ML inspection (~1.1M params). On `DISCARD`, the
ML model never runs, so compute is saved. The **ML classifier is the teacher**:
it scores a Value-of-Information reward and gives oracle demonstrations.

> Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
> data flow: [docs/DATA_FLOW.md](docs/DATA_FLOW.md) ·
> reward math: [docs/REWARD_JUSTIFICATION.md](docs/REWARD_JUSTIFICATION.md) ·
> ops: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Why a gatekeeper (and why DQN)

- **Compute savings**: skip the expensive classifier on signals it would add no
  decision value to (obvious background/noise), keep it for uncertain/high-value
  (threat) signals.
- **DQN** fits the binary action, reuses off-policy teacher demonstrations
  (sample-efficient), and with `gamma=0` is a stable contextual bandit. Enhanced
  with Double DQN, Dueling streams, and Prioritized Experience Replay.
- **Reward** = Value of Information: `forward iff U_ml ≥ κ`, with an asymmetric
  penalty for discarding a confirmable high-value detection. Every coefficient is
  in `configs/config.yaml`; class values are derived from the data.

## Quickstart

```bash
# environment (use venv/ — it has the full stack)
venv\Scripts\pip install -r requirements.txt

# 1. ML teacher
python train.py                 # base classifier (synthetic, 5-class)
python finetune_zenodo.py       # fine-tune on real 77 GHz Zenodo data
python scripts/verify_ml_model.py   # independent ML verification (acc, calibration, coverage)

# 2. RL gatekeeper (teacher auto-matched to the data domain)
python train_rl.py --data-source synthetic   # full-class coverage
python train_rl.py --data-source zenodo       # real 77 GHz (threat = Drone)

# 3. Measure savings and explore
python scripts/benchmark_gatekeeper.py
streamlit run simulator.py

# tests
python -m pytest tests/ -q
```

## Layout

| Area | Path |
|------|------|
| Shared preprocessing (feature tensor X) | `src/data/` |
| ML classifier (teacher) | `src/models/cnn_lstm.py` |
| Physics-based FMCW simulator | `src/data/synthetic_generator.py` |
| RL gatekeeper (env, encoder, agent, reward, data source, explainability) | `src/rl/` |
| ML explainability (Grad-CAM, integrated gradients) | `src/explain/` |
| MLflow tracking + model registry | `src/utils/experiment_tracking.py` |
| Training pipelines | `train.py`, `finetune_zenodo.py`, `train_rl.py` |
| Benchmarks & verification | `scripts/` |
| Dashboard / simulator | `simulator.py` |
| Tests | `tests/` (22) |
| Docs | `docs/` |

## Datasets

- **Real 77 GHz** — SAAB SIRS FMCW (`data/real/zenodo_77ghz/`): 6 drones, humans,
  birds, corner reflector. See [DATASETS.md](DATASETS.md).
- **Synthetic** — physics-based generator with JEM, rotor/wingbeat micro-Doppler,
  clutter and noise models (covers Aircraft/Noise, absent from the real set).

## Notes / honesty

- The real 77 GHz set has **no Aircraft or Noise** samples, so on real data only
  the **Drone** threat is data-validated; `verify_ml_model.py` documents this.
- On a balanced-class stream the gatekeeper can discard the non-threat majority;
  on the real set (78% drones) the achievable workload reduction is lower — a
  property of the data's threat base rate, not of the method.
