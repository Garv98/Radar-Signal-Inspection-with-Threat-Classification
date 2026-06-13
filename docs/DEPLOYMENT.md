# Deployment & Operations

## 1. Environment

```bash
# Windows (PowerShell) — use the project venv that has the full stack
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

The active environment must contain: `torch`, `gymnasium`, `mlflow`,
`scikit-learn`, `scipy`, `PyWavelets`, `streamlit`, `plotly`, `pytest`.
(Explainability uses pure-torch Grad-CAM / integrated gradients; `shap` is
optional.)

## 2. Build the pipeline end to end

```bash
# (a) Train / obtain the ML teacher on synthetic 24 GHz, then fine-tune on 77 GHz
python train.py                       # base classifier  -> outputs/models/best_model.pt
python finetune_zenodo.py             # 77 GHz fine-tune -> outputs/models/best_model_zenodo.pt

# (b) Train the RL gatekeeper. The teacher is auto-matched to the data domain
#     so its confidence/correctness (which score the reward) stay reliable:
#       synthetic -> best_model.pt   |   zenodo/mixed -> best_model_zenodo.pt
python train_rl.py --data-source synthetic   # full-class coverage, reliable synthetic teacher
python train_rl.py --data-source zenodo      # real 77 GHz (threat = Drone only)
python train_rl.py --ml-model outputs/models/best_model.pt   # explicit teacher override
python train_rl.py --fast                    # quick smoke test

# (c) Measure the realised compute savings
python scripts/benchmark_gatekeeper.py

# (d) Launch the dashboard
streamlit run simulator.py
```

## 3. Programmatic inference

```python
import torch, yaml
from src.rl.dqn_agent import GatekeeperAgent
from src.models.cnn_lstm import build_model
from src.data.dataset import iq_to_range_doppler, iq_to_doppler_profile, _normalize_env

cfg = yaml.safe_load(open("configs/config.yaml"))
device = "cpu"

# load teacher + gatekeeper
ml_ckpt = torch.load(cfg["rl"]["ml_model_path"], map_location=device, weights_only=False)
ml = build_model(ml_ckpt["config"]).eval(); ml.load_state_dict(ml_ckpt["model_state_dict"])
agent = GatekeeperAgent.from_checkpoint(cfg["rl"]["save_path"], device=device)

def inspect(iq):                      # iq: complex [P × N]
    rd  = iq_to_range_doppler(iq)
    dop = iq_to_doppler_profile(rd)
    env = _normalize_env({})
    if agent.select_action((rd, dop, env), training=False) == 0:
        return {"decision": "DISCARD"}          # ML skipped — compute saved
    spec = torch.from_numpy(rd).unsqueeze(0).unsqueeze(0)
    d = torch.from_numpy(dop).unsqueeze(0); e = torch.from_numpy(env).unsqueeze(0)
    p = torch.softmax(ml(spec, d, e), 1).squeeze(0)
    return {"decision": "FORWARD", "class": int(p.argmax()), "confidence": float(p.max())}
```

## 4. MLOps

- **Experiment tracking** — `train_rl.py` logs params/metrics/artifacts via
  `ExperimentTracker` (MLflow backend at `outputs/mlruns`, JSON fallback if
  MLflow is unavailable). View with `mlflow ui --backend-store-uri outputs/mlruns`.
- **Model registry** — each run registers the gatekeeper in
  `outputs/model_registry/registry.json`. Promote with
  `ModelRegistry.promote_to_production(name, version)`.
- **Drift monitoring** — `outputs/drift_baseline.json` stores reference feature
  statistics (RD-map mean/std/peak, class distribution). Compare live feature
  stats against it; alert when a z-score exceeds threshold or the class
  distribution diverges (e.g. population-stability index).

## 5. Tuning the cost / accuracy trade-off

All knobs live in `configs/config.yaml → reward:` (see `REWARD_JUSTIFICATION.md`):

| Goal | Change |
|------|--------|
| Save more compute (forward less) | ↑ `compute_cost` (κ) |
| Miss fewer threats (forward more) | ↑ `miss_aversion` (β) or ↑ `threat_value` |
| Cap forward rate | set `forward_budget` (ρ*) + ↑ `budget_penalty` (η) |

Re-run `train_rl.py` after changes and confirm with
`scripts/benchmark_gatekeeper.py` that threat recall stays acceptable.

## 6. Tests & CI

```bash
venv\Scripts\python -m pytest tests/ -q          # 22 tests: preprocessing, reward, encoder/agent, env/savings, explainability
```

## 7. Export (optional)

`requirements.txt` includes `onnx` / `onnxruntime`. The gatekeeper
(`agent.policy_net`) and the ML model export to ONNX with
`torch.onnx.export` for edge deployment; inputs are `(rd_map, doppler, env)`.
