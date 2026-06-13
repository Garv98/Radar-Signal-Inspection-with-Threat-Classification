# Data Flow

## Runtime (inference / deployment)

```
 ┌─────────┐   ┌───────────────┐   ┌──────────────┐   ┌───────────────┐
 │ Raw IQ  │──▶│ Preprocessing │──▶│  Feature X    │──▶│ RL Gatekeeper │
 │ [P×N]   │   │ RD/Doppler/env│   │ rd,dop,env    │   │ Q[disc,fwd]   │
 └─────────┘   └───────────────┘   └──────────────┘   └───────┬───────┘
                                                              │
                                       ┌──────────────────────┴───────────┐
                                       │ argmax Q                          │
                              DISCARD ◀┘                                   └▶ FORWARD
                              ┌─────────────┐                       ┌───────────────────┐
                              │ stop.        │                      │ ML Classifier(X)  │
                              │ ML NOT run.  │                      │ → posterior p[5]  │
                              │ compute saved│                      │ → class + threat  │
                              └─────────────┘                       └───────────────────┘
```

The ML classifier runs **only** on the FORWARD branch. Discards cost one tiny
gatekeeper forward pass (~33K params) instead of a full classifier (~1.1M).

## Training (the teacher is always run to score the reward)

```
 sample Signal (X, y)  from RadarSignalSource (real Zenodo + synthetic)
        │
        ├─▶ ML teacher(X) → posterior p           (scores reward both actions)
        │       │
        │       ├─▶ U_ml(X) = v[y]·(2·correct−1)·certainty
        │       └─▶ oracle teacher_action = 1[U_ml ≥ κ]   (curriculum demos)
        │
        ├─▶ agent.select_action(X)  (ε-greedy / teacher-mixed by phase)
        │
        ├─▶ reward(action, p, y, ρ)  →  r
        │
        └─▶ replay.push(X, action, r, X', done);  agent.learn()  (Double-DQN + PER)
```

`ml_invocations` (the deployment cost) is tracked as the count of FORWARD
actions, separately from the reward-scoring ML calls that training performs.

## Curriculum schedule

| Phase | Episodes (cfg) | Policy | Buffer |
|-------|----------------|--------|--------|
| 1 Demonstration | `warmup_episodes` | oracle `teacher_action` only | filled with expert transitions |
| 2 Guided | `guided_episodes` | teacher ratio `0.9 → 0` mixed with ε-greedy | learns each step |
| 3 RL | `rl_episodes` | pure ε-greedy (ε decays to `eps_end`) | learns each step |

### Model selection (safety)

The shipped checkpoint is the one with the **highest eval reward among policies
whose true threat recall ≥ `rl.selection.min_threat_recall`** (default 0.90), so
compute savings are never bought with missed threats. If no policy clears the
bar, the **highest-recall** policy is kept as a safe fallback. The final
reported metrics are computed on this shipped checkpoint (reloaded), not the
last in-memory policy.

### Teacher–data domain matching

The teacher's confidence/correctness define the reward, so it must be reliable
on the data it scores. `rl.ml_model_by_source` maps each data source to a
domain-matched teacher: `synthetic → best_model.pt` (≈99% on synthetic),
`zenodo/mixed → best_model_zenodo.pt` (77 GHz fine-tuned). `--ml-model`
overrides this.

## Datasets discovered

| Source | Path | Modality | Classes present |
|--------|------|----------|-----------------|
| Real 77 GHz | `data/real/zenodo_77ghz/data_SAAB_SIRS_77GHz_FMCW.npy` | complex IQ → CPI → RD map | Drone (D1–D6), Bird, Clutter (human/CR) |
| Synthetic | `src/data/synthetic_generator.py` | physics-based FMCW IQ | all 5 (adds Aircraft, Noise) |

`RadarSignalSource(mode=...)` selects `zenodo` / `synthetic` / `mixed`. Class
values for the reward are derived from the realised pool counts.

## Artifacts produced

| Artifact | Path |
|----------|------|
| Gatekeeper checkpoint | `outputs/models/gatekeeper_dqn.pt` |
| Training curves | `outputs/logs/rl_gatekeeper_training.png` |
| Drift baseline (feature stats) | `outputs/drift_baseline.json` |
| MLflow runs | `outputs/mlruns/` |
| Model registry | `outputs/model_registry/registry.json` |
