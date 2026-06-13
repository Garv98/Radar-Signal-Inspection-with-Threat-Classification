"""
train_rl.py  --  Curriculum training for the RL computational gatekeeper
========================================================================

Trains a DQN gatekeeper that decides, from the shared feature tensor X alone,
whether to DISCARD a signal or FORWARD it for (expensive) ML deep inspection.
The ML classifier is the *teacher*: it scores the Value-of-Information reward
and provides oracle demonstrations during warm-up.  The gatekeeper is the
*student*: a tiny network (~30K params vs ~1.1M for the ML model) that learns
to skip ML inference on signals where it adds little decision value.

Algorithm: Double Dueling DQN + Prioritized Experience Replay, gamma = 0
(per-signal contextual bandit -- the correct model for i.i.d. gatekeeping).
Justification for DQN over PPO/SAC: discrete binary action, high sample
efficiency from off-policy replay of teacher demonstrations, and stable myopic
targets (no bootstrap) suited to real-time single-step decisions.

Three-phase curriculum
-----------------------
  Phase 1  DEMONSTRATION  -- ML-oracle drives every action; buffer pre-filled.
  Phase 2  GUIDED         -- teacher ratio annealed teacher_ratio_start -> 0.
  Phase 3  RL             -- pure epsilon-greedy; student surpasses the oracle's
                            cost/accuracy trade-off.

MLOps: MLflow experiment tracking, model registry, and a drift baseline are
written to outputs/.

Usage
-----
    python train_rl.py
    python train_rl.py --fast
    python train_rl.py --data-source zenodo --rl 400
"""

from __future__ import annotations

import argparse
import sys
import time
import json
import random
from pathlib import Path
from collections import deque

import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.rl.data_source import RadarSignalSource, CLASS_NAMES, CLASS_TO_IDX
from src.rl.reward import RewardConfig, RewardFunction, derive_class_values, ACTION_FORWARD
from src.rl.radar_env import GatekeeperEnv
from src.rl.dqn_agent import GatekeeperAgent
from src.models.cnn_lstm import build_model
from src.utils.experiment_tracking import ExperimentTracker, ModelRegistry

CONFIG_PATH = Path("configs/config.yaml")
LOG_DIR = Path("outputs/logs")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_ml_teacher(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("config", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def estimate_forward_budget(source: RadarSignalSource, ml, reward_fn: RewardFunction,
                            device: str, n: int = 600) -> float:
    """
    Derive the target forward rate rho* from the data: the fraction of signals
    the cost-aware oracle would forward.  This makes the budget data-driven
    rather than a hardcoded constant.
    """
    import torch.nn.functional as F
    fwd = 0
    for _ in range(n):
        sig = source.sample(balanced=True)
        spec = torch.from_numpy(sig.rd_map).unsqueeze(0).unsqueeze(0).to(device)
        dop = torch.from_numpy(sig.doppler).unsqueeze(0).to(device)
        env = torch.from_numpy(sig.env).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(ml(spec, dop, env), dim=1).squeeze(0).cpu().numpy()
        u, _ = reward_fn.ml_utility(probs, sig.label)
        fwd += int(reward_fn.teacher_action(u) == ACTION_FORWARD)
    return fwd / max(n, 1)


def precompute_teacher_probs(source, ml, device, bs: int = 256) -> None:
    """
    Batch the entire (fixed) signal pool through the (fixed) teacher once and
    cache the posterior on each Signal.  Exact -- the teacher is deterministic
    in eval mode -- and removes all per-step ML inference during training.
    """
    import torch.nn.functional as F
    pool = source.pool
    todo = [s for s in pool if s.probs is None]
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        spec = torch.from_numpy(np.stack([s.rd_map for s in chunk])).unsqueeze(1).to(device)
        dop = torch.from_numpy(np.stack([s.doppler for s in chunk])).to(device)
        env = torch.from_numpy(np.stack([s.env for s in chunk])).to(device)
        with torch.no_grad():
            p = F.softmax(ml(spec, dop, env), dim=1).cpu().numpy().astype(np.float32)
        for s, pr in zip(chunk, p):
            s.probs = pr


def evaluate(env: GatekeeperEnv, agent: GatekeeperAgent, n_episodes: int,
             threat_idx: Optional[set] = None) -> dict:
    """
    Greedy-policy evaluation.  In addition to the oracle-referenced env metrics,
    computes the *true* threat recall (fraction of genuine threat signals that
    were FORWARDED) -- the safety metric used for model selection.
    """
    threat_idx = threat_idx or set()
    agg = {"forward_rate": 0.0, "workload_reduction": 0.0, "f1": 0.0,
           "recall": 0.0, "miss_rate": 0.0, "avg_reward": 0.0, "accuracy": 0.0}
    threat_total = threat_fwd = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, training=False)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            if info["true_label"] in threat_idx:
                threat_total += 1
                threat_fwd += int(info["action"] == "forward")
        m = env.episode_metrics()
        for k in agg:
            agg[k] += m[k]
    out = {k: v / n_episodes for k, v in agg.items()}
    out["threat_recall"] = (threat_fwd / threat_total) if threat_total else 1.0
    return out


def plot_training(history: dict, save_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor("#0d1525")
    for ax in axes.flat:
        ax.set_facecolor("#0a0e1a")
        ax.tick_params(colors="#7aa3cc")
        ax.title.set_color("#00d4ff")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3a5f")

    def _p(ax, data, color, title, ylabel):
        if data:
            ax.plot(data, color=color, linewidth=1.5)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.2)

    _p(axes[0, 0], history["ep_reward"], "#00d4ff", "Episode Reward", "reward")
    _p(axes[0, 1], history["loss"], "#FF851B", "DQN Loss", "loss")
    _p(axes[0, 2], history["epsilon"], "#2ECC40", "Exploration (epsilon)", "eps")
    _p(axes[1, 0], history["eval_reward"], "#00d4ff", "Eval Avg Reward", "reward")
    _p(axes[1, 1], history["eval_savings"], "#2ECC40", "ML Workload Reduction", "fraction")
    _p(axes[1, 2], history["eval_miss"], "#FF4136", "Threat Miss Rate", "rate")

    for ax in axes.flat:
        for x, label, c in history.get("phase_marks", []):
            ax.axvline(x, color=c, linestyle="--", alpha=0.5)

    plt.suptitle("RL Gatekeeper -- Curriculum Training", color="#00d4ff", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  training plot -> {save_path}")


def write_drift_baseline(source: RadarSignalSource, path: Path, n: int = 1000) -> None:
    """Reference feature statistics for production drift monitoring."""
    rd_means, rd_stds, peaks, labels = [], [], [], []
    for _ in range(n):
        s = source.sample(balanced=False)
        rd_means.append(float(s.rd_map.mean()))
        rd_stds.append(float(s.rd_map.std()))
        peaks.append(float(s.rd_map.max()))
        labels.append(s.label)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).tolist()
    baseline = {
        "n_samples": n,
        "rd_map_mean": {"mean": float(np.mean(rd_means)), "std": float(np.std(rd_means))},
        "rd_map_std": {"mean": float(np.mean(rd_stds)), "std": float(np.std(rd_stds))},
        "rd_map_peak": {"mean": float(np.mean(peaks)), "std": float(np.std(peaks))},
        "class_distribution": {CLASS_NAMES[i]: counts[i] for i in range(len(CLASS_NAMES))},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  drift baseline -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    rl_cfg = cfg["rl"]
    cur = rl_cfg["curriculum"]

    ap = argparse.ArgumentParser(description="RL gatekeeper curriculum training")
    ap.add_argument("--data-source", default=rl_cfg["data_source"],
                    choices=["zenodo", "synthetic", "mixed"])
    ap.add_argument("--warmup", type=int, default=cur["warmup_episodes"])
    ap.add_argument("--guided", type=int, default=cur["guided_episodes"])
    ap.add_argument("--rl", type=int, default=cur["rl_episodes"])
    ap.add_argument("--steps", type=int, default=cur["steps_per_episode"])
    ap.add_argument("--seed", type=int, default=cur["seed"])
    ap.add_argument("--ml-model", default=None,
                    help="override teacher checkpoint (else chosen by data source)")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--fast", action="store_true", help="smoke test")
    args = ap.parse_args()

    if args.fast:
        args.warmup, args.guided, args.rl, args.steps = 5, 8, 20, 120
        # Shrink the data pool so the smoke test is fast.
        rl_cfg["synthetic_per_class"] = 60
        rl_cfg["zenodo_max_segments"] = 400

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(rl_cfg["save_path"]); save_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RL GATEKEEPER TRAINING  (teacher = 77 GHz ML classifier)")
    print("=" * 70)

    # ── ML teacher (domain-matched so its reward signal is reliable) ────────
    if args.ml_model:
        ml_path = Path(args.ml_model)
    else:
        by_source = rl_cfg.get("ml_model_by_source", {})
        ml_path = Path(by_source.get(args.data_source, rl_cfg["ml_model_path"]))
    if not ml_path.exists():
        print(f"[ERROR] ML teacher not found: {ml_path}\n"
              f"        Run train.py / finetune_zenodo.py first."); return
    ml, ml_ckpt = load_ml_teacher(ml_path, device)
    print(f"  ML teacher: {ml_path.name}  (matched to data source '{args.data_source}')")

    # ── Data source (discovers real + synthetic) ────────────────────────────
    rd_fft = cfg["preprocessing"].get("fft_size", 128)
    dop_fft = cfg["features"].get("doppler_sequence_length", 32)
    source = RadarSignalSource(
        mode=args.data_source,
        synthetic_per_class=rl_cfg["synthetic_per_class"],
        zenodo_max_segments=rl_cfg["zenodo_max_segments"],
        range_fft=rd_fft, doppler_fft=dop_fft, seed=args.seed,
    )
    counts = source.label_counts()

    # ── Reward (class values derived from the data) ─────────────────────────
    rcfg = RewardConfig.from_dict(cfg.get("reward", {}))
    threat_idx = [CLASS_TO_IDX[c] for c in rcfg.threat_classes if c in CLASS_TO_IDX]
    threat_set = set(threat_idx)
    min_recall = float(rl_cfg.get("selection", {}).get("min_threat_recall", 0.90))
    class_values = derive_class_values(
        counts, gamma=rcfg.value_gamma, value_clip=rcfg.value_clip,
        threat_indices=threat_idx, threat_value=rcfg.threat_value,
        nonthreat_value=rcfg.nonthreat_value,
    )
    print("  class values: " + "  ".join(
        f"{CLASS_NAMES[i]}={class_values[i]:.2f}" for i in range(len(CLASS_NAMES))))

    reward_fn = RewardFunction(rcfg, class_values, forward_budget=None)
    budget = estimate_forward_budget(source, ml, reward_fn, device,
                                     n=120 if args.fast else 600)
    reward_fn.forward_budget = rcfg.forward_budget if rcfg.forward_budget is not None else budget
    print(f"  forward budget rho* = {reward_fn.forward_budget:.3f} "
          f"({'config' if rcfg.forward_budget is not None else 'data-derived'})")

    # ── Environments ────────────────────────────────────────────────────────
    env = GatekeeperEnv(ml, source, reward_fn, max_steps=args.steps,
                        device=device, seed=args.seed)
    eval_source = RadarSignalSource(
        mode=args.data_source, synthetic_per_class=max(100, rl_cfg["synthetic_per_class"] // 3),
        zenodo_max_segments=rl_cfg["zenodo_max_segments"],
        range_fft=rd_fft, doppler_fft=dop_fft, seed=args.seed + 1, verbose=False,
    )
    eval_env = GatekeeperEnv(ml, eval_source, reward_fn, max_steps=args.steps,
                             device=device, seed=args.seed + 1)

    # Precompute the (fixed) teacher posteriors for both pools so training does
    # no per-step ML inference -> ~20x faster, completes in minutes.
    print("  precomputing teacher posteriors (one batched pass) ...", flush=True)
    precompute_teacher_probs(source, ml, device)
    precompute_teacher_probs(eval_source, ml, device)

    # ── Agent ───────────────────────────────────────────────────────────────
    ag = rl_cfg["agent"]
    agent = GatekeeperAgent(
        doppler_dim=dop_fft, env_dim=3, n_actions=2,
        encoder_cfg=rl_cfg["encoder"],
        head_hidden=tuple(rl_cfg["head"]["hidden_dims"]),
        stream_hidden=rl_cfg["head"]["stream_hidden"],
        lr=ag["lr"], gamma=ag["gamma"],
        eps_start=ag["eps_start"], eps_end=ag["eps_end"], eps_decay=ag["eps_decay"],
        batch_size=ag["batch_size"], target_update=ag["target_update"],
        tau=ag["tau"], grad_clip=ag["grad_clip"], buffer_capacity=ag["buffer_capacity"],
        per_alpha=ag["per_alpha"], per_beta_start=ag["per_beta_start"],
        per_beta_frames=ag["per_beta_frames"], device=device,
    )

    # ── MLflow tracking ─────────────────────────────────────────────────────
    tcfg = cfg.get("tracking", {})
    tracker = ExperimentTracker(
        experiment_name=tcfg.get("experiment_name", "radar_rl_gatekeeper"),
        tracking_uri=tcfg.get("tracking_uri", "outputs/mlruns"),
        use_mlflow=(tcfg.get("use_mlflow", True) and not args.no_mlflow),
    )
    tracker.start_run(run_name=f"gatekeeper_{time.strftime('%Y%m%d_%H%M%S')}")
    tracker.log_params({
        "data_source": args.data_source, "warmup": args.warmup, "guided": args.guided,
        "rl_episodes": args.rl, "steps": args.steps, "gamma": ag["gamma"],
        "compute_cost": rcfg.compute_cost, "miss_aversion": rcfg.miss_aversion,
        "forward_budget": reward_fn.forward_budget,
        "gatekeeper_params": sum(p.numel() for p in agent.policy_net.parameters()),
        "ml_params": sum(p.numel() for p in ml.parameters()),
    })

    history = {"ep_reward": [], "loss": [], "epsilon": [], "eval_reward": [],
               "eval_savings": [], "eval_miss": [], "phase_marks": []}
    recent = deque(maxlen=50)
    t0 = time.time()

    def run_episode(use_teacher: bool, teacher_ratio: float = 0.0):
        obs, _ = env.reset()
        done = False; ep_r = 0.0; losses = []
        while not done:
            if use_teacher or random.random() < teacher_ratio:
                action = env.teacher_action()
            else:
                action = agent.select_action(obs, training=True)
            nobs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store(obs, action, r, nobs, done)
            if not use_teacher:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            ep_r += r; obs = nobs
        return ep_r, (float(np.mean(losses)) if losses else 0.0)

    # ── Phase 1: demonstrations ─────────────────────────────────────────────
    print(f"\n[Phase 1] {args.warmup} oracle-demonstration episodes")
    history["phase_marks"].append((0, "P1", "#2ECC40"))
    for ep in range(args.warmup):
        ep_r, _ = run_episode(use_teacher=True)
        recent.append(ep_r)
    print(f"  buffer filled: {len(agent.memory)} transitions")

    # ── Phase 2: guided ─────────────────────────────────────────────────────
    print(f"\n[Phase 2] {args.guided} guided episodes (teacher "
          f"{cur['teacher_ratio_start']:.2f} -> 0)")
    history["phase_marks"].append((len(history["ep_reward"]), "P2", "#FF851B"))
    for ep in range(args.guided):
        tr = cur["teacher_ratio_start"] * (1.0 - ep / max(args.guided - 1, 1))
        ep_r, loss = run_episode(use_teacher=False, teacher_ratio=tr)
        recent.append(ep_r)
        history["ep_reward"].append(ep_r); history["loss"].append(loss)
        history["epsilon"].append(agent.eps)
        if (ep + 1) % max(1, args.guided // 8) == 0:
            m = evaluate(eval_env, agent, cur["eval_episodes"] // 2 or 1, threat_set)
            history["eval_reward"].append(m["avg_reward"])
            history["eval_savings"].append(m["workload_reduction"])
            history["eval_miss"].append(m["miss_rate"])
            tracker.log_metrics({"eval_reward": m["avg_reward"],
                                 "eval_savings": m["workload_reduction"],
                                 "eval_threat_recall": m["threat_recall"]},
                                step=len(history["ep_reward"]))
            print(f"  [P2] ep {ep+1:3d}/{args.guided} tr={tr:.2f} eps={agent.eps:.3f} "
                  f"loss={loss:.4f} save={m['workload_reduction']:.2f} "
                  f"threat_recall={m['threat_recall']:.2f}")

    # ── Phase 3: RL ─────────────────────────────────────────────────────────
    print(f"\n[Phase 3] {args.rl} RL episodes  (select best reward with "
          f"threat_recall >= {min_recall:.2f})")
    history["phase_marks"].append((len(history["ep_reward"]), "P3", "#FF4136"))
    best_reward = -1e9          # best reward among recall-qualifying policies
    best_recall = -1.0          # fallback: highest recall seen
    saved_any = False
    eval_freq = max(1, args.rl // 20)

    def _save_best(m, reason):
        agent.save(str(save_path), extra={
            "class_values": class_values.tolist(),
            "forward_budget": reward_fn.forward_budget,
            "reward_config": rcfg.__dict__,
            "eval_metrics": m,
            "selection_reason": reason,
            "min_threat_recall": min_recall,
        })

    for ep in range(args.rl):
        ep_r, loss = run_episode(use_teacher=False, teacher_ratio=0.0)
        recent.append(ep_r)
        history["ep_reward"].append(ep_r); history["loss"].append(loss)
        history["epsilon"].append(agent.eps)
        if (ep + 1) % eval_freq == 0:
            m = evaluate(eval_env, agent, cur["eval_episodes"], threat_set)
            history["eval_reward"].append(m["avg_reward"])
            history["eval_savings"].append(m["workload_reduction"])
            history["eval_miss"].append(1.0 - m["threat_recall"])
            tracker.log_metrics({"eval_reward": m["avg_reward"],
                                 "eval_savings": m["workload_reduction"],
                                 "eval_threat_recall": m["threat_recall"],
                                 "eval_f1": m["f1"]}, step=len(history["ep_reward"]))
            flag = ""
            qualifies = m["threat_recall"] >= min_recall
            if qualifies and m["avg_reward"] > best_reward:
                best_reward = m["avg_reward"]; saved_any = True
                _save_best(m, "reward-among-recall-qualifying"); flag = "  [BEST]"
            elif not saved_any and m["threat_recall"] > best_recall:
                # no policy has met the recall bar yet -> keep the safest one
                best_recall = m["threat_recall"]
                _save_best(m, "fallback-highest-recall"); flag = "  [BEST(recall)]"
            print(f"  [P3] ep {ep+1:3d}/{args.rl} eps={agent.eps:.3f} loss={loss:.4f} "
                  f"R={m['avg_reward']:.2f} save={m['workload_reduction']:.2f} "
                  f"threat_recall={m['threat_recall']:.2f}{flag}")

    # ── Final eval on the SHIPPED (best saved) model ────────────────────────
    if save_path.exists():
        agent.load(str(save_path))      # evaluate the checkpoint we will ship
    final = evaluate(eval_env, agent, cur["eval_episodes"], threat_set)
    print("\n" + "=" * 70)
    print("  FINAL GATEKEEPER METRICS  (shipped checkpoint)")
    for k, v in final.items():
        print(f"    {k:20s}: {v:.4f}")
    print(f"  ML workload reduction: {final['workload_reduction']*100:.1f}%  "
          f"(ML inference avoided on discarded signals)")
    print(f"  true threat recall   : {final['threat_recall']*100:.1f}%  "
          f"(target >= {min_recall*100:.0f}%)")

    # ── MLOps artifacts ─────────────────────────────────────────────────────
    plot_path = LOG_DIR / "rl_gatekeeper_training.png"
    plot_training(history, str(plot_path))
    write_drift_baseline(source, Path("outputs/drift_baseline.json"))

    tracker.log_metrics({f"final_{k}": v for k, v in final.items()})
    tracker.log_artifact(str(plot_path))
    if save_path.exists():
        tracker.log_artifact(str(save_path))
    tracker.end_run()

    try:
        registry = ModelRegistry(tcfg.get("registry_dir", "outputs/model_registry"))
        version = registry.register_model(
            model_name=rl_cfg["registry_name"], model_path=str(save_path),
            metrics=final, config={"reward": rcfg.__dict__, "agent": ag},
            tags={"type": "rl_gatekeeper", "teacher": ml_path.name},
        )
        print(f"  registered {rl_cfg['registry_name']} {version}")
    except Exception as e:
        print(f"  [warn] registry skipped: {e}")

    print(f"\n  best model -> {save_path}")
    print(f"  total time : {(time.time()-t0)/60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
