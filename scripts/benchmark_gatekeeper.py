"""
benchmark_gatekeeper.py
=======================
Measure the real computational savings of the RL gatekeeper.

Compares two pipelines over an evaluation pool of signals:

  Baseline  : run the ML classifier on every signal (no gatekeeper).
  Gatekeeper: run the tiny gatekeeper on every signal, run the ML classifier
              only on FORWARD decisions.

Reports:
  * per-signal latency of gatekeeper vs ML classifier,
  * forward / discard rates,
  * measured wall-clock ML-workload reduction,
  * detection quality on the high-value (threat) classes so savings are not
    bought at the cost of missed threats.

Usage
-----
    python scripts/benchmark_gatekeeper.py
    python scripts/benchmark_gatekeeper.py --n 2000 --data-source mixed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.data_source import RadarSignalSource, CLASS_NAMES, CLASS_TO_IDX
from src.rl.dqn_agent import GatekeeperAgent
from src.rl.encoder import count_parameters
from src.models.cnn_lstm import build_model


def load_cfg(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def time_module(fn, warmup=10, iters=200) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    cfg = load_cfg()
    rl_cfg = cfg["rl"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="eval signals")
    ap.add_argument("--data-source", default=rl_cfg["data_source"],
                    choices=["zenodo", "synthetic", "mixed"])
    ap.add_argument("--agent", default=rl_cfg["save_path"])
    ap.add_argument("--ml", default=rl_cfg["ml_model_path"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rd_fft = cfg["preprocessing"].get("fft_size", 128)
    dop_fft = cfg["features"].get("doppler_sequence_length", 32)

    # ── models ──────────────────────────────────────────────────────────────
    ml_ckpt = torch.load(args.ml, map_location=device, weights_only=False)
    ml = build_model(ml_ckpt.get("config", {})).to(device).eval()
    ml.load_state_dict(ml_ckpt["model_state_dict"])

    if not Path(args.agent).exists():
        print(f"[ERROR] gatekeeper not found: {args.agent}. Run train_rl.py first.")
        return
    agent = GatekeeperAgent.from_checkpoint(args.agent, device=device)
    threat_idx = {CLASS_TO_IDX[c] for c in cfg["reward"]["threat_classes"]}

    # ── data ────────────────────────────────────────────────────────────────
    source = RadarSignalSource(mode=args.data_source,
                               synthetic_per_class=max(120, rl_cfg["synthetic_per_class"] // 3),
                               zenodo_max_segments=rl_cfg["zenodo_max_segments"],
                               range_fft=rd_fft, doppler_fft=dop_fft, seed=7)

    # ── per-module latency ──────────────────────────────────────────────────
    sig = source.sample()
    spec = torch.from_numpy(sig.rd_map).unsqueeze(0).unsqueeze(0).to(device)
    dop = torch.from_numpy(sig.doppler).unsqueeze(0).to(device)
    env = torch.from_numpy(sig.env).unsqueeze(0).to(device)
    rd_t = torch.from_numpy(sig.rd_map).unsqueeze(0).to(device)

    with torch.no_grad():
        ml_lat = time_module(lambda: ml(spec, dop, env))
        gk_lat = time_module(lambda: agent.policy_net(rd_t, dop, env))

    ml_params = count_parameters(ml)
    gk_params = count_parameters(agent.policy_net)

    # ── pipeline comparison ─────────────────────────────────────────────────
    forwards = 0
    threat_total = threat_fwd = 0
    threat_correct_fwd = 0
    t_base = t_gate = 0.0

    for _ in range(args.n):
        s = source.sample(balanced=False)
        spec = torch.from_numpy(s.rd_map).unsqueeze(0).unsqueeze(0).to(device)
        dop = torch.from_numpy(s.doppler).unsqueeze(0).to(device)
        env = torch.from_numpy(s.env).unsqueeze(0).to(device)
        rd_t = torch.from_numpy(s.rd_map).unsqueeze(0).to(device)
        is_threat = s.label in threat_idx
        threat_total += int(is_threat)

        # Baseline: ML always.
        tb = time.perf_counter()
        with torch.no_grad():
            _ = ml(spec, dop, env)
        t_base += time.perf_counter() - tb

        # Gatekeeper: gate, then ML only on forward.
        tg = time.perf_counter()
        with torch.no_grad():
            action = int(agent.policy_net(rd_t, dop, env).argmax(dim=1).item())
            if action == 1:
                logits = ml(spec, dop, env)
                pred = int(logits.argmax(dim=1).item())
        t_gate += time.perf_counter() - tg

        if action == 1:
            forwards += 1
            if is_threat:
                threat_fwd += 1
                threat_correct_fwd += int(pred == s.label)

    fwd_rate = forwards / args.n
    threat_recall = threat_fwd / max(threat_total, 1)

    print("\n" + "=" * 64)
    print("  GATEKEEPER BENCHMARK")
    print("=" * 64)
    print(f"  signals evaluated      : {args.n}")
    print(f"  gatekeeper params      : {gk_params:,}")
    print(f"  ML classifier params   : {ml_params:,}  ({ml_params/gk_params:.1f}x larger)")
    print(f"  gatekeeper latency      : {gk_lat*1e3:.3f} ms/signal")
    print(f"  ML classifier latency   : {ml_lat*1e3:.3f} ms/signal  "
          f"({ml_lat/gk_lat:.1f}x slower)")
    print("-" * 64)
    print(f"  forward rate           : {fwd_rate*100:.1f}%")
    print(f"  discard rate           : {(1-fwd_rate)*100:.1f}%  (ML inference skipped)")
    print(f"  ML calls  baseline={args.n}  gatekeeper={forwards}  "
          f"-> {(1-fwd_rate)*100:.1f}% fewer ML inferences")
    print(f"  wall-clock total  baseline={t_base*1e3:.0f} ms  "
          f"gatekeeper={t_gate*1e3:.0f} ms  "
          f"-> {(1-t_gate/max(t_base,1e-9))*100:.1f}% faster")
    print("-" * 64)
    print(f"  threat signals         : {threat_total}")
    print(f"  threat recall (forwarded): {threat_recall*100:.1f}%  "
          f"(higher = fewer missed threats)")
    if threat_fwd:
        print(f"  ML accuracy on forwarded threats: "
              f"{threat_correct_fwd/threat_fwd*100:.1f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
