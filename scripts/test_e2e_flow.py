"""
test_e2e_flow.py
================
End-to-end CORRECTNESS check of the on-spec 77 GHz pipeline:

    real Zenodo 77 GHz segment  ->  preprocessing (X)
        ->  RL gatekeeper(X)  ->  {DISCARD | FORWARD}
              FORWARD -> 77 GHz ML classifier(X) -> class

Verifies that the decisions are *sensible*, not just non-crashing:
  * ML accuracy per class on FORWARDED signals (should be high -- matched domain),
  * gatekeeper forward rate per class (threats forwarded >> non-threats),
  * true threat recall (forwarded threats / all threats),
  * realised ML workload reduction.

Usage:
    python scripts/test_e2e_flow.py            # uses the configured gatekeeper + 77 GHz model
    python scripts/test_e2e_flow.py --n 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.data_source import RadarSignalSource, CLASS_NAMES, CLASS_TO_IDX
from src.rl.dqn_agent import GatekeeperAgent
from src.models.cnn_lstm import build_model


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    rl_cfg = cfg["rl"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ml", default=rl_cfg["ml_model_by_source"]["zenodo"])
    ap.add_argument("--agent", default=rl_cfg["save_path"])
    args = ap.parse_args()

    device = "cpu"
    rd_fft = cfg["preprocessing"].get("fft_size", 128)
    dop_fft = cfg["features"].get("doppler_sequence_length", 32)
    threats = {CLASS_TO_IDX[c] for c in cfg["reward"]["threat_classes"]}

    # ── load 77 GHz model + gatekeeper ──────────────────────────────────────
    ml_ckpt = torch.load(args.ml, map_location=device, weights_only=False)
    ml = build_model(ml_ckpt.get("config", {})).to(device).eval()
    ml.load_state_dict(ml_ckpt["model_state_dict"])

    have_gate = Path(args.agent).exists()
    agent = GatekeeperAgent.from_checkpoint(args.agent, device=device) if have_gate else None

    src = RadarSignalSource(mode="zenodo", range_fft=rd_fft, doppler_fft=dop_fft,
                            seed=11, verbose=False)
    present = [CLASS_NAMES[i] for i in src.present_labels]
    print("=" * 66)
    print("  END-TO-END FLOW CORRECTNESS  (real 77 GHz -> gatekeeper -> 77 GHz ML)")
    print("=" * 66)
    print(f"  gatekeeper: {'loaded' if have_gate else 'MISSING (forward-all baseline)'}")
    print(f"  classes present in real data: {present}")

    @torch.no_grad()
    def ml_predict(s):
        spec = torch.from_numpy(s.rd_map).unsqueeze(0).unsqueeze(0)
        d = torch.from_numpy(s.doppler).unsqueeze(0)
        e = torch.from_numpy(s.env).unsqueeze(0)
        p = F.softmax(ml(spec, d, e), 1).squeeze(0).numpy()
        return int(p.argmax()), float(p.max())

    fwd = defaultdict(int); tot = defaultdict(int)
    correct_fwd = defaultdict(int); n_fwd = defaultdict(int)
    threat_total = threat_fwd = 0
    forwards = 0

    for _ in range(args.n):
        s = src.sample(balanced=True)
        tot[s.label] += 1
        if s.label in threats:
            threat_total += 1
        # gatekeeper decision on X (before ML)
        if agent is not None:
            action = int(np.argmax(agent.q_values((s.rd_map, s.doppler, s.env))))
        else:
            action = 1
        if action == 1:                       # FORWARD -> run ML
            forwards += 1
            fwd[s.label] += 1
            if s.label in threats:
                threat_fwd += 1
            pred, _ = ml_predict(s)
            n_fwd[s.label] += 1
            correct_fwd[s.label] += int(pred == s.label)

    print(f"\n  signals: {args.n}   forwarded: {forwards} "
          f"({forwards/args.n*100:.0f}%)   discarded: {args.n-forwards} "
          f"({(args.n-forwards)/args.n*100:.0f}%  ML skipped)")
    print("\n  per-class behaviour:")
    print(f"    {'class':9s} {'n':>5s} {'forward%':>9s} {'ML-acc(fwd)':>12s}")
    for i in src.present_labels:
        fr = fwd[i] / max(tot[i], 1) * 100
        acc = correct_fwd[i] / max(n_fwd[i], 1) * 100 if n_fwd[i] else float('nan')
        tag = " (threat)" if i in threats else ""
        print(f"    {CLASS_NAMES[i]:9s} {tot[i]:5d} {fr:8.0f}% {acc:11.0f}%{tag}")

    threat_recall = threat_fwd / max(threat_total, 1) * 100
    print(f"\n  true threat recall (forwarded/total threats): {threat_recall:.0f}%")
    print(f"  ML workload reduction: {(args.n-forwards)/args.n*100:.0f}%")

    # ── verdicts ────────────────────────────────────────────────────────────
    print("\n  CHECKS:")
    overall_acc = (sum(correct_fwd.values()) / max(sum(n_fwd.values()), 1)) * 100
    print(f"    [{'PASS' if overall_acc >= 85 else 'FAIL'}] ML accuracy on forwarded "
          f"signals >= 85%  (got {overall_acc:.0f}%)")
    if agent is not None and threats:
        print(f"    [{'PASS' if threat_recall >= 85 else 'WARN'}] threat recall >= 85% "
              f"(got {threat_recall:.0f}%)")
        # threats should be forwarded more than non-threats
        tfr = np.mean([fwd[i]/max(tot[i],1) for i in src.present_labels if i in threats]) if any(i in threats for i in src.present_labels) else 0
        ntfr = np.mean([fwd[i]/max(tot[i],1) for i in src.present_labels if i not in threats]) or 0
        print(f"    [{'PASS' if tfr >= ntfr else 'WARN'}] threats forwarded more than "
              f"non-threats  (threat {tfr*100:.0f}% vs non-threat {ntfr*100:.0f}%)")
    print("=" * 66)


if __name__ == "__main__":
    main()
