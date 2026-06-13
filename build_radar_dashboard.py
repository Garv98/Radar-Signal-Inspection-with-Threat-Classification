"""
build_radar_dashboard.py
========================
Generates a single self-contained, jaw-dropping 3D interactive dashboard
(`radar_dashboard.html`) that replicates the full simulator.py pipeline:

    real 77 GHz return -> preprocessing (X) -> RL gatekeeper(X)
        -> FORWARD -> 77 GHz CNN-LSTM -> class + threat
        -> DISCARD -> (ML skipped, compute saved)

This script runs the REAL models on a batch of real Zenodo returns, captures
everything each stage produces (RD map, Doppler/range profiles, gatekeeper
Q-values, ML probabilities, decision, reward), and bakes it into one HTML file
with a three.js rotating-radar scene + animated pipeline overlay panels.

Open the result by double-clicking radar_dashboard.html (needs internet once to
pull three.js from the CDN).

Usage:
    python build_radar_dashboard.py
    python build_radar_dashboard.py --returns 56 --seed 5 --out radar_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from src.rl.data_source import RadarSignalSource, CLASS_NAMES, CLASS_TO_IDX
from src.models.cnn_lstm import build_model
from src.rl.dqn_agent import GatekeeperAgent
from src.rl.encoder import count_parameters
from src.rl.reward import RewardConfig, RewardFunction, derive_class_values

CFG = yaml.safe_load(open(Path(__file__).parent / "configs" / "config.yaml"))
THREAT = CFG["reward"]["threat_classes"]
ML_PATH = Path(CFG["rl"]["ml_model_path"])
GK_PATH = Path(CFG["rl"]["save_path"])
R_MIN, R_MAX = 22.0, 130.0

TEMPLATE = Path(__file__).parent / "assets" / "radar_dashboard_template.html"


def load():
    ml = None
    if ML_PATH.exists():
        c = torch.load(ML_PATH, map_location="cpu", weights_only=False)
        ml = build_model(c.get("config", {})); ml.load_state_dict(c["model_state_dict"]); ml.eval()
    gate = GatekeeperAgent.from_checkpoint(str(GK_PATH), device="cpu") if GK_PATH.exists() else None
    rd = CFG["preprocessing"].get("fft_size", 128)
    dp = CFG["features"].get("doppler_sequence_length", 32)
    src = RadarSignalSource(mode="zenodo", range_fft=rd, doppler_fft=dp, seed=7,
                            verbose=False, keep_iq=True)
    return ml, gate, src


def run_pipeline(ml, gate, src, n, seed):
    rng = np.random.default_rng(seed)
    # reward (for the same VoI numbers the dashboard reports)
    rcfg = RewardConfig.from_dict(CFG.get("reward", {}))
    tix = [CLASS_TO_IDX[c] for c in rcfg.threat_classes if c in CLASS_TO_IDX]
    cv = derive_class_values(src.label_counts(), gamma=rcfg.value_gamma,
                             value_clip=rcfg.value_clip, threat_indices=tix,
                             threat_value=rcfg.threat_value, nonthreat_value=rcfg.nonthreat_value)
    reward_fn = RewardFunction(rcfg, cv, forward_budget=0.3)

    az = np.linspace(0, 360, n, endpoint=False); rng.shuffle(az)
    dets = []
    for k in range(n):
        s = src.sample(balanced=True)
        true_cls = CLASS_NAMES[s.label]
        q = gate.q_values((s.rd_map, s.doppler, s.env)) if gate is not None else np.array([0.0, 1.0])
        action = int(np.argmax(q))
        forward = action == 1

        if forward and ml is not None:
            spec = torch.from_numpy(s.rd_map).unsqueeze(0).unsqueeze(0)
            d = torch.from_numpy(s.doppler).unsqueeze(0); e = torch.from_numpy(s.env).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(ml(spec, d, e), 1).squeeze(0).numpy()
            pred = CLASS_NAMES[int(probs.argmax())]; conf = float(probs.max())
            pe = np.clip(probs, 1e-12, 1); ent = float(-np.sum(pe * np.log(pe)) / np.log(len(pe)))
        else:
            probs, pred, conf, ent = None, None, 0.0, 0.0

        reward, comp = reward_fn.reward(action, probs if probs is not None
                                        else F.softmax(ml(torch.from_numpy(s.rd_map).unsqueeze(0).unsqueeze(0),
                                              torch.from_numpy(s.doppler).unsqueeze(0),
                                              torch.from_numpy(s.env).unsqueeze(0)), 1).squeeze(0).detach().numpy(),
                                        s.label)

        rbin = int(s.rd_map.mean(axis=0).argmax())
        rng_m = R_MIN + (rbin / s.rd_map.shape[1]) * (R_MAX - R_MIN) * 0.9
        rd_q = (np.clip(s.rd_map, 0, 1) * 255).astype(np.uint8).flatten().tolist()

        dets.append({
            "id": k,
            "true": true_cls,
            "az": round(float(az[k] + rng.uniform(-3, 3)), 2),
            "range": round(float(rng_m), 1),
            "forward": forward,
            "q": [round(float(q[0]), 3), round(float(q[1]), 3)],
            "pred": pred,
            "conf": round(conf, 3),
            "entropy": round(ent, 3),
            "probs": [round(float(x), 3) for x in probs] if probs is not None else None,
            "threat": bool(forward and pred in THREAT),
            "reward": round(float(reward), 2),
            # value-of-information breakdown (what drove the gatekeeper decision)
            "u_ml": round(float(comp["u_ml"]), 3),
            "surplus": round(float(comp["u_surplus"]), 3),
            "certainty": round(float(comp["certainty"]), 3),
            "class_value": round(float(comp["class_value"]), 3),
            "ml_correct": (bool(pred == true_cls) if forward else None),
            "source": s.source,
            "rd": rd_q,                         # uint8 [32*128] row-major (doppler x range)
            "dop": [round(float(x), 3) for x in s.doppler.tolist()],
            "rng_prof": [round(float(x), 3) for x in s.rd_map.mean(axis=0).tolist()],
        })

    n_fwd = sum(d["forward"] for d in dets)
    n_thr_true = sum(d["true"] in THREAT for d in dets)
    n_thr_fwd = sum((d["true"] in THREAT) and d["forward"] for d in dets)
    meta = {
        "classes": CLASS_NAMES,
        "threat": THREAT,
        "rd_shape": list(s.rd_map.shape),     # [32, 128]
        "n": n,
        "forwarded": n_fwd,
        "discarded": n - n_fwd,
        "workload_reduction": round((n - n_fwd) / max(n, 1), 3),
        "threat_recall": round(n_thr_fwd / max(n_thr_true, 1), 3),
        "gk_params": count_parameters(gate.policy_net) if gate else 0,
        "ml_params": count_parameters(ml) if ml else 0,
        "present": [CLASS_NAMES[i] for i in src.present_labels],
        "kappa": round(float(rcfg.compute_cost), 3),
        "class_values": {CLASS_NAMES[i]: round(float(cv[i]), 2) for i in range(len(CLASS_NAMES))},
    }
    return {"detections": dets, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", type=int, default=52)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="radar_dashboard.html")
    args = ap.parse_args()

    print("[dashboard] loading models + real 77 GHz returns ...")
    ml, gate, src = load()
    if ml is None:
        print("ERROR: 77 GHz model missing (run finetune_zenodo.py)."); return
    data = run_pipeline(ml, gate, src, args.returns, args.seed)
    m = data["meta"]
    print(f"[dashboard] scene: {m['n']} returns | {m['forwarded']} forwarded | "
          f"{m['discarded']} discarded | workload saved {m['workload_reduction']*100:.0f}% | "
          f"threat recall {m['threat_recall']*100:.0f}%")

    if not TEMPLATE.exists():
        print(f"ERROR: template missing: {TEMPLATE}"); return
    html = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"))
    html = html.replace("/*__DATA__*/null", payload)

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"[dashboard] wrote {out}  ({size_mb:.1f} MB) -> double-click to open")


if __name__ == "__main__":
    main()
