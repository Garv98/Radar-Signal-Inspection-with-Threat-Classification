"""
verify_ml_model.py
==================
Independent verification of the ML teacher (`best_model_zenodo.pt`) on the real
77 GHz Zenodo data, plus the base model (`best_model.pt`) on synthetic data.

Reports, beyond accuracy/F1:
  * per-class precision/recall on the genuine held-out test split (seed 42),
  * full confusion matrix,
  * class support (exposes imbalance / absent classes),
  * confidence calibration (ECE) and confidence-vs-correctness -- this matters
    because the RL gatekeeper's reward uses the model's confidence/entropy.

Usage:
    python scripts/verify_ml_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.data_source import load_zenodo_signals, CLASS_NAMES
from src.models.cnn_lstm import build_model, count_parameters

ZEN_MODEL = Path("outputs/models/best_model_zenodo.pt")
BASE_MODEL = Path("outputs/models/best_model.pt")


@torch.no_grad()
def predict(model, specs, dops, envs, device, bs=128):
    probs = []
    for i in range(0, len(specs), bs):
        p = F.softmax(model(specs[i:i+bs].to(device), dops[i:i+bs].to(device),
                            envs[i:i+bs].to(device)), dim=1)
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def expected_calibration_error(conf, correct, n_bins=10):
    """ECE: weighted gap between confidence and accuracy across confidence bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        ece += (m.sum() / len(conf)) * abs(acc - avg_conf)
        rows.append((lo, hi, int(m.sum()), avg_conf, acc))
    return ece, rows


def evaluate_zenodo(device):
    print("\n" + "=" * 68)
    print("  TEACHER MODEL  —  best_model_zenodo.pt  on REAL 77 GHz data")
    print("=" * 68)
    if not ZEN_MODEL.exists():
        print("  model missing."); return

    ckpt = torch.load(ZEN_MODEL, map_location=device, weights_only=False)
    model = build_model(ckpt.get("config", {})).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  params: {count_parameters(model):,}")

    sigs = load_zenodo_signals(verbose=False)
    if not sigs:
        print("  Zenodo data not found."); return
    labels = np.array([s.label for s in sigs])
    print(f"  total real segments: {len(sigs)}")
    print(f"  full-set class counts: " + "  ".join(
        f"{CLASS_NAMES[i]}={int((labels==i).sum())}" for i in range(len(CLASS_NAMES))))

    # Reproduce the fine-tuning test split (seed 42, 15% stratified held-out).
    idx = np.arange(len(sigs))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, stratify=labels,
                                        random_state=42)
    test = [sigs[i] for i in idx_test]
    y = labels[idx_test]
    print(f"  held-out TEST segments: {len(test)}")

    specs = torch.stack([torch.from_numpy(s.rd_map).unsqueeze(0) for s in test])
    dops = torch.stack([torch.from_numpy(s.doppler) for s in test])
    envs = torch.stack([torch.from_numpy(s.env) for s in test])

    probs = predict(model, specs, dops, envs, device)
    preds = probs.argmax(1)
    conf = probs.max(1)
    correct = (preds == y).astype(float)

    present = sorted(set(y.tolist()) | set(preds.tolist()))
    names = [CLASS_NAMES[i] for i in present]
    print("\n  TEST-SPLIT classification report:")
    print(classification_report(y, preds, labels=present, target_names=names,
                                zero_division=0, digits=3))
    print("  confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y, preds, labels=present)
    header = "        " + " ".join(f"{n[:6]:>7}" for n in names)
    print(header)
    for r, n in zip(cm, names):
        print(f"  {n[:6]:>6} " + " ".join(f"{v:7d}" for v in r))

    acc = correct.mean()
    ece, rows = expected_calibration_error(conf, correct)
    print(f"\n  accuracy           : {acc*100:.1f}%")
    print(f"  mean confidence    : {conf.mean()*100:.1f}%")
    print(f"  calibration error (ECE): {ece:.3f}   "
          f"({'well calibrated' if ece < 0.1 else 'OVER/UNDER-CONFIDENT' })")
    print("  confidence bin   n     avg_conf   accuracy")
    for lo, hi, n, c, a in rows:
        flag = "  <-- gap" if abs(c - a) > 0.15 else ""
        print(f"    {lo:.1f}-{hi:.1f}    {n:5d}    {c*100:6.1f}%    {a*100:6.1f}%{flag}")

    # Caveats relevant to the gatekeeper
    print("\n  NOTES for the gatekeeper:")
    full_counts = Counter(labels.tolist())
    for ci, cn in enumerate(CLASS_NAMES):
        if full_counts.get(ci, 0) == 0:
            print(f"    - '{cn}' has NO real data -> model cannot learn it on 77 GHz "
                  f"(threat coverage gap)" if cn in ("Aircraft",) else
                  f"    - '{cn}' absent from real data.")
    dom = max(full_counts, key=full_counts.get)
    print(f"    - majority class is '{CLASS_NAMES[dom]}' "
          f"({full_counts[dom]/len(sigs)*100:.0f}% of data) -> accuracy is "
          f"optimistic; trust macro-F1 / per-class recall instead.")


def evaluate_base_synthetic(device, n_per_class=300):
    print("\n" + "=" * 68)
    print("  BASE MODEL  —  best_model.pt  on fresh synthetic data (5 classes)")
    print("=" * 68)
    if not BASE_MODEL.exists():
        print("  model missing."); return
    from src.rl.data_source import generate_synthetic_signals

    ckpt = torch.load(BASE_MODEL, map_location=device, weights_only=False)
    model = build_model(ckpt.get("config", {})).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  reported val_acc at train time: {ckpt.get('val_acc', '?')}")

    # FRESH samples (different seed) -> genuine generalization check.
    sigs = generate_synthetic_signals(per_class=n_per_class, seed=999, verbose=False)
    y = np.array([s.label for s in sigs])
    specs = torch.stack([torch.from_numpy(s.rd_map).unsqueeze(0) for s in sigs])
    dops = torch.stack([torch.from_numpy(s.doppler) for s in sigs])
    envs = torch.stack([torch.from_numpy(s.env) for s in sigs])

    probs = predict(model, specs, dops, envs, device)
    preds = probs.argmax(1)
    print(f"  fresh synthetic samples: {len(sigs)} ({n_per_class}/class)")
    print(classification_report(y, preds, target_names=CLASS_NAMES,
                                zero_division=0, digits=3))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    evaluate_zenodo(device)
    evaluate_base_synthetic(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
