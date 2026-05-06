"""
Fine-tune the pre-trained RadarClassifier on real Zenodo 77GHz FMCW data.

Strategy (two-phase fine-tuning):
  Phase 1 — Head only  (10 epochs, LR 5e-4)
    Freeze CNN + LSTM backbone, train only the classifier head.
    Quickly adapts the decision boundary to real data statistics.

  Phase 2 — Full fine-tune  (30 epochs, LR 5e-5)
    Unfreeze everything, low LR to gently adapt feature extractors
    without destroying the synthetic-data knowledge.

Saves:
  outputs/models/best_model_zenodo.pt   <- fine-tuned model (new)
  outputs/models/best_model.pt          <- original, NEVER touched

Usage
-----
    python finetune_zenodo.py
    python finetune_zenodo.py --phase1-epochs 15 --phase2-epochs 40
    python finetune_zenodo.py --fast   # quick smoke-test
"""

import argparse
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset import iq_to_range_doppler, iq_to_doppler_profile
from src.models.cnn_lstm import build_model, count_parameters
from src.training.metrics import (
    compute_metrics, plot_confusion_matrix, plot_training_history,
    print_classification_report,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE   = Path('data/real/zenodo_77ghz/data_SAAB_SIRS_77GHz_FMCW.npy')
SRC_MODEL   = Path('outputs/models/best_model.pt')          # never overwritten
DST_MODEL   = Path('outputs/models/best_model_zenodo.pt')   # fine-tuned output
LOG_DIR     = Path('outputs/logs')

CLASS_NAMES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']

ZENODO_CLASS_MAP = {
    'd1': 0, 'd2': 0, 'd3': 0, 'd4': 0, 'd5': 0, 'd6': 0,
    'seagull': 2, 'black-headed gull': 2, 'heron': 2,
    'pigeon': 2, 'raven': 2, 'gull': 2,
    'human_walk': 3, 'human_run': 3, 'human': 3,
    'cr': 3,
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading  (same logic as test_zenodo.py)
# ──────────────────────────────────────────────────────────────────────────────

def _label_from_string(s: str) -> int:
    s = s.strip().lower()
    if s in ZENODO_CLASS_MAP:
        return ZENODO_CLASS_MAP[s]
    for key, idx in ZENODO_CLASS_MAP.items():
        if key in s:
            return idx
    return -1


def load_zenodo(data_file: Path, max_samples: int = None):
    """
    Load Zenodo .npy, segment each recording into [32 x 128] CPIs.
    Returns (rd_maps list, labels array, filenames list).
    """
    PULSES     = 32
    RANGE_BINS = 128

    print(f"Loading {data_file.name} ...", flush=True)
    data = np.load(data_file, allow_pickle=True)
    print(f"  {data.shape[0]} recordings found")

    rd_maps, labels, fnames = [], [], []

    for row in range(data.shape[0]):
        if max_samples and len(rd_maps) >= max_samples:
            break

        cls_str = str(data[row, 0].flat[0])
        label   = _label_from_string(cls_str)
        if label < 0:
            continue

        iq_raw = data[row, 1]          # [1280 x N_pulses] complex128
        iq     = iq_raw.T              # [N_pulses x 1280]
        n_pulses, n_range = iq.shape

        # Centre-crop range to 128
        r_start = max(0, (n_range - RANGE_BINS) // 2)
        r_end   = r_start + RANGE_BINS
        if n_range < RANGE_BINS:
            pad = RANGE_BINS - n_range
            iq  = np.pad(iq, ((0, 0), (pad // 2, pad - pad // 2)))
            r_start, r_end = 0, RANGE_BINS
        iq = iq[:, r_start:r_end]      # [N x 128]

        # Segment into CPIs
        n_segs = max(1, n_pulses // PULSES)
        if n_pulses < PULSES:
            iq = np.pad(iq, ((0, PULSES - n_pulses), (0, 0)))
            n_segs = 1

        for seg in range(n_segs):
            if max_samples and len(rd_maps) >= max_samples:
                break
            cpi = iq[seg * PULSES:(seg + 1) * PULSES, :]
            rd  = iq_to_range_doppler(cpi, RANGE_BINS, PULSES)
            rd_maps.append(rd)
            labels.append(label)
            fnames.append(f"row{row}_{cls_str}_s{seg}")

    labels = np.array(labels, dtype=np.int64)
    print(f"  Extracted {len(rd_maps)} segments")
    counts = Counter(labels.tolist())
    for i, cls in enumerate(CLASS_NAMES):
        if i in counts:
            print(f"    {cls}: {counts[i]}")
    return rd_maps, labels, fnames


# ──────────────────────────────────────────────────────────────────────────────
# Feature -> tensor conversion
# ──────────────────────────────────────────────────────────────────────────────

def build_tensors(rd_maps, labels):
    """Convert RD maps to model-ready tensors."""
    specs = torch.stack([
        torch.from_numpy(rd).unsqueeze(0) for rd in rd_maps
    ])                                                     # [N, 1, 32, 128]
    dops  = torch.stack([
        torch.from_numpy(rd.mean(axis=1)) for rd in rd_maps
    ])                                                     # [N, 32]
    envs  = torch.zeros(len(rd_maps), 3, dtype=torch.float32)  # neutral
    lbls  = torch.tensor(labels, dtype=torch.long)
    return specs, dops, envs, lbls


def make_loader(specs, dops, envs, lbls, batch_size, shuffle=True, balanced=False):
    ds = TensorDataset(specs, dops, envs, lbls)
    sampler = None
    if balanced and shuffle:
        counts  = np.bincount(lbls.numpy(), minlength=len(CLASS_NAMES)).astype(float)
        counts  = np.where(counts == 0, 1.0, counts)
        w_cls   = 1.0 / counts
        w_samp  = torch.tensor(w_cls[lbls.numpy()], dtype=torch.double)
        sampler = WeightedRandomSampler(w_samp, len(w_samp), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler)


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum += val * n; self.count += n
    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    meter = AverageMeter()
    correct = total = 0
    for spec, dop, env, lbl in loader:
        spec, dop, env, lbl = spec.to(device), dop.to(device), env.to(device), lbl.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(spec, dop, env)
        loss   = criterion(logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        meter.update(loss.item(), spec.size(0))
        correct += logits.argmax(1).eq(lbl).sum().item()
        total   += lbl.size(0)
    return meter.avg, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    meter  = AverageMeter()
    preds, trues = [], []
    for spec, dop, env, lbl in loader:
        spec, dop, env, lbl = spec.to(device), dop.to(device), env.to(device), lbl.to(device)
        logits = model(spec, dop, env)
        meter.update(criterion(logits, lbl).item(), spec.size(0))
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(lbl.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    metrics = compute_metrics(trues, preds)
    return meter.avg, metrics['accuracy'] * 100.0, metrics, preds, trues


def run_phase(
    phase_name, model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    epochs, device, best_val_loss, patience=8
):
    """Generic training loop for one fine-tuning phase. Returns updated best_val_loss."""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    no_improve = 0

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        history['lr'].append(lr)

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_met, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        flag = ''
        if vl_loss < best_val_loss - 1e-4:
            best_val_loss = vl_loss
            no_improve = 0
            torch.save(model.state_dict(), DST_MODEL.with_suffix('.tmp'))
            flag = '  [BEST]'
        else:
            no_improve += 1

        print(f"  [{phase_name}] ep {epoch+1:3d}/{epochs}"
              f"  tr={tr_loss:.4f}/{tr_acc:.1f}%"
              f"  val={vl_loss:.4f}/{vl_acc:.1f}%"
              f"  F1={vl_met['f1_macro']:.4f}"
              f"  lr={lr:.1e}{flag}", flush=True)

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    return best_val_loss, history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1-epochs', type=int, default=20)
    parser.add_argument('--phase2-epochs', type=int, default=50)
    parser.add_argument('--batch',         type=int, default=32)
    parser.add_argument('--max-samples',   type=int, default=None)
    parser.add_argument('--fast',          action='store_true',
                        help='Smoke-test: 5+10 epochs')
    args = parser.parse_args()

    if args.fast:
        args.phase1_epochs = 5
        args.phase2_epochs = 10

    torch.manual_seed(42)
    np.random.seed(42)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DST_MODEL.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Sanity checks ──────────────────────────────────────────────────────────
    if not DATA_FILE.exists():
        print(f"Data not found: {DATA_FILE}")
        print("Run test_zenodo.py first to download the dataset.")
        return
    if not SRC_MODEL.exists():
        print(f"Source model not found: {SRC_MODEL}")
        print("Run train.py first.")
        return

    # ── Load Zenodo data ───────────────────────────────────────────────────────
    print("\n[1/4] Loading Zenodo real data ...")
    rd_maps, labels, _ = load_zenodo(DATA_FILE, args.max_samples)

    if len(rd_maps) == 0:
        print("No data loaded."); return

    # ── Split train / val / test ───────────────────────────────────────────────
    print("\n[2/4] Splitting data ...")
    idx = np.arange(len(labels))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, stratify=labels, random_state=42)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15 / 0.85, stratify=labels[idx_tv], random_state=42
    )
    print(f"  train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    def subset(indices):
        return ([rd_maps[i] for i in indices], labels[indices])

    specs_tr, dops_tr, envs_tr, lbls_tr = build_tensors(*subset(idx_train))
    specs_vl, dops_vl, envs_vl, lbls_vl = build_tensors(*subset(idx_val))
    specs_ts, dops_ts, envs_ts, lbls_ts = build_tensors(*subset(idx_test))

    train_loader = make_loader(specs_tr, dops_tr, envs_tr, lbls_tr,
                               args.batch, shuffle=True, balanced=True)
    val_loader   = make_loader(specs_vl, dops_vl, envs_vl, lbls_vl,
                               args.batch, shuffle=False)
    test_loader  = make_loader(specs_ts, dops_ts, envs_ts, lbls_ts,
                               args.batch, shuffle=False)

    # ── Load pre-trained model ─────────────────────────────────────────────────
    print(f"\n[3/4] Loading pre-trained model from {SRC_MODEL} ...")
    ckpt  = torch.load(SRC_MODEL, map_location=device, weights_only=False)
    model = build_model(ckpt.get('config', {})).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Original val_acc (synthetic): {ckpt.get('val_acc', '?')}%")

    # Class weights — only for classes present in training data.
    # Absent classes (Aircraft=1, Noise=4) get weight=0 so they never
    # inflate the loss or corrupt gradients.
    counts     = np.bincount(lbls_tr.numpy(), minlength=len(CLASS_NAMES)).astype(float)
    present    = counts > 0
    w_cls      = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    w_cls[present] = 1.0 / counts[present]
    # Renormalise weights of present classes so their mean = 1
    w_cls[present] /= w_cls[present].mean()
    w_cls_t = torch.tensor(w_cls, dtype=torch.float32).to(device)
    print(f"  Class weights: " +
          "  ".join(f"{CLASS_NAMES[i]}={w_cls[i]:.3f}" for i in range(len(CLASS_NAMES))))
    criterion = nn.CrossEntropyLoss(weight=w_cls_t, label_smoothing=0.05)

    # ── Baseline: model before fine-tuning ────────────────────────────────────
    print("\n  Baseline (before fine-tuning) on Zenodo test set:")
    _, base_acc, base_met, base_preds, base_trues = eval_epoch(
        model, test_loader, criterion, device
    )
    print(f"    Accuracy: {base_acc:.1f}%   F1-macro: {base_met['f1_macro']:.4f}")

    # ── Phase 1: Freeze backbone, train head only ──────────────────────────────
    print(f"\n[4/4] Fine-tuning ...")
    print(f"\n--- Phase 1: Head-only  ({args.phase1_epochs} epochs, LR=5e-4) ---")

    for name, param in model.named_parameters():
        # Freeze CNN, LSTM, env branches — only train classifier + skip_proj
        param.requires_grad = any(
            name.startswith(p) for p in ('fc1', 'fc2', 'out', 'skip_proj', 'drop')
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (head only): {trainable:,}")

    opt1   = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=1e-4
    )
    sched1 = CosineAnnealingLR(opt1, T_max=args.phase1_epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_val_loss, hist1 = run_phase(
        'P1', model, train_loader, val_loader,
        criterion, opt1, sched1,
        args.phase1_epochs, device, best_val_loss, patience=args.phase1_epochs
    )

    # ── Phase 2: Unfreeze all, low LR ─────────────────────────────────────────
    print(f"\n--- Phase 2: Full fine-tune  ({args.phase2_epochs} epochs, LR=5e-5) ---")

    for param in model.parameters():
        param.requires_grad = True

    print(f"  Trainable params (all): {count_parameters(model):,}")

    opt2   = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sched2 = CosineAnnealingLR(opt2, T_max=args.phase2_epochs, eta_min=1e-6)

    best_val_loss, hist2 = run_phase(
        'P2', model, train_loader, val_loader,
        criterion, opt2, sched2,
        args.phase2_epochs, device, best_val_loss, patience=12
    )

    # Load best weights saved during training
    tmp_path = DST_MODEL.with_suffix('.tmp')
    if tmp_path.exists():
        model.load_state_dict(torch.load(tmp_path, map_location=device, weights_only=True))
        tmp_path.unlink()

    # ── Save fine-tuned model ──────────────────────────────────────────────────
    torch.save({
        'model_state_dict': model.state_dict(),
        'config':           ckpt.get('config', {}),
        'fine_tuned_on':    'Zenodo 77GHz FMCW',
        'val_loss':         best_val_loss,
    }, DST_MODEL)
    print(f"\nFine-tuned model saved -> {DST_MODEL}")
    print(f"Original model intact  -> {SRC_MODEL}")

    # ── Final evaluation on test set ───────────────────────────────────────────
    print("\n" + "="*60)
    print(" Test Set Results")
    print("="*60)

    _, ft_acc, ft_met, ft_preds, ft_trues = eval_epoch(
        model, test_loader, criterion, device
    )

    print(f"\n  Before fine-tuning : acc={base_acc:.1f}%  F1={base_met['f1_macro']:.4f}")
    print(f"  After  fine-tuning : acc={ft_acc:.1f}%  F1={ft_met['f1_macro']:.4f}")
    print(f"  Improvement        : +{ft_acc - base_acc:.1f}% accuracy")

    print_classification_report(ft_trues, ft_preds, CLASS_NAMES)

    # Confusion matrix
    cm_path = str(LOG_DIR / 'zenodo_finetuned_confusion_matrix.png')
    plot_confusion_matrix(
        ft_met['confusion_matrix'], CLASS_NAMES,
        title=f'Zenodo Fine-tuned  (acc={ft_acc:.1f}%)',
        save_path=cm_path, normalize=True,
    )

    # Training curves (combined both phases)
    combined_history = {
        'train_loss': hist1['train_loss'] + hist2['train_loss'],
        'train_acc':  hist1['train_acc']  + hist2['train_acc'],
        'val_loss':   hist1['val_loss']   + hist2['val_loss'],
        'val_acc':    hist1['val_acc']    + hist2['val_acc'],
        'lr':         hist1['lr']         + hist2['lr'],
    }
    plot_training_history(
        combined_history,
        save_path=str(LOG_DIR / 'zenodo_finetune_history.png')
    )

    # Text report
    present = sorted(set(ft_trues.tolist()) | set(ft_preds.tolist()))
    names   = [CLASS_NAMES[i] for i in present]
    report  = classification_report(ft_trues, ft_preds, labels=present,
                                    target_names=names, zero_division=0)
    rpath   = LOG_DIR / 'zenodo_finetuned_report.txt'
    with open(rpath, 'w') as f:
        f.write("Zenodo 77GHz Fine-tuned Model\n")
        f.write(f"Before: acc={base_acc:.1f}%  F1={base_met['f1_macro']:.4f}\n")
        f.write(f"After : acc={ft_acc:.1f}%   F1={ft_met['f1_macro']:.4f}\n\n")
        f.write(report)
    print(f"\nReport saved        -> {rpath}")
    print(f"Confusion matrix    -> {cm_path}")
    print(f"Training curves     -> {LOG_DIR}/zenodo_finetune_history.png")
    print("\nDone.")


if __name__ == '__main__':
    main()
