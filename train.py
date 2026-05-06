"""
Radar Threat Classifier - End-to-End Training Script

Usage
-----
  python train.py                          # default: 2000 samples/class, 80 epochs
  python train.py --samples 5000 --epochs 120
  python train.py --samples 1000 --fast   # quick smoke-test

Outputs (in outputs/models/ and outputs/logs/)
-------
  best_model.pt          - best checkpoint (val loss)
  final_model.pt         - final checkpoint
  training_history.png   - loss / accuracy / LR curves
  confusion_matrix.png   - normalised confusion matrix (test set)
  classification_report.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# -- project imports ------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from src.data.synthetic_generator import SyntheticRadarGenerator
from src.data.dataset import build_datasets
from src.models.cnn_lstm import RadarClassifier, build_model, count_parameters
from src.training.metrics import (
    compute_metrics, plot_confusion_matrix, plot_training_history,
    print_classification_report,
)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def load_config(path: str = 'configs/config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing (better calibration)."""

    def __init__(self, smoothing: float = 0.10, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w

        return loss.mean()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


# ------------------------------------------------------------------------------
# Data generation
# ------------------------------------------------------------------------------

def generate_data(config: dict, samples_per_class: int, seed: int = 42):
    """Generate synthetic dataset and return IQ matrices, labels, metadata."""
    radar_cfg = config.get('radar', {})
    gen = SyntheticRadarGenerator(
        carrier_freq=float(radar_cfg.get('carrier_frequency', 24e9)),
        sampling_rate=float(radar_cfg.get('sampling_rate', 10000)),
        prf=float(radar_cfg.get('prf', 1000)),
        num_pulses=int(radar_cfg.get('num_pulses', 32)),
        num_samples=int(radar_cfg.get('num_samples', 128)),
        seed=seed,
    )

    classes = SyntheticRadarGenerator.CLASSES
    iq_all, labels_all, meta_all = [], [], []

    for cls in classes:
        print(f"  Generating {samples_per_class} x {cls} ...", flush=True)
        for _ in range(samples_per_class):
            iq, meta = gen.generate_sample(cls)
            iq_all.append(iq)
            labels_all.append(meta['label'])
            meta_all.append(meta)

    # Shuffle
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(iq_all))
    iq_all     = [iq_all[i]     for i in perm]
    labels_all = [labels_all[i] for i in perm]
    meta_all   = [meta_all[i]   for i in perm]

    print(f"  Total: {len(iq_all)} samples  "
          f"({samples_per_class} per class x {len(classes)} classes)")
    return iq_all, labels_all, meta_all


# ------------------------------------------------------------------------------
# Training / validation loops
# ------------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, total_epochs,
    grad_clip: float = 1.0,
):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    for batch_idx, (spec, dop, env, lbl) in enumerate(loader):
        spec, dop, env, lbl = (
            spec.to(device), dop.to(device), env.to(device), lbl.to(device)
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(spec, dop, env)
        loss   = criterion(logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_meter.update(loss.item(), spec.size(0))
        preds    = logits.argmax(dim=1)
        correct += preds.eq(lbl).sum().item()
        total   += lbl.size(0)

        if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
            print(
                f"  [{epoch+1:3d}/{total_epochs}]"
                f" step {batch_idx+1}/{len(loader)}"
                f"  loss={loss_meter.avg:.4f}"
                f"  acc={100.*correct/total:.1f}%",
                flush=True,
            )

    return loss_meter.avg, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for spec, dop, env, lbl in loader:
        spec, dop, env, lbl = (
            spec.to(device), dop.to(device), env.to(device), lbl.to(device)
        )
        logits = model(spec, dop, env)
        loss   = criterion(logits, lbl)
        loss_meter.update(loss.item(), spec.size(0))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics    = compute_metrics(all_labels, all_preds)
    return loss_meter.avg, metrics['accuracy'] * 100.0, metrics


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Radar Threat Classifier')
    parser.add_argument('--config',   default='configs/config.yaml')
    parser.add_argument('--samples',  type=int,   default=2000,
                        help='Samples per class (default 2000 -> 10 000 total)')
    parser.add_argument('--epochs',   type=int,   default=80)
    parser.add_argument('--batch',    type=int,   default=64)
    parser.add_argument('--lr',       type=float, default=5e-4)
    parser.add_argument('--workers',  type=int,   default=0)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--fast',     action='store_true',
                        help='Quick smoke-test: 300 samples/class, 10 epochs')
    args = parser.parse_args()

    if args.fast:
        args.samples = 300
        args.epochs  = 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- Config ----------------------------------------------------------------
    config = load_config(args.config)

    out_dir   = Path(config.get('paths', {}).get('models', 'outputs/models'))
    log_dir   = Path(config.get('paths', {}).get('logs',   'outputs/logs'))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\n{'='*60}")
    print(f" Radar Threat Classifier - Training")
    print(f"{'='*60}")
    print(f" Device  : {device}")
    print(f" Samples : {args.samples}/class  ({args.samples * 5} total)")
    print(f" Epochs  : {args.epochs}")
    print(f" Batch   : {args.batch}")

    # -- Generate synthetic data -----------------------------------------------
    print("\n[1/4] Generating synthetic radar data ...")
    t0 = time.time()
    iq_matrices, labels, metadata = generate_data(config, args.samples, args.seed)
    print(f"  Done in {time.time()-t0:.1f}s")

    # -- Build datasets --------------------------------------------------------
    print("\n[2/4] Building train / val / test datasets ...")
    feat_cfg = config.get('features', {})
    train_ds, val_ds, test_ds = build_datasets(
        iq_matrices, labels, metadata,
        train_frac=float(config.get('dataset', {}).get('train_split', 0.70)),
        val_frac=float(config.get('dataset', {}).get('val_split', 0.15)),
        seed=args.seed,
        range_fft=int(feat_cfg.get('spectrogram_size', [32, 128])[1]),
        doppler_fft=int(feat_cfg.get('spectrogram_size', [32, 128])[0]),
    )

    # Use weighted sampler for balanced batches
    sampler = train_ds.get_weighted_sampler()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        sampler=sampler, num_workers=args.workers, pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch,
        shuffle=False, num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch,
        shuffle=False, num_workers=args.workers,
    )
    print(f"  Loaders: {len(train_loader)} train | {len(val_loader)} val | {len(test_loader)} test batches")

    # -- Build model -----------------------------------------------------------
    print("\n[3/4] Building model ...")
    model = build_model(config).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # Class weights for loss
    class_weights = train_ds.get_class_weights().to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.08, weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, amsgrad=True,
    )

    # Warmup 5 epochs then cosine anneal
    warmup_epochs = min(5, args.epochs // 8)
    warmup_sched  = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler     = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    # -- Training loop ---------------------------------------------------------
    print(f"\n[4/4] Training for {args.epochs} epochs ...\n")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    best_val_loss = float('inf')
    best_val_acc  = 0.0
    patience      = max(15, args.epochs // 5)
    no_improve    = 0

    t_train = time.time()
    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]['lr']
        history['lr'].append(lr_now)

        # Train
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs,
        )
        # Validate
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        flag = ''
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            no_improve    = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'metrics': val_metrics,
                'config': config,
            }, out_dir / 'best_model.pt')
            flag = '  [BEST]'
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}"
            f"  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.1f}%"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%"
            f"  F1={val_metrics['f1_macro']:.4f}"
            f"  lr={lr_now:.2e}"
            f"{flag}",
            flush=True,
        )

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, out_dir / 'final_model.pt')

    elapsed = time.time() - t_train
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val loss={best_val_loss:.4f}  acc={best_val_acc:.1f}%")

    # -- Test evaluation -------------------------------------------------------
    print("\n" + "="*60)
    print(" Test Set Evaluation (loading best model)")
    print("="*60)

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    CLASS_NAMES = SyntheticRadarGenerator.CLASSES

    # Full test pass — collect predictions and labels
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for spec, dop, env, lbl in test_loader:
            spec, dop, env = spec.to(device), dop.to(device), env.to(device)
            logits = model(spec, dop, env)
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_l.extend(lbl.numpy())
    all_p = np.array(all_p)
    all_l = np.array(all_l)
    final_metrics = compute_metrics(all_l, all_p, CLASS_NAMES)

    print_classification_report(all_l, all_p, CLASS_NAMES)

    print(f"\n Test Accuracy : {final_metrics['accuracy']*100:.2f}%")
    print(f" Test F1-macro : {final_metrics['f1_macro']:.4f}")
    print(f" Test F1-weight: {final_metrics['f1_weighted']:.4f}")

    # Save classification report
    from sklearn.metrics import classification_report
    report = classification_report(all_l, all_p, target_names=CLASS_NAMES, zero_division=0)
    report_path = log_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy : {final_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Test F1-macro : {final_metrics['f1_macro']:.4f}\n\n")
        f.write(report)
    print(f"\nClassification report saved -> {report_path}")

    # -- Plots -----------------------------------------------------------------
    print("\nGenerating plots ...")

    # Training curves
    hist_path = str(log_dir / 'training_history.png')
    plot_training_history(history, save_path=hist_path)

    # Confusion matrix
    cm_path = str(log_dir / 'confusion_matrix.png')
    plot_confusion_matrix(
        final_metrics['confusion_matrix'],
        CLASS_NAMES,
        title=f'Test Confusion Matrix  (acc={final_metrics["accuracy"]*100:.1f}%)',
        save_path=cm_path,
        normalize=True,
    )

    print(f"Plots saved -> {log_dir}/")
    print("\nAll done.")


if __name__ == '__main__':
    main()
