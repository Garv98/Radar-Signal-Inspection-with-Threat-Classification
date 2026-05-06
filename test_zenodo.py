"""
Test trained RadarClassifier on the Zenodo 77 GHz FMCW Radar Dataset.

Dataset: "Radar Measurements on Drones, Birds and Humans with a 77GHz FMCW Sensor"
Source : https://zenodo.org/records/5845259
License: CC BY 4.0  (no registration required, direct download)

Class mapping to our 5-class taxonomy:
    Drone (any of 6 types) -> Drone   (0)
    Bird                   -> Bird    (2)
    Human                  -> Clutter (3)  [slow, distributed — closest match]

Usage
-----
    # Download + test (first run, ~1.6 GB)
    python test_zenodo.py

    # Skip download if already done
    python test_zenodo.py --no-download

    # Quick test on a subset
    python test_zenodo.py --no-download --max-samples 300
"""

import argparse
import sys
import os
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset import iq_to_range_doppler, iq_to_doppler_profile
from src.models.cnn_lstm import build_model
from src.training.metrics import (
    compute_metrics, plot_confusion_matrix, print_classification_report,
)

# ── Constants ──────────────────────────────────────────────────────────────────
ZENODO_RECORD_ID = '5845259'
ZENODO_API_URL   = f'https://zenodo.org/api/records/{ZENODO_RECORD_ID}'
DATA_DIR         = Path('data/real/zenodo_77ghz')
MODEL_PATH       = Path('outputs/models/best_model.pt')
CLASS_NAMES      = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']

# Dataset structure (discovered by inspection):
#   data.shape = (130, 6)   — 130 recordings x 6 columns
#   col[0]: class label string  e.g. 'D1'..'D6', 'seagull', 'human_walk', 'CR'
#   col[1]: IQ matrix  [1280 range_bins x N_pulses]  complex128
#   col[2]: range axis (N_pulses x 1) float64
#   col[3]: Doppler/time axis (N_pulses x 1) float64
#   col[4]: per-pulse numeric label (N_pulses x 1) uint8  {1,2,3}
#   col[5]: auxiliary flag (N_pulses x 1) uint8
#
# Class mapping to our 5-class taxonomy:
ZENODO_CLASS_MAP = {
    # 6 drone models -> Drone (0)
    'd1': 0, 'd2': 0, 'd3': 0, 'd4': 0, 'd5': 0, 'd6': 0,
    # Birds -> Bird (2)
    'seagull': 2, 'black-headed gull': 2, 'heron': 2,
    'pigeon': 2,  'raven': 2, 'gull': 2,
    # Humans -> Clutter (3)  [slow-moving, no exact class]
    'human_walk': 3, 'human_run': 3, 'human': 3,
    # CR = clutter/rain -> Clutter (3)
    'cr': 3,
}


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────

def download_zenodo(data_dir: Path):
    """Download all files from Zenodo record using the public API."""
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file list from Zenodo record {ZENODO_RECORD_ID} ...")
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    record = resp.json()

    files = record.get('files', [])
    if not files:
        # Try v2 API
        files = record.get('metadata', {}).get('related_identifiers', [])

    print(f"Found {len(files)} file(s) in record.")
    for f in files:
        fname  = f.get('key', f.get('filename', 'unknown'))
        url    = f.get('links', {}).get('self', '')
        size   = f.get('size', 0)
        dest   = data_dir / fname

        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {fname}  (already downloaded)")
            continue

        print(f"  Downloading {fname}  ({size/1e6:.1f} MB) ...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()

        with open(dest, 'wb') as out:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                out.write(chunk)
                downloaded += len(chunk)
                print(f"    {downloaded/1e6:.1f}/{size/1e6:.1f} MB", end='\r')
        print(f"  Saved -> {dest}")

    # Extract any zip files
    for zf in data_dir.glob('*.zip'):
        print(f"Extracting {zf.name} ...")
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(data_dir)
        print(f"  Extracted to {data_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Data loader — handles .npy Range-Doppler files
# ──────────────────────────────────────────────────────────────────────────────


def _label_from_string(s: str) -> int:
    """Map a Zenodo class string to our class index."""
    s = s.strip().lower()
    # Exact match first
    if s in ZENODO_CLASS_MAP:
        return ZENODO_CLASS_MAP[s]
    # Partial match
    for key, idx in ZENODO_CLASS_MAP.items():
        if key in s:
            return idx
    return -1  # unknown


def load_zenodo_data(data_dir: Path, max_samples: int = None):
    """
    Load the Zenodo 77GHz FMCW .npy file.

    Data structure (130 recordings x 6 columns):
      col[0]: class label string
      col[1]: IQ matrix  [1280 range_bins x N_pulses]  complex128

    Strategy:
      1. Transpose IQ -> [N_pulses x 1280]
      2. Segment into non-overlapping 32-pulse windows
      3. For each window, centre-crop range to 128 bins
      4. Compute Range-Doppler map [32 x 128]
      5. Label comes from col[0] string
    """
    PULSES   = 32    # CPI length (must match training)
    RANGE_BINS = 128  # range bins (must match training)

    npy_files = sorted(data_dir.rglob('*.npy'))
    if not npy_files:
        print(f"No .npy files found under {data_dir}")
        return [], np.array([], dtype=np.int64), []

    print(f"Found {len(npy_files)} .npy file(s)")

    # Load main data file (allow_pickle for object arrays)
    data_file = npy_files[0]
    print(f"Loading {data_file.name} ...")
    data = np.load(data_file, allow_pickle=True)
    print(f"  Array shape: {data.shape}  ({data.shape[0]} recordings)")

    rd_maps, labels, filenames = [], [], []

    for row_idx in range(data.shape[0]):
        if max_samples and len(rd_maps) >= max_samples:
            break

        # Class label
        cls_str = str(data[row_idx, 0].flat[0])
        label   = _label_from_string(cls_str)

        # IQ matrix: [1280 range_bins x N_pulses]
        iq_raw = data[row_idx, 1]                    # complex128 [R x N]
        if iq_raw.ndim != 2:
            print(f"  [skip] row {row_idx} unexpected IQ dim {iq_raw.ndim}")
            continue

        # Transpose -> [N_pulses x R]
        iq = iq_raw.T                                 # [N x 1280]
        n_pulses, n_range = iq.shape

        # Centre-crop range to RANGE_BINS (1280 -> 128)
        r_start = (n_range - RANGE_BINS) // 2
        r_end   = r_start + RANGE_BINS
        if r_start < 0:
            # If fewer range bins than needed, zero-pad
            pad = RANGE_BINS - n_range
            iq  = np.pad(iq, ((0, 0), (pad//2, pad - pad//2)))
            r_start, r_end = 0, RANGE_BINS
        iq_crop = iq[:, r_start:r_end]               # [N x 128]

        # Segment into non-overlapping 32-pulse CPIs
        n_segs = n_pulses // PULSES
        if n_segs == 0:
            # Too short — zero-pad to PULSES
            pad_len = PULSES - n_pulses
            iq_crop = np.pad(iq_crop, ((0, pad_len), (0, 0)))
            n_segs  = 1

        for seg in range(n_segs):
            if max_samples and len(rd_maps) >= max_samples:
                break
            cpi = iq_crop[seg * PULSES:(seg + 1) * PULSES, :]  # [32 x 128]
            rd  = iq_to_range_doppler(cpi, range_fft_size=RANGE_BINS,
                                      doppler_fft_size=PULSES)  # [32 x 128]
            rd_maps.append(rd)
            labels.append(label)
            filenames.append(f"row{row_idx}_{cls_str}_seg{seg}")

        print(f"  row {row_idx:3d}: {cls_str:35s}  "
              f"IQ={iq_raw.shape}  segs={n_segs}  label={label}",
              flush=True)

    return rd_maps, np.array(labels, dtype=np.int64), filenames


# ──────────────────────────────────────────────────────────────────────────────
# Inference — works on pre-computed RD maps (not raw IQ)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, rd_maps, device, batch_size=64):
    """
    rd_maps: list of float32 [32, 128] Range-Doppler maps (already normalised).
    Returns predictions, confidences, probabilities.
    """
    model.eval()
    all_preds, all_confs, all_probs = [], [], []

    for start in range(0, len(rd_maps), batch_size):
        batch = rd_maps[start:start + batch_size]

        specs, dops, envs = [], [], []
        for rd in batch:
            spec = torch.from_numpy(rd).unsqueeze(0)         # [1, 32, 128]
            dop  = torch.from_numpy(rd.mean(axis=1))         # [32] Doppler profile
            env  = torch.zeros(3, dtype=torch.float32)       # neutral env
            specs.append(spec)
            dops.append(dop)
            envs.append(env)

        spec_t = torch.stack(specs).to(device)
        dop_t  = torch.stack(dops).to(device)
        env_t  = torch.stack(envs).to(device)

        logits = model(spec_t, dop_t, env_t)
        probs  = F.softmax(logits, dim=1).cpu().numpy()

        all_preds.extend(probs.argmax(axis=1))
        all_confs.extend(probs.max(axis=1))
        all_probs.extend(probs)

        done = min(start + batch_size, len(rd_maps))
        print(f"  Inference: {done}/{len(rd_maps)}", end='\r', flush=True)

    print()
    return np.array(all_preds), np.array(all_confs), np.array(all_probs)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Test model on Zenodo 77GHz FMCW dataset')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip download (data already present)')
    parser.add_argument('--data-dir',    default=str(DATA_DIR))
    parser.add_argument('--model',       default=str(MODEL_PATH))
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--save-dir',    default='outputs/logs')
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    model_path = Path(args.model)
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ───────────────────────────────────────────────────────────────
    if not args.no_download:
        print("="*60)
        print(" Downloading Zenodo 77GHz FMCW Radar Dataset")
        print(" https://zenodo.org/records/5845259")
        print("="*60)
        download_zenodo(data_dir)
    else:
        print(f"Skipping download. Using data from {data_dir}")

    # ── Load model ─────────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run train.py first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Loading model from {model_path} ...")
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(ckpt.get('config', {})).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded (epoch {ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')}%)")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading .npy files from {data_dir} ...")
    rd_maps, labels, filenames = load_zenodo_data(data_dir, args.max_samples)

    if len(rd_maps) == 0:
        print("\nNo data loaded.")
        print("Expected .npy files under:", data_dir)
        print("Check the dataset was downloaded correctly.")
        return

    print(f"\nLoaded {len(rd_maps)} samples")
    labelled_mask = labels >= 0
    n_labelled    = labelled_mask.sum()
    print(f"  Labelled   : {n_labelled}")
    print(f"  Unlabelled : {len(labels) - n_labelled}")
    for i, cls in enumerate(CLASS_NAMES):
        count = (labels[labelled_mask] == i).sum()
        if count:
            print(f"    {cls}: {count}")

    # ── Run inference ──────────────────────────────────────────────────────────
    print("\nRunning inference ...")
    preds, confs, probs = run_inference(model, rd_maps, device)

    # ── Results ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Results")
    print("="*60)

    if n_labelled == 0:
        print("No labelled samples — showing prediction distribution:\n")
        for i, cls in enumerate(CLASS_NAMES):
            count = (preds == i).sum()
            pct   = 100.0 * count / max(len(preds), 1)
            bar   = '#' * int(pct / 2)
            print(f"  {cls:10s}: {count:5d}  ({pct:5.1f}%)  {bar}")
    else:
        lbl_true  = labels[labelled_mask]
        lbl_pred  = preds[labelled_mask]
        lbl_confs = confs[labelled_mask]

        metrics = compute_metrics(lbl_true, lbl_pred, CLASS_NAMES)
        print(f"\n  Accuracy  : {metrics['accuracy']*100:.2f}%")
        print(f"  F1-macro  : {metrics['f1_macro']:.4f}")
        print(f"  Avg conf  : {lbl_confs.mean()*100:.1f}%")

        print_classification_report(lbl_true, lbl_pred, CLASS_NAMES)

        print("\n  Per-class detail:")
        for i, cls in enumerate(CLASS_NAMES):
            mask = lbl_true == i
            if not mask.any():
                continue
            recall = (lbl_pred[mask] == i).mean() * 100
            conf   = lbl_confs[mask].mean() * 100
            print(f"    {cls:10s}  n={mask.sum():5d}  "
                  f"recall={recall:5.1f}%  avg_conf={conf:5.1f}%")

        # Save confusion matrix
        cm_path = str(save_dir / 'zenodo_confusion_matrix.png')
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            CLASS_NAMES,
            title=f'Zenodo 77GHz Real Data  (acc={metrics["accuracy"]*100:.1f}%)',
            save_path=cm_path,
            normalize=True,
        )

        # Save report
        from sklearn.metrics import classification_report as skl_report
        present = sorted(set(lbl_true.tolist()) | set(lbl_pred.tolist()))
        names   = [CLASS_NAMES[i] for i in present]
        report  = skl_report(lbl_true, lbl_pred, labels=present,
                              target_names=names, zero_division=0)
        rpath  = save_dir / 'zenodo_report.txt'
        with open(rpath, 'w') as f:
            f.write("Zenodo 77GHz FMCW Real Data Test\n")
            f.write(f"Accuracy : {metrics['accuracy']*100:.2f}%\n")
            f.write(f"F1-macro : {metrics['f1_macro']:.4f}\n\n")
            f.write(report)

        print(f"\n  Report saved -> {rpath}")
        print(f"  Confusion matrix -> {cm_path}")

    # ── Sample predictions ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Sample Predictions (first 15)")
    print("="*60)
    print(f"  {'File':>30}  {'True':>10}  {'Pred':>10}  {'Conf':>6}  Status")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*6}  ------")
    for i in range(min(15, len(rd_maps))):
        true_name = CLASS_NAMES[labels[i]] if labels[i] >= 0 else 'unknown'
        pred_name = CLASS_NAMES[preds[i]]
        status    = 'OK' if labels[i] == preds[i] else ('MISS' if labels[i] >= 0 else '-')
        fname     = filenames[i][-30:] if len(filenames[i]) > 30 else filenames[i]
        print(f"  {fname:>30}  {true_name:>10}  {pred_name:>10}  "
              f"{confs[i]*100:5.1f}%  {status}")

    print("\nDone.")


if __name__ == '__main__':
    main()
