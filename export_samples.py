"""
Export real IQ segments from Zenodo dataset as individual .npy files
ready to upload into the Streamlit UI.

Each exported file:
  - Shape  : (32, 128)  complex128
  - Format : numpy .npy
  - Name   : <class>_<recording>_seg<n>.npy

Usage
-----
    python export_samples.py               # exports 5 samples per class
    python export_samples.py --n 10        # 10 per class
    python export_samples.py --out my_dir  # custom output folder
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

DATA_FILE  = Path('data/real/zenodo_77ghz/data_SAAB_SIRS_77GHz_FMCW.npy')
PULSES     = 32
RANGE_BINS = 128

ZENODO_CLASS_MAP = {
    'd1': 'Drone', 'd2': 'Drone', 'd3': 'Drone',
    'd4': 'Drone', 'd5': 'Drone', 'd6': 'Drone',
    'seagull': 'Bird', 'black-headed gull': 'Bird',
    'heron': 'Bird', 'pigeon': 'Bird', 'raven': 'Bird', 'gull': 'Bird',
    'human_walk': 'Human', 'human_run': 'Human',
    'cr': 'Clutter',
}

def label_from_string(s: str) -> str:
    s = s.strip().lower()
    if s in ZENODO_CLASS_MAP:
        return ZENODO_CLASS_MAP[s]
    for key, cls in ZENODO_CLASS_MAP.items():
        if key in s:
            return cls
    return 'Unknown'


def extract_segments(data_file: Path, n_per_class: int, out_dir: Path):
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        print("Run test_zenodo.py first to download the dataset.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {data_file.name} ...")
    data = np.load(data_file, allow_pickle=True)

    saved    = defaultdict(int)
    exported = []

    for row in range(data.shape[0]):
        cls_str  = str(data[row, 0].flat[0])
        cls_name = label_from_string(cls_str)

        if saved[cls_name] >= n_per_class:
            continue

        iq_raw   = data[row, 1]         # [1280 x N_pulses]
        iq       = iq_raw.T             # [N_pulses x 1280]
        n_pulses, n_range = iq.shape

        # Centre-crop range to 128 bins
        r_start = max(0, (n_range - RANGE_BINS) // 2)
        r_end   = r_start + RANGE_BINS
        if n_range < RANGE_BINS:
            pad = RANGE_BINS - n_range
            iq  = np.pad(iq, ((0, 0), (pad // 2, pad - pad // 2)))
            r_start, r_end = 0, RANGE_BINS
        iq = iq[:, r_start:r_end]       # [N x 128]

        n_segs = max(1, n_pulses // PULSES)

        for seg in range(n_segs):
            if saved[cls_name] >= n_per_class:
                break

            cpi = iq[seg * PULSES:(seg + 1) * PULSES, :]   # [32 x 128] complex
            if cpi.shape != (PULSES, RANGE_BINS):
                continue

            safe_cls = cls_str.replace(' ', '_').replace('-', '_')
            fname    = f"{cls_name}_{safe_cls}_r{row}_s{seg}.npy"
            fpath    = out_dir / fname
            np.save(fpath, cpi)
            saved[cls_name] += 1
            exported.append((cls_name, fpath))

    print(f"\nExported {len(exported)} files to: {out_dir.resolve()}\n")
    by_class = defaultdict(list)
    for cls, fp in exported:
        by_class[cls].append(fp.name)

    for cls, files in sorted(by_class.items()):
        print(f"  {cls} ({len(files)} files):")
        for f in files:
            print(f"    {f}")

    print(f"\nHow to use in the UI:")
    print(f"  1. Open http://localhost:8501")
    print(f"  2. In the sidebar, scroll to 'Upload IQ Data'")
    print(f"  3. Upload any .npy file from: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',   type=int, default=5,
                        help='Samples to export per class (default 5)')
    parser.add_argument('--out', default='data/real/upload_samples',
                        help='Output directory')
    args = parser.parse_args()

    extract_segments(DATA_FILE, args.n, Path(args.out))


if __name__ == '__main__':
    main()
