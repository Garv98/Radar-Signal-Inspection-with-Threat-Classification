"""
PyTorch Dataset for Radar Threat Classification

Converts raw IQ matrices from the synthetic generator into:
  • spectrogram  — Range-Doppler map  [1 × 32 × 128]  → CNN branch
  • doppler_seq  — 1-D Doppler profile [32]             → LSTM branch
  • env_features — normalised env. vector [3]            → FC branch

Feature extraction (Range-Doppler map):
  1. FFT along fast-time axis (range processing)   → [32 × 128]
  2. FFT along slow-time axis (Doppler processing)  → [32 × 128]
  3. fftshift + log10-magnitude (dB scale)
  4. Min-Max normalisation to [0, 1]

Doppler profile:
  - Mean of Range-Doppler map across range bins → [32]
  - Captures JEM sidebands, rotor harmonics, wing-beat, etc.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.fft import fft, fftshift
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List, Callable


# ──────────────────────────────────────────────────────────────────────────────
# Low-level feature extraction (pure NumPy, no torch dependency)
# ──────────────────────────────────────────────────────────────────────────────

def iq_to_range_doppler(
    iq: np.ndarray,
    range_fft_size: int = 128,
    doppler_fft_size: int = 32,
) -> np.ndarray:
    """
    Compute normalised Range-Doppler map from a complex IQ matrix.

    Args:
        iq: Complex IQ matrix [num_pulses × num_samples]
        range_fft_size:   FFT size along fast-time  (range dimension)
        doppler_fft_size: FFT size along slow-time  (Doppler dimension)

    Returns:
        float32 array [doppler_fft_size × range_fft_size] in [0, 1]
    """
    # 1. Range FFT  (fast-time / samples axis)
    range_fft = fft(iq, n=range_fft_size, axis=1)           # [P × R]
    # 2. Doppler FFT (slow-time / pulses axis)
    rd = fft(range_fft, n=doppler_fft_size, axis=0)          # [D × R]
    # 3. Centre zero-Doppler and compute dB magnitude
    rd = fftshift(rd, axes=(0, 1))
    rd_db = 20.0 * np.log10(np.abs(rd) + 1e-10)             # [D × R]
    # 4. Normalise to [0, 1]
    rd_min, rd_max = rd_db.min(), rd_db.max()
    if rd_max > rd_min:
        rd_db = (rd_db - rd_min) / (rd_max - rd_min)
    else:
        rd_db = np.zeros_like(rd_db)
    return rd_db.astype(np.float32)


def iq_to_doppler_profile(rd_map: np.ndarray) -> np.ndarray:
    """
    1-D Doppler profile = mean across range bins of the Range-Doppler map.
    Input:  rd_map  [D × R] normalised float32
    Output: [D]     float32
    """
    return rd_map.mean(axis=1).astype(np.float32)


def _normalize_env(metadata: Dict) -> np.ndarray:
    """Normalise environmental features to [0, 1]."""
    env   = metadata.get('environmental', {})
    rain  = float(env.get('rain',        0.0))   / 20.0
    temp  = (float(env.get('temperature', 20.0)) + 10.0) / 50.0
    pres  = (float(env.get('pressure',  1013.0)) - 980.0) / 60.0
    return np.clip(np.array([rain, temp, pres], dtype=np.float32), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# In-memory Dataset (used by train.py)
# ──────────────────────────────────────────────────────────────────────────────

class InMemoryRadarDataset(Dataset):
    """
    Holds pre-computed features in RAM for fast mini-batch access.

    This is the recommended dataset class.  All IQ → feature conversion
    happens in __init__ so __getitem__ is very fast during training.
    """

    CLASS_NAMES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']

    def __init__(
        self,
        iq_matrices: List[np.ndarray],
        labels: List[int],
        metadata: List[Dict],
        augment: bool = False,
        range_fft: int = 128,
        doppler_fft: int = 32,
    ):
        assert len(iq_matrices) == len(labels) == len(metadata), \
            "iq_matrices, labels, and metadata must all have the same length"

        self.labels   = np.array(labels, dtype=np.int64)
        self.metadata = metadata
        self.augment  = augment
        self.range_fft   = range_fft
        self.doppler_fft = doppler_fft

        # Pre-compute features once
        self.spectrograms: List[np.ndarray] = []
        self.doppler_seqs: List[np.ndarray] = []
        self.env_feats:    List[np.ndarray] = []

        for iq, meta in zip(iq_matrices, metadata):
            rd  = iq_to_range_doppler(iq, range_fft, doppler_fft)
            dop = iq_to_doppler_profile(rd)
            env = _normalize_env(meta)
            self.spectrograms.append(rd)
            self.doppler_seqs.append(dop)
            self.env_feats.append(env)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = torch.from_numpy(self.spectrograms[idx]).unsqueeze(0)  # [1, D, R]
        dop  = torch.from_numpy(self.doppler_seqs[idx])               # [D]
        env  = torch.from_numpy(self.env_feats[idx])                  # [3]
        lbl  = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        if self.augment:
            spec, dop = _augment_rd(spec, dop)

        return spec, dop, env, lbl

    # ── Class balance helpers ──────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = np.bincount(self.labels, minlength=len(self.CLASS_NAMES)).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        w = w / w.sum() * len(self.CLASS_NAMES)
        return torch.tensor(w, dtype=torch.float32)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        class_w   = self.get_class_weights().numpy()
        sample_w  = class_w[self.labels]
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).double(),
            num_samples=len(self.labels),
            replacement=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def _augment_rd(
    spec: torch.Tensor,   # [1, D, R]
    dop:  torch.Tensor,   # [D]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SpecAugment-inspired augmentation on the Range-Doppler map:
      • Circular shift in Doppler and range directions
      • Random frequency (Doppler) masking — up to D/6 bins zeroed
      • Random range masking                — up to R/8 bins zeroed
      • Amplitude scaling ∈ [0.8, 1.2]
      • Small additive Gaussian noise
    """
    D, R = spec.shape[1], spec.shape[2]

    # Circular shifts
    d_shift = int(np.random.randint(-3, 4))
    r_shift = int(np.random.randint(-5, 6))
    spec = torch.roll(spec, d_shift, dims=1)
    spec = torch.roll(spec, r_shift, dims=2)
    dop  = torch.roll(dop,  d_shift, dims=0)

    # Doppler-frequency mask
    if np.random.random() < 0.5:
        mlen  = int(np.random.randint(1, max(2, D // 6)))
        mstart = int(np.random.randint(0, D - mlen))
        spec[0, mstart:mstart + mlen, :] = 0.0
        dop[mstart:mstart + mlen] = 0.0

    # Range mask
    if np.random.random() < 0.5:
        mlen  = int(np.random.randint(1, max(2, R // 8)))
        mstart = int(np.random.randint(0, R - mlen))
        spec[0, :, mstart:mstart + mlen] = 0.0

    # Amplitude scaling
    scale = float(np.random.uniform(0.8, 1.2))
    spec  = spec * scale

    # Additive noise
    spec = spec + 0.015 * torch.randn_like(spec)
    spec = spec.clamp(0.0, 1.0)

    return spec, dop


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build train / val / test splits
# ──────────────────────────────────────────────────────────────────────────────

def build_datasets(
    iq_matrices: List[np.ndarray],
    labels: List[int],
    metadata: List[Dict],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
    range_fft:  int   = 128,
    doppler_fft: int  = 32,
) -> Tuple[InMemoryRadarDataset, InMemoryRadarDataset, InMemoryRadarDataset]:
    """
    Stratified split into train / val / test InMemoryRadarDatasets.
    """
    labels_arr = np.array(labels)
    idx        = np.arange(len(labels_arr))

    test_frac  = 1.0 - train_frac - val_frac
    idx_tv, idx_test = train_test_split(
        idx, test_size=test_frac,
        stratify=labels_arr, random_state=seed,
    )
    relative_val = val_frac / (train_frac + val_frac)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=relative_val,
        stratify=labels_arr[idx_tv], random_state=seed,
    )

    print(f"Split: train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    def _make(indices, augment):
        label = 'train' if augment else 'eval'
        print(f"  Building {label} dataset ({len(indices)} samples) ...")
        return InMemoryRadarDataset(
            [iq_matrices[i] for i in indices],
            labels_arr[indices].tolist(),
            [metadata[i]     for i in indices],
            augment=augment,
            range_fft=range_fft,
            doppler_fft=doppler_fft,
        )

    return _make(idx_train, True), _make(idx_val, False), _make(idx_test, False)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy disk-based Dataset (kept for backward compatibility with old train loops)
# ──────────────────────────────────────────────────────────────────────────────

class RadarDataset(Dataset):
    """
    Disk-based dataset supporting synthetic pkl files and MAFAT format.

    For new training pipelines, prefer InMemoryRadarDataset + build_datasets().
    This class is kept for backward compatibility.
    """

    CLASS_NAMES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        config: Optional[dict] = None,
        precomputed: bool = False,
    ):
        self.data_dir    = data_dir
        self.split       = split
        self.transform   = transform
        self.precomputed = precomputed
        self.config      = config or {}
        self.use_environmental = self.config.get('features', {}).get('use_environmental', True)
        self.range_fft   = 128
        self.doppler_fft = 32

        self.samples:     List = []
        self.labels:      List = []
        self.metadata:    List = []
        self.env_features: List = []

        self._load_data()

    def _load_data(self):
        synthetic_path = os.path.join(self.data_dir, 'synthetic', 'synthetic_dataset.pkl')
        if os.path.exists(synthetic_path):
            self._load_synthetic(synthetic_path)
        else:
            print(f"No data found in {self.data_dir}. Run data generation first.")

    def _load_synthetic(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        all_samples  = data['samples']
        all_labels   = data['labels']
        all_metadata = data['metadata']

        idx = np.arange(len(all_samples))
        tr, tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=all_labels)
        vl, ts  = train_test_split(tmp, test_size=0.50, random_state=42, stratify=all_labels[tmp])

        split_idx = {'train': tr, 'val': vl, 'test': ts}[self.split]

        self.samples  = all_samples[split_idx]
        self.labels   = all_labels[split_idx]
        self.metadata = [all_metadata[i] for i in split_idx]

        for meta in self.metadata:
            env = meta.get('environmental', {})
            self.env_features.append([
                env.get('rain', 0.0),
                env.get('temperature', 20.0),
                env.get('pressure', 1013.0),
            ])
        self.env_features = np.array(self.env_features, dtype=np.float32)
        print(f"Loaded {len(self.samples)} '{self.split}' samples from synthetic data.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        iq  = self.samples[idx]
        lbl = self.labels[idx]
        env = self.env_features[idx] if len(self.env_features) else np.zeros(3, np.float32)

        rd  = iq_to_range_doppler(iq, self.range_fft, self.doppler_fft)
        dop = iq_to_doppler_profile(rd)

        if self.transform is not None:
            rd, dop, env = self.transform(rd, dop, env)

        env_norm = np.clip(np.array([
            env[0] / 20.0,
            (env[1] + 10.0) / 50.0,
            (env[2] - 980.0) / 60.0,
        ], dtype=np.float32), 0.0, 1.0)

        spec_t = torch.FloatTensor(rd).unsqueeze(0)
        dop_t  = torch.FloatTensor(dop)
        env_t  = torch.FloatTensor(env_norm)
        lbl_t  = torch.tensor(int(lbl), dtype=torch.long)

        return spec_t, dop_t, env_t, lbl_t

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=len(self.CLASS_NAMES)).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w = len(self.labels) / (len(self.CLASS_NAMES) * counts)
        return torch.tensor(w, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy augmentation transform (kept for backward compat)
# ──────────────────────────────────────────────────────────────────────────────

class RadarDataAugmentation:
    def __init__(
        self,
        time_shift:      float = 0.1,
        freq_shift:      float = 0.05,
        noise_level:     float = 0.05,
        amplitude_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.time_shift      = time_shift
        self.freq_shift      = freq_shift
        self.noise_level     = noise_level
        self.amplitude_range = amplitude_range

    def __call__(self, spec, dop, env):
        if np.random.random() > 0.5:
            shift = int(np.random.uniform(-self.time_shift, self.time_shift) * spec.shape[1])
            spec = np.roll(spec, shift, axis=1)
        if np.random.random() > 0.5:
            shift = int(np.random.uniform(-self.freq_shift, self.freq_shift) * spec.shape[0])
            spec = np.roll(spec, shift, axis=0)
        if np.random.random() > 0.5:
            spec = np.clip(spec * np.random.uniform(*self.amplitude_range), 0, 1)
        if np.random.random() > 0.5:
            spec = np.clip(spec + np.random.randn(*spec.shape) * self.noise_level, 0, 1)
        return spec, dop, env


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory (legacy)
# ──────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    data_dir:    str,
    batch_size:  int  = 32,
    num_workers: int  = 0,
    config:      Optional[dict] = None,
    augment_train: bool = True,
) -> Dict[str, DataLoader]:
    transform = RadarDataAugmentation() if augment_train else None
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = RadarDataset(data_dir, split=split,
                          transform=(transform if split == 'train' else None),
                          config=config)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
        )
    return loaders
