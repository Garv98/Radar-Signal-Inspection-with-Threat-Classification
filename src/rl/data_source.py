"""
data_source.py
==============
Unified radar-signal source for the RL gatekeeper.

The gatekeeper must be trained on the *exact* same feature representation that
the ML classifier consumes.  This module discovers the available data in the
repository and produces a pool of pre-computed feature tensors

    X = (rd_map [1 x D x R], doppler_seq [D], env [3], label)

drawing from two discovered modalities:

  * ``zenodo``    -- the real 77 GHz SAAB SIRS FMCW dataset
                     (data/real/zenodo_77ghz/), segmented into CPIs.
  * ``synthetic`` -- the physics-based :class:`SyntheticRadarGenerator`,
                     used to cover classes that are absent from the real set
                     (Aircraft, Noise) and to balance the pool.

Reuses the existing preprocessing (``iq_to_range_doppler`` /
``iq_to_doppler_profile``) and the existing Zenodo class mapping so there is a
single feature-engineering path shared by the ML model and the RL agent.

Nothing here is hardcoded beyond the dataset's own documented geometry
(CPI = 32 pulses x 128 range bins, matching the ML model's input); those values
are read from the config / model metadata by the callers.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.dataset import iq_to_range_doppler, iq_to_doppler_profile, _normalize_env
from src.data.synthetic_generator import SyntheticRadarGenerator

# Canonical 5-class taxonomy shared with the ML classifier.
CLASS_NAMES = ["Drone", "Aircraft", "Bird", "Clutter", "Noise"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Real Zenodo recording label-string -> canonical class index.
# (Identical mapping to finetune_zenodo.py so the gatekeeper sees the same
#  labels the ML model was fine-tuned on.)
ZENODO_CLASS_MAP = {
    "d1": 0, "d2": 0, "d3": 0, "d4": 0, "d5": 0, "d6": 0,
    "seagull": 2, "black-headed gull": 2, "heron": 2,
    "pigeon": 2, "raven": 2, "gull": 2,
    "human_walk": 3, "human_run": 3, "human": 3,
    "cr": 3,
}

DEFAULT_ZENODO_FILE = Path("data/real/zenodo_77ghz/data_SAAB_SIRS_77GHz_FMCW.npy")


class Signal:
    """A single pre-computed feature tensor and its ground-truth label."""

    __slots__ = ("rd_map", "doppler", "env", "label", "source", "iq", "probs")

    def __init__(self, rd_map: np.ndarray, doppler: np.ndarray,
                 env: np.ndarray, label: int, source: str,
                 iq: Optional[np.ndarray] = None):
        self.rd_map = rd_map      # float32 [D, R]   (normalised RD map)
        self.doppler = doppler    # float32 [D]
        self.env = env            # float32 [3]
        self.label = int(label)
        self.source = source      # 'zenodo' | 'synthetic'
        self.iq = iq              # complex [P, R] raw CPI (optional, for signal views)
        self.probs = None         # cached teacher posterior (fixed model -> compute once)


def _label_from_zenodo_string(s: str) -> int:
    s = s.strip().lower()
    if s in ZENODO_CLASS_MAP:
        return ZENODO_CLASS_MAP[s]
    for key, idx in ZENODO_CLASS_MAP.items():
        if key in s:
            return idx
    return -1


def _signal_from_iq(iq: np.ndarray, label: int, source: str,
                    metadata: Optional[Dict], range_fft: int,
                    doppler_fft: int, keep_iq: bool = False) -> Signal:
    """Run the shared preprocessing on a raw IQ CPI -> feature tensor."""
    rd = iq_to_range_doppler(iq, range_fft_size=range_fft, doppler_fft_size=doppler_fft)
    dop = iq_to_doppler_profile(rd)
    env = _normalize_env(metadata or {})
    return Signal(rd, dop, env, label, source,
                  iq=(iq.astype(np.complex64) if keep_iq else None))


def load_zenodo_signals(
    data_file: Path = DEFAULT_ZENODO_FILE,
    max_segments: Optional[int] = None,
    range_fft: int = 128,
    doppler_fft: int = 32,
    verbose: bool = True,
    keep_iq: bool = False,
) -> List[Signal]:
    """
    Load the real 77 GHz Zenodo dataset and segment each recording into CPIs,
    producing the shared feature tensor for every segment.
    """
    data_file = Path(data_file)
    if not data_file.exists():
        if verbose:
            print(f"[data_source] Zenodo file not found: {data_file}")
        return []

    pulses = doppler_fft
    range_bins = range_fft

    if verbose:
        print(f"[data_source] Loading real 77 GHz data: {data_file.name}")
    data = np.load(data_file, allow_pickle=True)

    signals: List[Signal] = []
    for row in range(data.shape[0]):
        if max_segments and len(signals) >= max_segments:
            break

        cls_str = str(data[row, 0].flat[0])
        label = _label_from_zenodo_string(cls_str)
        if label < 0:
            continue

        iq = np.asarray(data[row, 1]).T            # [N_pulses x 1280]
        n_pulses, n_range = iq.shape

        # Centre-crop / pad range to the model's range dimension.
        if n_range >= range_bins:
            r0 = (n_range - range_bins) // 2
            iq = iq[:, r0:r0 + range_bins]
        else:
            pad = range_bins - n_range
            iq = np.pad(iq, ((0, 0), (pad // 2, pad - pad // 2)))

        # Segment slow-time into coherent processing intervals of `pulses`.
        if n_pulses < pulses:
            iq = np.pad(iq, ((0, pulses - n_pulses), (0, 0)))
            n_segs = 1
        else:
            n_segs = n_pulses // pulses

        for seg in range(n_segs):
            if max_segments and len(signals) >= max_segments:
                break
            cpi = iq[seg * pulses:(seg + 1) * pulses, :]
            signals.append(
                _signal_from_iq(cpi, label, "zenodo", None, range_fft, doppler_fft,
                                keep_iq=keep_iq)
            )

    if verbose:
        counts = Counter(s.label for s in signals)
        summary = "  ".join(
            f"{CLASS_NAMES[i]}={counts.get(i, 0)}" for i in range(len(CLASS_NAMES))
        )
        print(f"[data_source]   real segments: {len(signals)}  ({summary})")
    return signals


def generate_synthetic_signals(
    per_class: int,
    seed: int = 42,
    range_fft: int = 128,
    doppler_fft: int = 32,
    classes: Optional[List[str]] = None,
    verbose: bool = True,
    keep_iq: bool = False,
) -> List[Signal]:
    """Generate balanced synthetic signals across the requested classes."""
    gen = SyntheticRadarGenerator(seed=seed, num_pulses=doppler_fft)
    classes = classes or CLASS_NAMES
    signals: List[Signal] = []
    for cls_name in classes:
        for _ in range(per_class):
            iq, meta = gen.generate_sample(cls_name)
            signals.append(
                _signal_from_iq(iq, CLASS_TO_IDX[cls_name], "synthetic",
                                meta, range_fft, doppler_fft, keep_iq=keep_iq)
            )
    if verbose:
        print(f"[data_source]   synthetic samples: {len(signals)} "
              f"({per_class}/class x {len(classes)} classes)")
    return signals


class RadarSignalSource:
    """
    A samplable pool of feature tensors for the gatekeeper environment.

    Discovers and combines real + synthetic signals according to ``mode`` and
    exposes :meth:`sample` to draw a random :class:`Signal`.  The label
    distribution of the pool is exposed via :meth:`label_counts` so the reward
    function can derive class values from the actual data.
    """

    def __init__(
        self,
        mode: str = "mixed",
        synthetic_per_class: int = 600,
        zenodo_file: Path = DEFAULT_ZENODO_FILE,
        zenodo_max_segments: Optional[int] = None,
        range_fft: int = 128,
        doppler_fft: int = 32,
        seed: int = 42,
        verbose: bool = True,
        keep_iq: bool = False,
    ):
        if mode not in ("zenodo", "synthetic", "mixed"):
            raise ValueError(f"Unknown data_source mode: {mode}")

        self.mode = mode
        self.range_fft = range_fft
        self.doppler_fft = doppler_fft
        self.rng = np.random.default_rng(seed)

        pool: List[Signal] = []
        if mode in ("zenodo", "mixed"):
            pool += load_zenodo_signals(
                zenodo_file, zenodo_max_segments, range_fft, doppler_fft, verbose,
                keep_iq=keep_iq,
            )
        if mode in ("synthetic", "mixed"):
            # In mixed mode, synthesise the classes the real set lacks plus a
            # balancing amount of the rest, so every class is represented.
            pool += generate_synthetic_signals(
                synthetic_per_class, seed, range_fft, doppler_fft,
                classes=None, verbose=verbose, keep_iq=keep_iq,
            )

        if not pool:
            raise RuntimeError(
                "RadarSignalSource produced an empty pool. "
                "Check the Zenodo file path or enable synthetic generation."
            )

        self.pool = pool
        # Pre-index by label so we can draw class-balanced batches if desired.
        self._by_label: Dict[int, List[int]] = {}
        for i, s in enumerate(pool):
            self._by_label.setdefault(s.label, []).append(i)

        if verbose:
            print(f"[data_source] pool ready: {len(pool)} signals "
                  f"across {len(self._by_label)} classes (mode={mode})")

    def __len__(self) -> int:
        return len(self.pool)

    @property
    def present_labels(self) -> List[int]:
        return sorted(self._by_label.keys())

    def label_counts(self) -> np.ndarray:
        """Counts per canonical class index (length = len(CLASS_NAMES))."""
        counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for lbl, idxs in self._by_label.items():
            counts[lbl] = len(idxs)
        return counts

    def sample(self, balanced: bool = True) -> Signal:
        """Draw one random signal (class-balanced by default)."""
        if balanced:
            lbl = int(self.rng.choice(self.present_labels))
            idx = int(self.rng.choice(self._by_label[lbl]))
        else:
            idx = int(self.rng.integers(0, len(self.pool)))
        return self.pool[idx]

    def sample_label(self, label: int) -> Signal:
        """Draw a random signal of a specific class (falls back to any if absent)."""
        idxs = self._by_label.get(int(label))
        if not idxs:
            return self.sample()
        return self.pool[int(self.rng.choice(idxs))]

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
