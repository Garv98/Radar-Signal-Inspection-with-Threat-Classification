"""Preprocessing: the shared feature path must be deterministic and bounded."""

import numpy as np

from src.data.dataset import iq_to_range_doppler, iq_to_doppler_profile


def _fake_iq(pulses=32, samples=128, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((pulses, samples))
            + 1j * rng.standard_normal((pulses, samples)))


def test_rd_map_shape_and_range():
    iq = _fake_iq()
    rd = iq_to_range_doppler(iq, range_fft_size=128, doppler_fft_size=32)
    assert rd.shape == (32, 128)
    assert rd.dtype == np.float32
    assert rd.min() >= 0.0 and rd.max() <= 1.0


def test_rd_map_deterministic():
    iq = _fake_iq(seed=42)
    a = iq_to_range_doppler(iq)
    b = iq_to_range_doppler(iq)
    assert np.array_equal(a, b)


def test_doppler_profile():
    rd = iq_to_range_doppler(_fake_iq())
    dop = iq_to_doppler_profile(rd)
    assert dop.shape == (32,)
    assert np.allclose(dop, rd.mean(axis=1))
