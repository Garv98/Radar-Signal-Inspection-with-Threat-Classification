"""
Feature Extraction Module for Radar Signals

This module extracts features from preprocessed radar signals including:
- Doppler shift calculation
- Micro-Doppler signatures
- Amplitude features
- Pulse characteristics
- Environmental feature integration
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
from scipy.stats import skew, kurtosis
from typing import Tuple, Dict, List, Optional


def extract_doppler_shift(
    iq_matrix: np.ndarray,
    carrier_freq: float = 24e9,
    prf: float = 1000,
    num_bins: int = 64
) -> Tuple[np.ndarray, float]:
    """
    Calculate Doppler shift and velocity from IQ matrix.

    The Doppler shift is related to velocity by:
    v = (f_d * c) / (2 * f_c)

    where f_d is Doppler frequency, c is speed of light, f_c is carrier frequency.

    Args:
        iq_matrix: IQ data matrix [num_pulses x num_samples]
        carrier_freq: Radar carrier frequency (Hz)
        prf: Pulse Repetition Frequency (Hz)
        num_bins: Number of Doppler bins for FFT

    Returns:
        Tuple of (doppler_spectrum, estimated_velocity)
    """
    # Ensure numeric types
    carrier_freq = float(carrier_freq)
    prf = float(prf)
    num_bins = int(num_bins)

    c = 3e8  # Speed of light

    # Extract phase from IQ data for each range bin
    # Average across range bins to get slow-time signal
    slow_time_signal = np.mean(iq_matrix, axis=1)

    # Compute Doppler FFT
    doppler_fft = fft(slow_time_signal, n=num_bins)
    doppler_fft = fftshift(doppler_fft)
    doppler_spectrum = np.abs(doppler_fft)

    # Doppler frequency bins
    doppler_freqs = np.linspace(-prf/2, prf/2, num_bins)

    # Find peak Doppler frequency
    peak_idx = np.argmax(doppler_spectrum)
    peak_doppler_freq = doppler_freqs[peak_idx]

    # Calculate velocity from Doppler shift
    # v = (f_d * lambda) / 2 = (f_d * c) / (2 * f_c)
    velocity = (peak_doppler_freq * c) / (2 * carrier_freq)

    return doppler_spectrum, velocity


def extract_micro_doppler(
    spectrogram: np.ndarray,
    threshold_db: float = -30
) -> Dict[str, np.ndarray]:
    """
    Extract micro-Doppler signature features from spectrogram.

    Micro-Doppler signatures arise from rotating or vibrating parts
    (e.g., drone rotors, bird wings) and appear as time-varying
    frequency modulations around the main Doppler shift.

    Args:
        spectrogram: Time-frequency spectrogram (in dB)
        threshold_db: Threshold for feature extraction

    Returns:
        Dictionary containing:
        - 'bandwidth': Time-varying bandwidth of micro-Doppler
        - 'centroid': Spectral centroid over time
        - 'periodicity': Detected periodic components
        - 'envelope': Micro-Doppler envelope
    """
    # Normalize spectrogram
    spec_norm = spectrogram - np.max(spectrogram)

    # Create mask for significant returns
    mask = spec_norm > threshold_db

    # Calculate spectral features over time
    num_time_bins = spectrogram.shape[1]
    num_freq_bins = spectrogram.shape[0]
    freq_axis = np.arange(num_freq_bins)

    bandwidth = np.zeros(num_time_bins)
    centroid = np.zeros(num_time_bins)

    for t in range(num_time_bins):
        col = spectrogram[:, t]
        col_linear = 10 ** (col / 10)  # Convert from dB to linear
        col_linear = col_linear / (np.sum(col_linear) + 1e-10)  # Normalize

        # Spectral centroid
        centroid[t] = np.sum(freq_axis * col_linear)

        # Bandwidth (standard deviation)
        bandwidth[t] = np.sqrt(np.sum(((freq_axis - centroid[t]) ** 2) * col_linear))

    # Extract periodicity using autocorrelation
    centroid_centered = centroid - np.mean(centroid)
    autocorr = np.correlate(centroid_centered, centroid_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

    # Find peaks in autocorrelation (periodic components)
    peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=5)
    periodicity = peaks[0] if len(peaks) > 0 else 0

    # Extract envelope (max frequency extent over time)
    envelope_upper = np.zeros(num_time_bins)
    envelope_lower = np.zeros(num_time_bins)

    for t in range(num_time_bins):
        active_freqs = np.where(mask[:, t])[0]
        if len(active_freqs) > 0:
            envelope_upper[t] = np.max(active_freqs)
            envelope_lower[t] = np.min(active_freqs)

    return {
        'bandwidth': bandwidth,
        'centroid': centroid,
        'periodicity': np.array([periodicity]),
        'envelope_upper': envelope_upper,
        'envelope_lower': envelope_lower,
        'envelope_width': envelope_upper - envelope_lower
    }


def extract_amplitude_features(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Extract amplitude-based features from signal.

    Args:
        signal_data: Input signal (magnitude or IQ)

    Returns:
        Dictionary of amplitude features
    """
    # Convert to magnitude if complex
    if np.iscomplexobj(signal_data):
        magnitude = np.abs(signal_data)
    else:
        magnitude = np.abs(signal_data)

    # Flatten if multi-dimensional
    magnitude = magnitude.flatten()

    features = {
        'peak': np.max(magnitude),
        'mean': np.mean(magnitude),
        'std': np.std(magnitude),
        'variance': np.var(magnitude),
        'rms': np.sqrt(np.mean(magnitude ** 2)),
        'peak_to_mean': np.max(magnitude) / (np.mean(magnitude) + 1e-10),
        'dynamic_range': np.max(magnitude) - np.min(magnitude),
        'skewness': skew(magnitude),
        'kurtosis': kurtosis(magnitude),
        'crest_factor': np.max(magnitude) / (np.sqrt(np.mean(magnitude ** 2)) + 1e-10)
    }

    return features


def extract_pulse_characteristics(
    iq_matrix: np.ndarray,
    fs: float = 10000,
    prf: float = 1000
) -> Dict[str, float]:
    """
    Extract pulse-related characteristics.

    Args:
        iq_matrix: IQ data matrix [num_pulses x num_samples]
        fs: Sampling frequency (Hz)
        prf: Pulse Repetition Frequency (Hz)

    Returns:
        Dictionary of pulse characteristics
    """
    num_pulses, num_samples = iq_matrix.shape

    # Calculate pulse-to-pulse correlation
    correlations = []
    for i in range(num_pulses - 1):
        corr = np.abs(np.corrcoef(
            np.abs(iq_matrix[i]),
            np.abs(iq_matrix[i + 1])
        )[0, 1])
        correlations.append(corr)

    # Calculate energy per pulse
    pulse_energies = np.sum(np.abs(iq_matrix) ** 2, axis=1)

    # Estimate pulse width from autocorrelation
    avg_pulse = np.mean(np.abs(iq_matrix), axis=0)
    autocorr = np.correlate(avg_pulse, avg_pulse, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find -3dB point for pulse width estimation
    autocorr_norm = autocorr / autocorr[0]
    try:
        pulse_width_samples = np.where(autocorr_norm < 0.5)[0][0]
        pulse_width = pulse_width_samples / fs
    except IndexError:
        pulse_width = num_samples / fs

    characteristics = {
        'num_pulses': num_pulses,
        'samples_per_pulse': num_samples,
        'prf': prf,
        'pulse_width': pulse_width,
        'pri': 1 / prf,  # Pulse Repetition Interval
        'mean_pulse_correlation': np.mean(correlations) if correlations else 0,
        'pulse_correlation_std': np.std(correlations) if correlations else 0,
        'mean_pulse_energy': np.mean(pulse_energies),
        'pulse_energy_std': np.std(pulse_energies),
        'duty_cycle': pulse_width * prf
    }

    return characteristics


def extract_frequency_signature(
    fft_data: np.ndarray,
    fs: float = 10000,
    num_peaks: int = 5
) -> Dict[str, np.ndarray]:
    """
    Extract frequency-domain signature features.

    Args:
        fft_data: FFT magnitude data
        fs: Sampling frequency (Hz)
        num_peaks: Number of dominant peaks to extract

    Returns:
        Dictionary of frequency signature features
    """
    # Ensure 1D
    if fft_data.ndim > 1:
        fft_data = np.mean(fft_data, axis=0)

    n = len(fft_data)
    freqs = np.linspace(-fs/2, fs/2, n)

    # Find dominant peaks
    peaks, properties = signal.find_peaks(fft_data, height=np.max(fft_data) - 20)

    # Sort by peak height
    if len(peaks) > 0:
        peak_heights = properties['peak_heights']
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_peaks = peaks[sorted_indices[:num_peaks]]
        top_peak_freqs = freqs[top_peaks]
        top_peak_mags = fft_data[top_peaks]
    else:
        top_peak_freqs = np.zeros(num_peaks)
        top_peak_mags = np.zeros(num_peaks)

    # Pad if fewer peaks found
    if len(top_peak_freqs) < num_peaks:
        top_peak_freqs = np.pad(top_peak_freqs, (0, num_peaks - len(top_peak_freqs)))
        top_peak_mags = np.pad(top_peak_mags, (0, num_peaks - len(top_peak_mags)))

    # Spectral features
    fft_linear = 10 ** (fft_data / 20)  # Convert from dB
    fft_norm = fft_linear / (np.sum(fft_linear) + 1e-10)

    spectral_centroid = np.sum(freqs * fft_norm)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_norm))
    spectral_flatness = np.exp(np.mean(np.log(fft_linear + 1e-10))) / (np.mean(fft_linear) + 1e-10)

    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(fft_linear)
    rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]

    return {
        'dominant_frequencies': top_peak_freqs,
        'dominant_magnitudes': top_peak_mags,
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'spectral_flatness': spectral_flatness,
        'spectral_rolloff': spectral_rolloff,
        'num_significant_peaks': len(peaks)
    }


def add_environmental_features(
    features: np.ndarray,
    rain: float,
    temperature: float,
    pressure: float
) -> np.ndarray:
    """
    Append environmental factors that affect Doppler shift.

    Environmental factors influence radar signal propagation:
    - Rain: causes signal attenuation and clutter returns
    - Temperature: affects atmospheric refraction index
    - Pressure: affects atmospheric density and refraction

    Args:
        features: Existing feature array
        rain: Rainfall rate in mm/hr (0 = no rain)
        temperature: Temperature in Celsius
        pressure: Atmospheric pressure in hPa

    Returns:
        Feature array with environmental factors appended
    """
    # Normalize environmental features to reasonable ranges
    # Rain: 0-100 mm/hr -> 0-1
    rain_norm = np.clip(rain / 100.0, 0, 1)

    # Temperature: -40 to 50 C -> 0-1
    temp_norm = np.clip((temperature + 40) / 90.0, 0, 1)

    # Pressure: 950-1050 hPa -> 0-1
    pressure_norm = np.clip((pressure - 950) / 100.0, 0, 1)

    env_features = np.array([rain_norm, temp_norm, pressure_norm])

    # Handle different input shapes
    if features.ndim == 1:
        return np.concatenate([features, env_features])
    else:
        # Broadcast for batch processing
        batch_size = features.shape[0]
        env_batch = np.tile(env_features, (batch_size, 1))
        return np.concatenate([features, env_batch], axis=1)


def extract_all_features(
    iq_matrix: np.ndarray,
    spectrogram: np.ndarray,
    config: dict,
    environmental: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract all features from radar data.

    Args:
        iq_matrix: Raw IQ data matrix
        spectrogram: Preprocessed spectrogram
        config: Configuration dictionary
        environmental: Dict with 'rain', 'temperature', 'pressure' keys

    Returns:
        Dictionary containing all extracted features
    """
    carrier_freq = config.get('carrier_frequency', 24e9)
    prf = config.get('prf', 1000)
    fs = config.get('sampling_rate', 10000)

    # Extract Doppler features
    doppler_spectrum, velocity = extract_doppler_shift(
        iq_matrix, carrier_freq=carrier_freq, prf=prf
    )

    # Extract micro-Doppler features
    micro_doppler = extract_micro_doppler(spectrogram)

    # Extract amplitude features
    amplitude = extract_amplitude_features(iq_matrix)

    # Extract pulse characteristics
    pulse_chars = extract_pulse_characteristics(iq_matrix, fs=fs, prf=prf)

    # Extract frequency signature
    fft_avg = np.mean(np.abs(fft(iq_matrix, axis=1)), axis=0)
    freq_sig = extract_frequency_signature(fft_avg, fs=fs)

    # Combine scalar features into vector
    scalar_features = np.array([
        velocity,
        amplitude['peak'],
        amplitude['mean'],
        amplitude['std'],
        amplitude['rms'],
        amplitude['skewness'],
        amplitude['kurtosis'],
        pulse_chars['mean_pulse_correlation'],
        pulse_chars['pulse_width'],
        freq_sig['spectral_centroid'],
        freq_sig['spectral_spread'],
        freq_sig['spectral_flatness'],
        np.mean(micro_doppler['bandwidth']),
        np.std(micro_doppler['centroid']),
        micro_doppler['periodicity'][0]
    ])

    # Add environmental features if provided
    if environmental is not None:
        scalar_features = add_environmental_features(
            scalar_features,
            rain=environmental.get('rain', 0),
            temperature=environmental.get('temperature', 20),
            pressure=environmental.get('pressure', 1013)
        )

    return {
        'scalar_features': scalar_features,
        'doppler_spectrum': doppler_spectrum,
        'micro_doppler': micro_doppler,
        'velocity': velocity,
        'spectrogram': spectrogram
    }
