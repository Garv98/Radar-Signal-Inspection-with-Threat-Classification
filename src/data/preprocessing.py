"""
Data Preprocessing Module for Radar Signal Processing

This module handles all signal preprocessing operations including:
- FFT transformation
- Spectrogram generation
- Signal denoising
- Normalization
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import pywt
from typing import Tuple, Optional, Literal


def apply_fft(
    signal_data: np.ndarray,
    n_fft: int = 128,
    axis: int = -1
) -> np.ndarray:
    """
    Apply FFT to convert time-domain signal to frequency-domain.

    Args:
        signal_data: Input time-domain signal (can be complex IQ data)
        n_fft: Number of FFT points
        axis: Axis along which to compute FFT

    Returns:
        Frequency-domain representation (magnitude spectrum)
    """
    # Apply FFT
    fft_result = fft(signal_data, n=n_fft, axis=axis)

    # Shift zero frequency to center
    fft_shifted = fftshift(fft_result, axes=axis)

    # Return magnitude spectrum (in dB)
    magnitude = np.abs(fft_shifted)
    # Avoid log(0)
    magnitude = np.where(magnitude > 0, magnitude, 1e-10)
    magnitude_db = 20 * np.log10(magnitude)

    return magnitude_db


def compute_spectrogram(
    iq_data: np.ndarray,
    fs: float = 10000,
    nperseg: int = 64,
    noverlap: Optional[int] = None,
    nfft: int = 128,
    window: str = "hann"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate time-frequency spectrogram from IQ data.

    Args:
        iq_data: Complex IQ data array
        fs: Sampling frequency (Hz)
        nperseg: Length of each segment
        noverlap: Number of overlapping points (default: nperseg//2)
        nfft: FFT length
        window: Window function type

    Returns:
        Tuple of (frequencies, times, spectrogram)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    # Handle complex IQ data
    if np.iscomplexobj(iq_data):
        # Use complex signal directly for STFT
        frequencies, times, Sxx = signal.spectrogram(
            iq_data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=False,  # Full spectrum for complex signals
            mode='complex'
        )
        # Convert to power spectrum
        Sxx = np.abs(Sxx) ** 2
    else:
        frequencies, times, Sxx = signal.spectrogram(
            iq_data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft
        )

    # Shift frequencies to center zero and convert to dB
    Sxx = fftshift(Sxx, axes=0)
    frequencies = fftshift(frequencies)

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    return frequencies, times, Sxx_db


def denoise_signal(
    signal_data: np.ndarray,
    method: Literal["wavelet", "median", "gaussian"] = "wavelet",
    **kwargs
) -> np.ndarray:
    """
    Remove noise from radar signal.

    Args:
        signal_data: Input signal (can be complex)
        method: Denoising method ('wavelet', 'median', 'gaussian')
        **kwargs: Additional parameters for specific methods
            - wavelet: wavelet='db4', level=4, threshold_mode='soft'
            - median: kernel_size=5
            - gaussian: sigma=1.0

    Returns:
        Denoised signal
    """
    is_complex = np.iscomplexobj(signal_data)

    if is_complex:
        # Process real and imaginary parts separately
        real_denoised = denoise_signal(signal_data.real, method, **kwargs)
        imag_denoised = denoise_signal(signal_data.imag, method, **kwargs)
        return real_denoised + 1j * imag_denoised

    if method == "wavelet":
        wavelet = kwargs.get("wavelet", "db4")
        level = kwargs.get("level", 4)
        threshold_mode = kwargs.get("threshold_mode", "soft")

        # Wavelet decomposition
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        # Calculate threshold using MAD (Median Absolute Deviation)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))

        # Apply thresholding to detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients
        for coeff in coeffs[1:]:
            if threshold_mode == "soft":
                denoised_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))
            else:
                denoised_coeffs.append(pywt.threshold(coeff, threshold, mode='hard'))

        # Reconstruct signal
        return pywt.waverec(denoised_coeffs, wavelet)[:len(signal_data)]

    elif method == "median":
        kernel_size = kwargs.get("kernel_size", 5)
        return signal.medfilt(signal_data, kernel_size=kernel_size)

    elif method == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(signal_data, sigma=sigma)

    else:
        raise ValueError(f"Unknown denoising method: {method}")


def normalize(
    data: np.ndarray,
    method: Literal["minmax", "zscore"] = "minmax",
    feature_range: Tuple[float, float] = (0, 1),
    axis: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Normalize data using specified method.

    Args:
        data: Input data array
        method: Normalization method ('minmax' or 'zscore')
        feature_range: Output range for minmax scaling
        axis: Axis along which to normalize (None for global)

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    params = {"method": method}

    if method == "minmax":
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)

        # Avoid division by zero
        data_range = data_max - data_min
        data_range = np.where(data_range == 0, 1, data_range)

        normalized = (data - data_min) / data_range

        # Scale to feature_range
        min_val, max_val = feature_range
        normalized = normalized * (max_val - min_val) + min_val

        params.update({
            "min": data_min,
            "max": data_max,
            "feature_range": feature_range
        })

    elif method == "zscore":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

        # Avoid division by zero
        std = np.where(std == 0, 1, std)

        normalized = (data - mean) / std

        params.update({
            "mean": mean,
            "std": std
        })

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def bandpass_filter(
    signal_data: np.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float,
    order: int = 5
) -> np.ndarray:
    """
    Apply bandpass filter to signal.

    Args:
        signal_data: Input signal
        low_freq: Lower cutoff frequency (Hz)
        high_freq: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Ensure valid frequency range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))

    b, a = signal.butter(order, [low, high], btype='band')

    # Handle complex signals
    if np.iscomplexobj(signal_data):
        real_filtered = signal.filtfilt(b, a, signal_data.real)
        imag_filtered = signal.filtfilt(b, a, signal_data.imag)
        return real_filtered + 1j * imag_filtered

    return signal.filtfilt(b, a, signal_data)


def generate_range_doppler_map(
    iq_matrix: np.ndarray,
    range_fft_size: int = 128,
    doppler_fft_size: int = 32
) -> np.ndarray:
    """
    Generate Range-Doppler map from IQ matrix.

    Args:
        iq_matrix: 2D IQ matrix [num_pulses x num_samples]
        range_fft_size: FFT size for range processing
        doppler_fft_size: FFT size for Doppler processing

    Returns:
        Range-Doppler map in dB
    """
    num_pulses, num_samples = iq_matrix.shape

    # Range FFT (along fast-time / samples axis)
    range_fft = fft(iq_matrix, n=range_fft_size, axis=1)

    # Doppler FFT (along slow-time / pulses axis)
    doppler_fft = fft(range_fft, n=doppler_fft_size, axis=0)

    # Shift and compute magnitude
    rd_map = fftshift(doppler_fft, axes=(0, 1))
    rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-10)

    return rd_map_db


def preprocess_iq_matrix(
    iq_matrix: np.ndarray,
    config: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for IQ matrix.

    Args:
        iq_matrix: Raw IQ matrix [num_pulses x num_samples]
        config: Configuration dictionary with preprocessing parameters

    Returns:
        Tuple of (spectrogram, range_doppler_map)
    """
    # Get config parameters
    fs = config.get("sampling_rate", 10000)
    fft_size = config.get("fft_size", 128)
    denoise_method = config.get("denoise_method", "wavelet")
    normalize_method = config.get("normalize_method", "minmax")

    # Flatten for denoising if needed
    original_shape = iq_matrix.shape

    # Denoise each pulse
    denoised_matrix = np.zeros_like(iq_matrix)
    for i in range(iq_matrix.shape[0]):
        denoised_matrix[i] = denoise_signal(iq_matrix[i], method=denoise_method)

    # Generate spectrogram (using first pulse or averaged)
    avg_pulse = np.mean(denoised_matrix, axis=0)
    _, _, spectrogram = compute_spectrogram(avg_pulse, fs=fs, nfft=fft_size)

    # Generate Range-Doppler map
    rd_map = generate_range_doppler_map(denoised_matrix, range_fft_size=fft_size)

    # Normalize
    spectrogram, _ = normalize(spectrogram, method=normalize_method)
    rd_map, _ = normalize(rd_map, method=normalize_method)

    return spectrogram, rd_map
