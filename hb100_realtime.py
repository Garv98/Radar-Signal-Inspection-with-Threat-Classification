"""
HB100 24 GHz Radar — Real-Time ML Classification
=================================================

Hardware chain:
    HB100 I-pin --> Arduino A0
    HB100 Q-pin --> Arduino A1
    HB100 GND   --> Arduino GND
    HB100 VCC   --> Arduino 5V (or 3.3V)

STEP 1 — Flash this sketch to Arduino (copy into Arduino IDE):
──────────────────────────────────────────────────────────────
/*
  HB100 IQ Capture Sketch — Timestamp-based for accurate resampling
  Sampling rate: ~4000 Hz per channel (tunable via PERIOD_US)
  Baud rate: 115200
  Protocol: On receiving 'S', captures 4096 I/Q pairs with timestamps,
            sends as binary: [4 bytes timestamp_us][1 byte I][1 byte Q]
            followed by 0xDEAD end marker.
*/

const int I_PIN    = A0;
const int Q_PIN    = A1;
const int N_PAIRS  = 4096;
const int PERIOD_US = 250;  // 250 us = 4000 Hz per channel

// Reduce ADC prescaler from 128 to 16 for faster reads (~13 us per read)
void adc_fast() {
  ADCSRA = (ADCSRA & ~0x07) | 0x04;  // prescaler = 16
}

void setup() {
  Serial.begin(115200);
  adc_fast();
}

void loop() {
  if (Serial.available() && Serial.read() == 'S') {
    unsigned long t;
    uint8_t i_val, q_val;

    for (int n = 0; n < N_PAIRS; n++) {
      t     = micros();
      i_val = analogRead(I_PIN) >> 2;   // 10-bit -> 8-bit (0-255)
      q_val = analogRead(Q_PIN) >> 2;

      // Send: [4 bytes timestamp] [1 byte I] [1 byte Q]
      Serial.write((uint8_t*)&t, 4);
      Serial.write(i_val);
      Serial.write(q_val);

      // Busy-wait for next sample time
      while (micros() - t < PERIOD_US) {}
    }

    // End marker
    Serial.write(0xDE);
    Serial.write(0xAD);
  }
}

──────────────────────────────────────────────────────────────
STEP 2 — Run this script:
    python hb100_realtime.py --port COM3
    python hb100_realtime.py --port COM3 --model outputs/models/best_model.pt
    python hb100_realtime.py --port COM3 --continuous --interval 2
    python hb100_realtime.py --port COM3 --save-dir captures/
──────────────────────────────────────────────────────────────

Signal conditioning notes:
    The HB100 I/Q outputs are ~100-400 mV peak. For best SNR, add an
    op-amp gain stage (LM358, gain=10) before the Arduino ADC pins.
    Without amplification, the signal still works but with lower SNR.

Frequency note:
    HB100 nominal: 24.125 GHz (±25 MHz batch variation).
    The classifier works on PATTERNS (rotor harmonics, JEM sidebands,
    wing-beat modulation), not absolute velocities. Frequency drift
    is compensated by the resampling + normalisation pipeline below.
"""

import argparse
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
from scipy.signal import resample, butter, filtfilt, resample_poly
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))

# ── Model targets (must match training config) ────────────────────────────────
TARGET_FS        = 10_000   # Hz  — model was trained at this sample rate
NUM_PULSES       = 32       # IQ matrix rows
SAMPLES_PER_PULSE = 128     # IQ matrix cols
N_PAIRS          = NUM_PULSES * SAMPLES_PER_PULSE   # = 4096 pairs to collect

CLASS_NAMES  = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']
THREAT_LEVEL = {'Drone': 0.90, 'Aircraft': 0.80, 'Bird': 0.20,
                'Clutter': 0.10, 'Noise': 0.00}


# ──────────────────────────────────────────────────────────────────────────────
# Serial capture
# ──────────────────────────────────────────────────────────────────────────────

def capture_from_arduino(ser: serial.Serial, n_pairs: int = N_PAIRS):
    """
    Send 'S' trigger, receive n_pairs × (4-byte timestamp, 1-byte I, 1-byte Q).
    Returns:
        timestamps_us : float array [n_pairs]   microsecond timestamps
        I_raw         : float array [n_pairs]   normalised to [-1, +1]
        Q_raw         : float array [n_pairs]   normalised to [-1, +1]
    """
    ser.reset_input_buffer()
    ser.write(b'S')

    BYTES_PER_PAIR = 6     # 4 timestamp + 1 I + 1 Q
    END_MARKER     = b'\xDE\xAD'
    total_bytes    = n_pairs * BYTES_PER_PAIR + 2   # +2 for end marker

    print(f"  Capturing {n_pairs} pairs (~{n_pairs/4000:.1f}s) ...", end='', flush=True)
    t_start = time.time()

    buf = bytearray()
    while len(buf) < total_bytes:
        chunk = ser.read(min(512, total_bytes - len(buf)))
        buf.extend(chunk)
        elapsed = time.time() - t_start
        print(f"\r  Capturing {len(buf)}/{total_bytes} bytes  ({elapsed:.1f}s)", end='', flush=True)
        if elapsed > 30:
            raise TimeoutError("Arduino capture timed out. Check connections.")

    # Verify end marker
    if buf[-2:] != END_MARKER:
        print(f"\n  [warn] End marker not found — got {buf[-2:].hex()}")

    print(f"\r  Captured {n_pairs} pairs in {time.time()-t_start:.2f}s              ")

    # Unpack binary data
    timestamps = np.empty(n_pairs, dtype=np.float64)
    I_vals     = np.empty(n_pairs, dtype=np.float32)
    Q_vals     = np.empty(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        offset = i * BYTES_PER_PAIR
        t_us, = struct.unpack_from('<I', buf, offset)      # uint32 little-endian
        i_u8  = buf[offset + 4]                            # uint8 0-255
        q_u8  = buf[offset + 5]                            # uint8 0-255
        timestamps[i] = t_us
        I_vals[i]     = i_u8
        Q_vals[i]     = q_u8

    # Handle micros() overflow (wraps at ~71 min)
    timestamps = np.unwrap(timestamps * (2 * np.pi / 2**32)) / (2 * np.pi / 2**32)

    # Normalise ADC values to [-1, +1]
    I_vals = (I_vals - 127.5) / 127.5
    Q_vals = (Q_vals - 127.5) / 127.5

    return timestamps, I_vals, Q_vals


# ──────────────────────────────────────────────────────────────────────────────
# Signal preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(timestamps_us, I_raw, Q_raw, target_fs=TARGET_FS):
    """
    Full preprocessing pipeline:
      1. Interpolate non-uniform timestamps -> uniform time grid
      2. Resample to target_fs
      3. DC removal (remove mean per pulse)
      4. Bandpass filter (20 Hz - 4500 Hz) — removes DC drift + aliasing
      5. Normalise to unit variance
      6. Reshape to [NUM_PULSES x SAMPLES_PER_PULSE] complex IQ matrix

    Returns: complex IQ matrix [32 x 128]
    """
    # 1. Infer actual sampling rate from timestamps
    duration_s   = (timestamps_us[-1] - timestamps_us[0]) * 1e-6
    actual_fs    = len(timestamps_us) / duration_s
    print(f"  Arduino sampling rate: {actual_fs:.1f} Hz/channel")

    # 2. Interpolate to uniform grid at actual_fs, then resample to target_fs
    t_uniform = np.linspace(timestamps_us[0], timestamps_us[-1], len(timestamps_us))

    interp_I = interp1d(timestamps_us, I_raw, kind='linear', fill_value='extrapolate')
    interp_Q = interp1d(timestamps_us, Q_raw, kind='linear', fill_value='extrapolate')

    I_uniform = interp_I(t_uniform).astype(np.float32)
    Q_uniform = interp_Q(t_uniform).astype(np.float32)

    # 3. Resample to target_fs (exact rational resampling)
    n_target = int(round(len(I_uniform) * target_fs / actual_fs))
    n_target = max(N_PAIRS, n_target)   # ensure at least N_PAIRS samples

    # Use resample_poly for best quality (no spectral leakage)
    from math import gcd
    # Rational approximation: up/down
    actual_fs_int = int(round(actual_fs))
    g = gcd(target_fs, actual_fs_int)
    up, down = target_fs // g, actual_fs_int // g

    # Limit up/down to avoid huge filters
    if max(up, down) > 100:
        # Fall back to scipy.signal.resample (FFT-based)
        I_resampled = resample(I_uniform, n_target)
        Q_resampled = resample(Q_uniform, n_target)
    else:
        I_resampled = resample_poly(I_uniform, up, down).astype(np.float32)
        Q_resampled = resample_poly(Q_uniform, up, down).astype(np.float32)

    # Trim/pad to exactly N_PAIRS samples
    I_resampled = I_resampled[:N_PAIRS] if len(I_resampled) >= N_PAIRS else \
                  np.pad(I_resampled, (0, N_PAIRS - len(I_resampled)))
    Q_resampled = Q_resampled[:N_PAIRS] if len(Q_resampled) >= N_PAIRS else \
                  np.pad(Q_resampled, (0, N_PAIRS - len(Q_resampled)))

    # 4. Reshape to [NUM_PULSES x SAMPLES_PER_PULSE]
    I_mat = I_resampled.reshape(NUM_PULSES, SAMPLES_PER_PULSE)
    Q_mat = Q_resampled.reshape(NUM_PULSES, SAMPLES_PER_PULSE)
    iq    = (I_mat + 1j * Q_mat).astype(np.complex64)

    # 5. DC removal — subtract mean per pulse (removes HB100 LO leakage)
    iq -= iq.mean(axis=1, keepdims=True)

    # 6. Bandpass filter per pulse: 20 Hz - 4500 Hz
    #    Removes: DC drift (below 20 Hz) + aliasing components (above Nyquist/2)
    nyq  = target_fs / 2
    low  = 20.0  / nyq
    high = min(4500.0, nyq * 0.9) / nyq
    b, a = butter(4, [low, high], btype='band')

    for p in range(NUM_PULSES):
        real = filtfilt(b, a, iq[p].real)
        imag = filtfilt(b, a, iq[p].imag)
        iq[p] = real + 1j * imag

    # 7. Normalise to unit power
    power = np.mean(np.abs(iq) ** 2)
    if power > 1e-12:
        iq /= np.sqrt(power)

    return iq


# ──────────────────────────────────────────────────────────────────────────────
# Range-Doppler map (same as dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def iq_to_rd_map(iq: np.ndarray) -> np.ndarray:
    """Convert IQ matrix [32 x 128] to normalised Range-Doppler map [32 x 128]."""
    from scipy.fft import fft, fftshift
    range_fft  = fft(iq, n=SAMPLES_PER_PULSE, axis=1)
    doppler_fft = fft(range_fft, n=NUM_PULSES, axis=0)
    rd = fftshift(doppler_fft, axes=(0, 1))
    rd_db = 20.0 * np.log10(np.abs(rd) + 1e-10)
    mn, mx = rd_db.min(), rd_db.max()
    if mx > mn:
        rd_db = (rd_db - mn) / (mx - mn)
    return rd_db.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Model inference
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path: Path):
    import torch
    from src.models.cnn_lstm import build_model
    ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
    model = build_model(ckpt.get('config', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def run_inference(model, rd_map: np.ndarray, n_ensemble: int = 1,
                  rd_maps_extra: list = None):
    """
    Run model inference. If rd_maps_extra is provided (list of additional RD maps),
    averages probabilities across all maps for a more stable prediction.
    """
    import torch
    import torch.nn.functional as F

    all_maps = [rd_map] + (rd_maps_extra or [])
    prob_sum = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    for rd in all_maps:
        spec = torch.from_numpy(rd).unsqueeze(0).unsqueeze(0)   # [1,1,32,128]
        dop  = torch.from_numpy(rd.mean(axis=1)).unsqueeze(0)   # [1,32]
        env  = torch.zeros(1, 3)
        with torch.no_grad():
            logits = model(spec, dop, env)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()
        prob_sum += probs

    avg_probs = prob_sum / len(all_maps)
    idx       = int(avg_probs.argmax())
    return idx, float(avg_probs[idx]), avg_probs


# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────

def display_result(pred_idx, confidence, probs, frame_num=None):
    cls    = CLASS_NAMES[pred_idx]
    threat = THREAT_LEVEL[cls]

    label = f"Frame {frame_num}" if frame_num else "Result"

    if threat >= 0.7:
        alert = "!! THREAT DETECTED !!"
    elif threat >= 0.3:
        alert = "-- LOW THREAT --"
    else:
        alert = "   NO THREAT   "

    print(f"\n{'='*50}")
    print(f"  {label:>8}  |  {alert}")
    print(f"{'='*50}")
    print(f"  Prediction : {cls.upper():10s}  ({confidence:.1%} confidence)")
    print(f"  Threat lvl : {'#' * int(threat*20):<20} {threat:.0%}")
    print(f"{'─'*50}")
    print(f"  Class probabilities:")
    for i, c in enumerate(CLASS_NAMES):
        bar = '#' * int(probs[i] * 25)
        print(f"    {c:10s}  {probs[i]*100:5.1f}%  {bar}")
    print(f"{'='*50}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HB100 real-time radar classification'
    )
    parser.add_argument('--port',       required=True,
                        help='Arduino serial port (e.g. COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud',       type=int, default=115200)
    parser.add_argument('--model',      default='outputs/models/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--continuous', action='store_true',
                        help='Keep classifying in a loop')
    parser.add_argument('--interval',   type=float, default=0,
                        help='Seconds to wait between captures (0 = immediate)')
    parser.add_argument('--ensemble',   type=int, default=3,
                        help='Captures to average per prediction (default 3)')
    parser.add_argument('--save-dir',   default=None,
                        help='Save captured .npy files for UI upload')
    parser.add_argument('--no-model',   action='store_true',
                        help='Capture and save only, skip inference')
    args = parser.parse_args()

    model_path = Path(args.model)
    save_dir   = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = None
    if not args.no_model:
        if not model_path.exists():
            print(f"Model not found: {model_path}. Run train.py first.")
            return
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path)
        print("  Model ready.")

    # ── Open serial port ───────────────────────────────────────────────────────
    print(f"\nConnecting to Arduino on {args.port} @ {args.baud} baud ...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=30)
    except serial.SerialException as e:
        print(f"Cannot open port: {e}")
        print("\nAvailable ports:")
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"  {p.device}  —  {p.description}")
        return

    time.sleep(2)   # Wait for Arduino reset
    ser.reset_input_buffer()
    print("  Connected. Arduino is ready.")

    # ── Capture + classify loop ────────────────────────────────────────────────
    frame_num = 0
    try:
        while True:
            frame_num += 1
            print(f"\n[Frame {frame_num}] Starting capture ...")

            # Capture N ensemble frames and average probabilities
            rd_maps = []
            for k in range(max(1, args.ensemble)):
                if k > 0 and args.interval > 0:
                    time.sleep(args.interval)
                try:
                    timestamps, I_raw, Q_raw = capture_from_arduino(ser)
                except TimeoutError as e:
                    print(f"  Timeout: {e}")
                    break

                # Full preprocessing pipeline
                iq    = preprocess(timestamps, I_raw, Q_raw)
                rd    = iq_to_rd_map(iq)
                rd_maps.append((rd, iq))

            if not rd_maps:
                continue

            # Save the first capture as representative
            rd_repr, iq_repr = rd_maps[0]

            if save_dir:
                ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = save_dir / f"hb100_{ts}_f{frame_num}.npy"
                np.save(fname, iq_repr)
                print(f"  Saved -> {fname}  (upload to UI)")

            # Inference
            if model is not None:
                extra_rds = [rd for rd, _ in rd_maps[1:]]
                idx, conf, probs = run_inference(
                    model, rd_repr,
                    rd_maps_extra=extra_rds,
                )
                display_result(idx, conf, probs, frame_num)

            if not args.continuous:
                break

            if args.interval > 0:
                print(f"\n  Next capture in {args.interval:.1f}s ...")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == '__main__':
    main()
