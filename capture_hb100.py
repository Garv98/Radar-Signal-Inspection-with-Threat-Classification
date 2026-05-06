"""
HB100 24 GHz Radar — Real Data Capture Script

Hardware setup (two options):
═══════════════════════════════════════════════════════════

OPTION A — USB Sound Card (easiest, ~$5)
─────────────────────────────────────────
  HB100 I-pin  ──[1kΩ]──► Left  channel (TIP)   of 3.5mm jack
  HB100 Q-pin  ──[1kΩ]──► Right channel (RING)  of 3.5mm jack
  HB100 GND    ──────────► Sleeve (GND) of 3.5mm jack
  USB sound card ─────────► PC USB port

  Note: HB100 I/Q outputs are ~100mV. The 1kΩ resistor protects
        the sound card. Add an op-amp amplifier (e.g. LM358) for
        better SNR if signal is too weak.

OPTION B — Arduino (more control)
───────────────────────────────────
  HB100 I-pin  ──► Arduino A0
  HB100 Q-pin  ──► Arduino A1
  HB100 GND    ──► Arduino GND
  HB100 VCC    ──► Arduino 5V

  Flash this sketch to Arduino first:
  ┌──────────────────────────────────────┐
  │ void setup() { Serial.begin(115200); }│
  │ void loop() {                         │
  │   int i = analogRead(A0);             │
  │   int q = analogRead(A1);             │
  │   Serial.print(i); Serial.print(','); │
  │   Serial.println(q);                  │
  │ }                                     │
  └──────────────────────────────────────┘

Usage
─────
  # List available audio devices (Option A)
  python capture_hb100.py --list-devices

  # Capture via sound card (Option A)
  python capture_hb100.py --mode audio --device 1 --output capture.npy

  # Capture via Arduino (Option B)
  python capture_hb100.py --mode serial --port COM3 --output capture.npy

  # Capture multiple frames (for batch testing)
  python capture_hb100.py --mode audio --frames 10 --output-dir captures/

  # Preview signal in real-time before saving
  python capture_hb100.py --mode audio --preview
"""

import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Model parameters — must match training config
SAMPLE_RATE      = 10000   # Hz  (resample if your ADC uses different rate)
NUM_PULSES       = 32      # rows of IQ matrix
SAMPLES_PER_PULSE = 128    # cols of IQ matrix
CPI_SAMPLES      = NUM_PULSES * SAMPLES_PER_PULSE   # 4096 total samples per frame
CPI_DURATION     = CPI_SAMPLES / SAMPLE_RATE        # ~0.41 seconds


# ──────────────────────────────────────────────────────────────────────────────
# Device listing
# ──────────────────────────────────────────────────────────────────────────────

def list_audio_devices():
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed: pip install sounddevice")
        return

    print("\nAvailable audio input devices:")
    print(f"  {'ID':>4}  {'Name':<40}  {'Channels':>8}  {'Sample Rate':>12}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*8}  {'-'*12}")
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] >= 2:
            print(f"  {i:>4}  {dev['name']:<40}  "
                  f"{dev['max_input_channels']:>8}  "
                  f"{int(dev['default_samplerate']):>12}")
    print("\nUse the ID number with --device")


# ──────────────────────────────────────────────────────────────────────────────
# Audio capture (Option A — Sound card)
# ──────────────────────────────────────────────────────────────────────────────

def capture_audio(device_id: int, n_frames: int = 1,
                  output_dir: Path = None, preview: bool = False):
    try:
        import sounddevice as sd
        from scipy.signal import resample
    except ImportError:
        print("Install required packages: pip install sounddevice scipy")
        return []

    print(f"\nCapturing from device ID {device_id}")
    print(f"  Sample rate  : {SAMPLE_RATE} Hz")
    print(f"  CPI duration : {CPI_DURATION:.2f} s per frame")
    print(f"  Frames       : {n_frames}")

    # Get device native sample rate
    dev_info    = sd.query_devices(device_id, 'input')
    native_rate = int(dev_info['default_samplerate'])
    n_native    = int(CPI_DURATION * native_rate)

    saved_files = []

    for frame_idx in range(n_frames):
        print(f"\n  [Frame {frame_idx+1}/{n_frames}] Recording {CPI_DURATION:.2f}s ... ",
              end='', flush=True)

        raw = sd.rec(n_native, samplerate=native_rate,
                     channels=2, dtype='float32', device=device_id)
        sd.wait()
        print("done")

        # Channel split: Left = I, Right = Q
        I_raw = raw[:, 0]
        Q_raw = raw[:, 1]

        # Resample to target SAMPLE_RATE if needed
        if native_rate != SAMPLE_RATE:
            target_len = CPI_SAMPLES
            I_raw = resample(I_raw, target_len).astype(np.float32)
            Q_raw = resample(Q_raw, target_len).astype(np.float32)
        else:
            I_raw = I_raw[:CPI_SAMPLES]
            Q_raw = Q_raw[:CPI_SAMPLES]

        # Form complex IQ matrix [32 x 128]
        iq_complex = (I_raw + 1j * Q_raw).reshape(NUM_PULSES, SAMPLES_PER_PULSE)

        # DC removal (remove mean per pulse)
        iq_complex -= iq_complex.mean(axis=1, keepdims=True)

        signal_power = np.mean(np.abs(iq_complex) ** 2)
        print(f"    Signal power : {10*np.log10(signal_power + 1e-12):.1f} dB")

        if preview:
            _preview_signal(iq_complex)

        # Save
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"hb100_audio_frame{frame_idx}_{ts}.npy"
            path = output_dir / name
        else:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"hb100_audio_{ts}.npy"
            path = Path(name)

        np.save(path, iq_complex)
        print(f"    Saved -> {path}  (shape {iq_complex.shape})")
        saved_files.append(path)

        if frame_idx < n_frames - 1:
            time.sleep(0.1)

    return saved_files


# ──────────────────────────────────────────────────────────────────────────────
# Arduino / Serial capture (Option B)
# ──────────────────────────────────────────────────────────────────────────────

def capture_serial(port: str, baud: int = 115200, n_frames: int = 1,
                   output_dir: Path = None, adc_bits: int = 10,
                   adc_vref: float = 5.0):
    try:
        import serial as pyserial
    except ImportError:
        print("Install pyserial: pip install pyserial")
        return []

    print(f"\nConnecting to Arduino on {port} at {baud} baud ...")

    try:
        ser = pyserial.Serial(port, baud, timeout=2)
    except Exception as e:
        print(f"Cannot open {port}: {e}")
        print("\nAvailable COM ports:")
        try:
            from serial.tools import list_ports
            for p in list_ports.comports():
                print(f"  {p.device}  —  {p.description}")
        except Exception:
            pass
        return []

    time.sleep(2)   # Arduino resets on serial connect
    ser.flushInput()
    print(f"  Connected. Collecting {CPI_SAMPLES} samples per frame ...")

    adc_max  = 2 ** adc_bits - 1
    adc_mid  = adc_max / 2
    saved_files = []

    for frame_idx in range(n_frames):
        print(f"\n  [Frame {frame_idx+1}/{n_frames}] Reading ... ", end='', flush=True)

        I_vals, Q_vals = [], []
        t_start = time.time()

        while len(I_vals) < CPI_SAMPLES:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line or ',' not in line:
                continue
            try:
                parts = line.split(',')
                I_vals.append(float(parts[0]))
                Q_vals.append(float(parts[1]))
            except (ValueError, IndexError):
                continue

        elapsed = time.time() - t_start
        print(f"done ({elapsed:.2f}s)")

        # Convert ADC counts to voltage, centre around 0
        I_arr = (np.array(I_vals[:CPI_SAMPLES], dtype=np.float32) - adc_mid) / adc_mid
        Q_arr = (np.array(Q_vals[:CPI_SAMPLES], dtype=np.float32) - adc_mid) / adc_mid

        iq_complex = (I_arr + 1j * Q_arr).reshape(NUM_PULSES, SAMPLES_PER_PULSE)
        iq_complex -= iq_complex.mean(axis=1, keepdims=True)   # DC removal

        signal_power = np.mean(np.abs(iq_complex) ** 2)
        print(f"    Signal power : {10*np.log10(signal_power + 1e-12):.1f} dB")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"hb100_serial_frame{frame_idx}_{ts}.npy"
            path = output_dir / name
        else:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = Path(f"hb100_serial_{ts}.npy")

        np.save(path, iq_complex)
        print(f"    Saved -> {path}  (shape {iq_complex.shape})")
        saved_files.append(path)

    ser.close()
    return saved_files


# ──────────────────────────────────────────────────────────────────────────────
# Signal preview (text-based spectrum)
# ──────────────────────────────────────────────────────────────────────────────

def _preview_signal(iq: np.ndarray):
    from scipy.fft import fft, fftshift

    doppler_fft = np.abs(fftshift(fft(iq, axis=0), axes=0))
    profile     = doppler_fft.mean(axis=1)
    profile    /= profile.max() + 1e-10

    print("\n    Doppler spectrum (text preview):")
    print("    " + "-" * 42)
    n_bins = len(profile)
    step   = max(1, n_bins // 16)
    for i in range(0, n_bins, step):
        bar = int(profile[i] * 30)
        freq_label = f"{i - n_bins//2:+4d}"
        print(f"    bin{freq_label} | {'#' * bar}")
    print("    " + "-" * 42)


# ──────────────────────────────────────────────────────────────────────────────
# Quick inference on a saved file
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_on_file(npy_path: Path, model_path: Path = None):
    import torch
    import torch.nn.functional as F
    from src.data.dataset import iq_to_range_doppler
    from src.models.cnn_lstm import build_model

    CLASS_NAMES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']

    if model_path is None:
        model_path = Path('outputs/models/best_model.pt')
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    ckpt  = torch.load(model_path, map_location='cpu', weights_only=False)
    model = build_model(ckpt.get('config', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    iq  = np.load(npy_path, allow_pickle=False)
    rd  = iq_to_range_doppler(iq)

    spec = torch.from_numpy(rd).unsqueeze(0).unsqueeze(0)
    dop  = torch.from_numpy(rd.mean(axis=1)).unsqueeze(0)
    env  = torch.zeros(1, 3)

    with torch.no_grad():
        probs = F.softmax(model(spec, dop, env), dim=1).squeeze().numpy()

    pred = CLASS_NAMES[probs.argmax()]
    conf = probs.max()

    print(f"\n  Prediction : {pred}  ({conf:.1%} confidence)")
    print(f"  All probs  :")
    for i, cls in enumerate(CLASS_NAMES):
        bar = '#' * int(probs[i] * 30)
        print(f"    {cls:10s}  {probs[i]*100:5.1f}%  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Capture real IQ data from HB100 radar'
    )
    parser.add_argument('--mode',        choices=['audio', 'serial'],
                        default='audio', help='Capture method')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio input devices')
    parser.add_argument('--device',      type=int, default=None,
                        help='Audio device ID (use --list-devices to find)')
    parser.add_argument('--port',        default='COM3',
                        help='Serial port for Arduino (e.g. COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud',        type=int, default=115200)
    parser.add_argument('--frames',      type=int, default=1,
                        help='Number of CPI frames to capture')
    parser.add_argument('--output',      default=None,
                        help='Output .npy file path (single frame)')
    parser.add_argument('--output-dir',  default='data/real/hb100_captures',
                        help='Output directory for multiple frames')
    parser.add_argument('--preview',     action='store_true',
                        help='Show text Doppler spectrum after capture')
    parser.add_argument('--infer',       action='store_true',
                        help='Run model inference immediately after capture')
    parser.add_argument('--model',       default='outputs/models/best_model.pt',
                        help='Model path for --infer')
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    out_dir = Path(args.output_dir) if args.frames > 1 else None
    if args.output:
        out_dir = Path(args.output).parent

    print("=" * 60)
    print(" HB100 24 GHz Radar — IQ Capture")
    print("=" * 60)
    print(f" Mode          : {args.mode}")
    print(f" Frames        : {args.frames}")
    print(f" CPI duration  : {CPI_DURATION:.2f} s  ({CPI_SAMPLES} samples)")
    print(f" IQ shape      : ({NUM_PULSES}, {SAMPLES_PER_PULSE}) complex128")
    print(f" Output        : {out_dir or 'current directory'}")

    if args.mode == 'audio':
        if args.device is None:
            print("\nNo device specified. Listing devices:\n")
            list_audio_devices()
            print("\nRe-run with: --device <ID>")
            return
        files = capture_audio(
            device_id=args.device,
            n_frames=args.frames,
            output_dir=out_dir,
            preview=args.preview,
        )
    else:
        files = capture_serial(
            port=args.port,
            baud=args.baud,
            n_frames=args.frames,
            output_dir=out_dir,
        )

    if files and args.infer:
        print("\nRunning model inference ...")
        for f in files:
            print(f"\n  File: {f.name}")
            run_inference_on_file(f, Path(args.model))

    if files:
        print(f"\n{'='*60}")
        print(f" Captured {len(files)} file(s)")
        print(f" Upload to UI: http://localhost:8501")
        print(f" Sidebar -> 'Upload IQ Data' -> select any .npy file")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
