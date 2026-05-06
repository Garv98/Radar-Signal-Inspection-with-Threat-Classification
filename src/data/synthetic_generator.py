"""
Synthetic Radar Data Generator

Generates realistic synthetic radar signatures for:
- Aircraft / Fighter Jets (with physics-based Jet Engine Modulation)
- Drones (rotor micro-Doppler)
- Birds (wing-beat micro-Doppler)
- Environmental clutter
- Sensor noise
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Optional
import os
import pickle


# ──────────────────────────────────────────────
# Fighter jet configuration table
# Physics sources:
#   • JEM fundamental = fan_blades × RPM / 60  (Hz)
#   • Velocity → Doppler: fd = 2·v·fc / c
#   • PRF-aliased Doppler: ((fd + PRF/2) mod PRF) − PRF/2
# ──────────────────────────────────────────────
FIGHTER_JET_CONFIGS = {
    # ── Single-engine 4th-gen fighter ──────────────────────────────────────────
    'f16': {
        'description': 'F-16 Fighting Falcon (GE F110)',
        'velocity_range': (200.0, 680.0),   # up to Mach 2.0
        'rcs_range':      (3.0,   15.0),    # m²  (no shaping)
        'twin_engine':    False,
        'fan_blades':     26,               # GE F110 fan stage
        'hp_comp_blades': 52,               # LP+HP compressor (approximate)
        'max_rpm':        8700.0,
        'afterburner_capable': True,
        'base_snr_db':    28.0,
    },
    # ── Stealth 5th-gen fighter ─────────────────────────────────────────────────
    'f22_stealth': {
        'description': 'F-22 Raptor (P&W F119, stealth)',
        'velocity_range': (200.0, 765.0),   # Mach 2.25
        'rcs_range':      (0.0001, 0.005),  # Very low RCS due to RAM + shaping
        'twin_engine':    True,
        'fan_blades':     22,
        'hp_comp_blades': 44,
        'max_rpm':        9000.0,
        'afterburner_capable': True,
        'base_snr_db':    10.0,             # Hard to detect
    },
    # ── Twin-engine 4th-gen air-superiority ───────────────────────────────────
    'su27': {
        'description': 'Su-27 Flanker (AL-31F)',
        'velocity_range': (200.0, 800.0),   # Mach 2.35
        'rcs_range':      (8.0,   25.0),
        'twin_engine':    True,
        'fan_blades':     28,               # AL-31F fan
        'hp_comp_blades': 56,
        'max_rpm':        8000.0,
        'afterburner_capable': True,
        'base_snr_db':    30.0,
    },
    # ── Commercial high-bypass turbofan ────────────────────────────────────────
    'commercial': {
        'description': 'Commercial Airliner (CFM56)',
        'velocity_range': (200.0, 280.0),   # Mach 0.6–0.85
        'rcs_range':      (20.0, 100.0),    # Large RCS
        'twin_engine':    True,
        'fan_blades':     20,               # CFM56 fan
        'hp_comp_blades': 40,
        'max_rpm':        6000.0,
        'afterburner_capable': False,
        'base_snr_db':    35.0,
    },
    # ── Turboprop / propeller aircraft ─────────────────────────────────────────
    'turboprop': {
        'description': 'Turboprop / Military Transport (prop signature)',
        'velocity_range': (120.0, 250.0),
        'rcs_range':      (15.0,  80.0),
        'twin_engine':    False,            # Simplified (could be 4-engine)
        'fan_blades':     6,                # Propeller blades
        'hp_comp_blades': 0,                # No separate HP compressor
        'max_rpm':        1200.0,           # Propeller RPM
        'afterburner_capable': False,
        'base_snr_db':    32.0,
    },
}


class SyntheticRadarGenerator:
    """Generate synthetic radar data for training."""

    CLASSES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

    def __init__(
        self,
        carrier_freq: float = 24e9,
        sampling_rate: float = 10000.0,
        prf: float = 1000.0,
        num_pulses: int = 32,
        num_samples: int = 128,
        seed: Optional[int] = None,
    ):
        self.c = 3e8
        self.carrier_freq = carrier_freq
        self.wavelength = self.c / carrier_freq
        self.sampling_rate = sampling_rate
        self.prf = prf
        self.num_pulses = num_pulses
        self.num_samples = num_samples
        self._nyquist = prf / 2.0          # Slow-time Nyquist

        if seed is not None:
            np.random.seed(seed)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _velocity_to_doppler(self, velocity: float) -> float:
        """True (un-aliased) Doppler shift for a given radial velocity."""
        return (2.0 * velocity * self.carrier_freq) / self.c

    def _alias_doppler(self, fd: float) -> float:
        """Fold fd into the unambiguous PRF window [−PRF/2, +PRF/2]."""
        return ((fd + self._nyquist) % self.prf) - self._nyquist

    def _generate_base_signal(
        self,
        doppler_freq: float,
        rcs: float = 1.0,
        snr_db: float = 20.0,
    ) -> np.ndarray:
        """
        IQ matrix [num_pulses × num_samples] for a point target.
        doppler_freq is the ALIASED Doppler (already within ±PRF/2).
        """
        fast_time = np.arange(self.num_samples) / self.sampling_rate
        slow_time = np.arange(self.num_pulses)  / self.prf

        amplitude = np.sqrt(max(rcs, 1e-9))

        # Random range bin for the target
        range_bin = np.random.randint(10, self.num_samples - 10)

        iq = np.zeros((self.num_pulses, self.num_samples), dtype=complex)
        gaussian_window = signal.windows.gaussian(self.num_samples, std=3)
        target_response = np.roll(gaussian_window, range_bin - self.num_samples // 2)

        for p in range(self.num_pulses):
            doppler_phase = 2.0 * np.pi * doppler_freq * slow_time[p]
            iq[p] = amplitude * target_response * np.exp(1j * doppler_phase)

        # Add AWGN
        sig_power = amplitude ** 2
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            np.random.randn(self.num_pulses, self.num_samples) +
            1j * np.random.randn(self.num_pulses, self.num_samples)
        )
        return iq + noise

    # ──────────────────────────────────────────────────────────────────────────
    # JEM helper
    # ──────────────────────────────────────────────────────────────────────────

    def _add_jem(
        self,
        iq: np.ndarray,
        jet_cfg: dict,
        throttle: float,
        afterburner: bool = False,
    ) -> np.ndarray:
        """
        Inject Jet Engine Modulation sidebands into the IQ matrix.

        Physics:
          f_JEM = n_blades × RPM / 60  Hz
          Sidebands at (body_doppler ± k·f_JEM) appear in the slow-time
          spectrum.  Because slow-time samples at PRF, high f_JEM aliases
          into [−PRF/2, +PRF/2] via the same fold applied to Doppler.

          A real engine creates harmonics k = 1,2,…,K with amplitudes
          decaying ~ 1/k (roughly).  Twin-engine aircraft have two slightly
          different RPMs → two interleaved sideband combs.
        """
        slow_time = np.arange(self.num_pulses) / self.prf
        n_engines = 2 if jet_cfg['twin_engine'] else 1

        for eng in range(n_engines):
            # ±2% RPM variation between engines
            rpm_factor = 1.0 + 0.02 * np.random.randn()
            rps = throttle * jet_cfg['max_rpm'] * rpm_factor / 60.0

            blade_stages = []
            if jet_cfg['fan_blades'] > 0:
                blade_stages.append((jet_cfg['fan_blades'], 1.0))
            if jet_cfg['hp_comp_blades'] > 0:
                blade_stages.append((jet_cfg['hp_comp_blades'], 0.4))

            for n_blades, stage_amplitude in blade_stages:
                f_fund = n_blades * rps   # Hz (physical, may be >> PRF/2)

                # Up to 6 harmonics; amplitude decays as 1/harmonic
                for k in range(1, 7):
                    f_jem = f_fund * k
                    mod_amp = stage_amplitude * 0.08 / k

                    # Apply as slow-time AM → creates sidebands in Doppler FFT
                    jem_mod = mod_amp * np.cos(2.0 * np.pi * f_jem * slow_time)
                    # Broadcast across range samples
                    iq += iq * jem_mod[:, np.newaxis]

        # Afterburner: broadband noise + low-frequency combustion instability
        if afterburner:
            ab_noise_level = 0.15 * np.mean(np.abs(iq))
            ab_noise = ab_noise_level * (
                np.random.randn(self.num_pulses, self.num_samples) +
                1j * np.random.randn(self.num_pulses, self.num_samples)
            )
            # Low-frequency oscillation from combustion dynamics (30–120 Hz)
            f_combustion = np.random.uniform(30.0, 120.0)
            combustion_mod = np.sin(2.0 * np.pi * f_combustion * slow_time)
            ab_noise *= (1.0 + 0.4 * combustion_mod[:, np.newaxis])
            iq += ab_noise

        return iq

    # ──────────────────────────────────────────────────────────────────────────
    # Per-class generators
    # ──────────────────────────────────────────────────────────────────────────

    def generate_drone(
        self,
        velocity: Optional[float] = None,
        num_rotors: int = 4,
        rotor_freq: Optional[float] = None,
        snr_db: float = 15.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Drone: low velocity, prominent rotor micro-Doppler at 50–200 Hz.

        Each rotor produces harmonics (k·f_rotor) with decreasing amplitude.
        Different rotors have slightly different frequencies (manufacturing
        variation), making the spectral pattern distinctive and complex.
        """
        if velocity is None:
            velocity = np.random.uniform(0.0, 30.0)
        if rotor_freq is None:
            rotor_freq = np.random.uniform(50.0, 200.0)

        fd_body = self._alias_doppler(self._velocity_to_doppler(velocity))
        iq = self._generate_base_signal(fd_body, rcs=0.01, snr_db=snr_db)

        slow_time = np.arange(self.num_pulses) / self.prf
        for rotor in range(num_rotors):
            # ±5% frequency spread between rotors
            f_rotor = rotor_freq * (1.0 + 0.05 * np.random.randn())

            # Multiple blade harmonics (DJI-style multirotor: 2 blades)
            for blade in range(2):
                blade_phase = np.pi * blade
                # Fundamental + 3 harmonics
                for k in range(1, 5):
                    mod_amp = 0.25 / k
                    phase_mod = 2.0 * np.pi * k * f_rotor * slow_time + blade_phase
                    iq *= (1.0 + mod_amp * np.exp(1j * phase_mod))[:, np.newaxis]

        return iq, {
            'class': 'Drone',
            'label': self.CLASS_TO_IDX['Drone'],
            'velocity': velocity,
            'rotor_frequency': rotor_freq,
            'num_rotors': num_rotors,
            'snr_db': snr_db,
        }

    def generate_fighter_jet(
        self,
        jet_type: Optional[str] = None,
        maneuver: Optional[str] = None,
        snr_db: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a physics-based fighter jet / aircraft radar signature.

        jet_type choices: 'f16', 'f22_stealth', 'su27', 'commercial',
                          'turboprop', or None (random)
        maneuver choices: 'cruise', 'banking', 'climbing', 'diving',
                          'afterburner', or None (random)
        """
        if jet_type is None:
            jet_type = np.random.choice(
                list(FIGHTER_JET_CONFIGS.keys()),
                p=[0.25, 0.15, 0.25, 0.25, 0.10],
            )
        jet_cfg = FIGHTER_JET_CONFIGS[jet_type]

        if maneuver is None:
            options = ['cruise', 'banking', 'climbing', 'diving']
            if jet_cfg['afterburner_capable']:
                options.append('afterburner')
            maneuver = np.random.choice(options)

        # Throttle level by maneuver
        throttle_map = {
            'cruise':      np.random.uniform(0.60, 0.80),
            'banking':     np.random.uniform(0.70, 0.90),
            'climbing':    np.random.uniform(0.80, 1.00),
            'diving':      np.random.uniform(0.50, 0.75),
            'afterburner': np.random.uniform(0.95, 1.00),
        }
        throttle = throttle_map[maneuver]

        # Physical parameters
        velocity = np.random.uniform(*jet_cfg['velocity_range'])
        rcs = np.random.uniform(*jet_cfg['rcs_range'])
        if snr_db is None:
            snr_db = jet_cfg['base_snr_db'] + np.random.randn() * 3.0

        # Aliased Doppler (high-speed jets fold many times into PRF window)
        fd_true = self._velocity_to_doppler(velocity)
        fd_aliased = self._alias_doppler(fd_true)

        # Maneuver: linear Doppler ramp over the CPI (Coherent Processing Interval)
        slow_time = np.arange(self.num_pulses) / self.prf
        doppler_ramp = {
            'cruise':      0.0,
            'banking':     np.random.uniform(-30.0, 30.0),
            'climbing':    np.random.uniform(10.0, 50.0),
            'diving':      np.random.uniform(-50.0, -10.0),
            'afterburner': np.random.uniform(20.0, 80.0),
        }[maneuver]

        # Build base signal with potential Doppler ramp
        if abs(doppler_ramp) < 1e-3:
            iq = self._generate_base_signal(fd_aliased, rcs=rcs, snr_db=snr_db)
        else:
            # Time-varying Doppler: generate pulse-by-pulse
            range_bin = np.random.randint(10, self.num_samples - 10)
            amplitude = np.sqrt(max(rcs, 1e-9))
            gaussian_window = signal.windows.gaussian(self.num_samples, std=3)
            target_response = np.roll(gaussian_window,
                                      range_bin - self.num_samples // 2)
            sig_power = amplitude ** 2
            noise_power = sig_power / (10 ** (snr_db / 10.0))

            iq = np.zeros((self.num_pulses, self.num_samples), dtype=complex)
            for p in range(self.num_pulses):
                t = slow_time[p]
                # Instantaneous Doppler: aliased body + linear chirp
                inst_fd = fd_aliased + doppler_ramp * t
                phase = 2.0 * np.pi * inst_fd * t
                iq[p] = amplitude * target_response * np.exp(1j * phase)

            noise = np.sqrt(noise_power / 2.0) * (
                np.random.randn(self.num_pulses, self.num_samples) +
                1j * np.random.randn(self.num_pulses, self.num_samples)
            )
            iq += noise

        # Inject JEM sidebands
        use_afterburner = (maneuver == 'afterburner') and jet_cfg['afterburner_capable']
        iq = self._add_jem(iq, jet_cfg, throttle, afterburner=use_afterburner)

        return iq, {
            'class': 'Aircraft',
            'label': self.CLASS_TO_IDX['Aircraft'],
            'jet_type': jet_type,
            'maneuver': maneuver,
            'velocity': velocity,
            'rcs': rcs,
            'throttle': throttle,
            'snr_db': snr_db,
            'doppler_ramp': doppler_ramp,
        }

    def generate_aircraft(
        self,
        velocity: Optional[float] = None,
        rcs: Optional[float] = None,
        snr_db: float = 25.0,
    ) -> Tuple[np.ndarray, Dict]:
        """Wrapper: delegates to the full fighter-jet simulator."""
        return self.generate_fighter_jet()

    def generate_bird(
        self,
        velocity: Optional[float] = None,
        wingbeat_freq: Optional[float] = None,
        snr_db: float = 10.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Bird: sinusoidal wingbeat micro-Doppler at 2–15 Hz.
        The signature is a narrow sinusoidal modulation — very different from
        the multi-harmonic rotor pattern of drones.
        """
        if velocity is None:
            velocity = np.random.uniform(5.0, 25.0)
        if wingbeat_freq is None:
            wingbeat_freq = np.random.uniform(2.0, 15.0)

        rcs = np.random.uniform(0.001, 0.01)
        fd_body = self._alias_doppler(self._velocity_to_doppler(velocity))
        iq = self._generate_base_signal(fd_body, rcs=rcs, snr_db=snr_db)

        slow_time = np.arange(self.num_pulses) / self.prf
        wing_tip_velocity = np.random.uniform(0.5, 2.0)   # m/s peak

        for p in range(self.num_pulses):
            t = slow_time[p]
            # Sinusoidal wing-tip velocity → sinusoidal phase modulation
            v_wing = wing_tip_velocity * np.sin(2.0 * np.pi * wingbeat_freq * t)
            fd_wing = self._velocity_to_doppler(v_wing)
            iq[p] *= np.exp(1j * 2.0 * np.pi * fd_wing * t)

        return iq, {
            'class': 'Bird',
            'label': self.CLASS_TO_IDX['Bird'],
            'velocity': velocity,
            'wingbeat_frequency': wingbeat_freq,
            'rcs': rcs,
            'snr_db': snr_db,
        }

    def generate_clutter(
        self,
        clutter_type: str = 'ground',
        snr_db: float = 5.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Environmental clutter: ground, weather, or sea — no coherent target.
        """
        iq = np.zeros((self.num_pulses, self.num_samples), dtype=complex)

        if clutter_type == 'ground':
            for rb in range(self.num_samples):
                if np.random.random() > 0.7:
                    amp = np.random.exponential(0.1)
                    doppler = np.random.uniform(-5.0, 5.0)
                    for p in range(self.num_pulses):
                        phase = 2.0 * np.pi * doppler * p / self.prf
                        iq[p, rb] = amp * np.exp(1j * phase)

        elif clutter_type == 'weather':
            rain_vel = np.random.uniform(-10.0, 10.0)
            for rb in range(self.num_samples):
                amp = np.random.exponential(0.05)
                dop = rain_vel + np.random.randn() * 2.0
                for p in range(self.num_pulses):
                    phase = 2.0 * np.pi * dop * p / self.prf
                    phase += np.random.uniform(0, 2.0 * np.pi)
                    iq[p, rb] += amp * np.exp(1j * phase)

        elif clutter_type == 'sea':
            bragg_vel = np.sqrt(9.8 * self.wavelength / (2.0 * np.pi))
            for rb in range(self.num_samples):
                if np.random.random() > 0.5:
                    amp = np.random.exponential(0.1)
                    dop = bragg_vel * np.random.choice([-1, 1])
                    dop += np.random.randn() * bragg_vel * 0.3
                    for p in range(self.num_pulses):
                        fd = self._velocity_to_doppler(dop)
                        phase = 2.0 * np.pi * fd * p / self.prf
                        iq[p, rb] = amp * np.exp(1j * phase)

        # Add noise floor
        sig_power = np.mean(np.abs(iq) ** 2) + 1e-10
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            np.random.randn(self.num_pulses, self.num_samples) +
            1j * np.random.randn(self.num_pulses, self.num_samples)
        )
        iq += noise

        clutter_types = ['ground', 'weather', 'sea']
        if clutter_type == 'ground':
            pass
        elif clutter_type not in clutter_types:
            clutter_type = 'ground'

        return iq, {
            'class': 'Clutter',
            'label': self.CLASS_TO_IDX['Clutter'],
            'clutter_type': clutter_type,
            'snr_db': snr_db,
        }

    def generate_noise(
        self,
        noise_type: str = 'thermal',
    ) -> Tuple[np.ndarray, Dict]:
        """Pure sensor noise: thermal, phase, or quantization."""
        if noise_type == 'thermal':
            power = np.random.uniform(0.01, 0.1)
            iq = np.sqrt(power / 2.0) * (
                np.random.randn(self.num_pulses, self.num_samples) +
                1j * np.random.randn(self.num_pulses, self.num_samples)
            )
        elif noise_type == 'phase':
            amp = np.random.uniform(0.01, 0.05)
            phase = np.cumsum(np.random.randn(self.num_pulses, self.num_samples) * 0.1, axis=0)
            iq = amp * np.exp(1j * phase)
        else:  # quantization / mixed
            power = np.random.uniform(0.01, 0.1)
            iq = power * (
                np.random.uniform(-1, 1, (self.num_pulses, self.num_samples)) +
                1j * np.random.uniform(-1, 1, (self.num_pulses, self.num_samples))
            )

        return iq, {
            'class': 'Noise',
            'label': self.CLASS_TO_IDX['Noise'],
            'noise_type': noise_type,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Sample / dataset generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_sample(
        self,
        class_name: str,
        environmental: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        generators = {
            'Drone':    self.generate_drone,
            'Aircraft': self.generate_aircraft,
            'Bird':     self.generate_bird,
            'Clutter':  self.generate_clutter,
            'Noise':    self.generate_noise,
        }
        if class_name not in generators:
            raise ValueError(f"Unknown class: {class_name}")

        iq, metadata = generators[class_name](**kwargs)

        if environmental is None:
            environmental = {
                'rain':        np.random.uniform(0.0, 20.0),
                'temperature': np.random.uniform(-10.0, 40.0),
                'pressure':    np.random.uniform(980.0, 1040.0),
            }
        metadata['environmental'] = environmental
        return iq, metadata

    def generate_dataset(
        self,
        samples_per_class: int = 1000,
        output_dir: str = 'data/synthetic',
        balanced: bool = True,
    ) -> None:
        """Generate and save a complete synthetic dataset (pkl)."""
        os.makedirs(output_dir, exist_ok=True)
        all_samples, all_labels, all_metadata = [], [], []

        for class_name in self.CLASSES:
            print(f"  Generating {samples_per_class} × {class_name} ...")
            for _ in range(samples_per_class):
                iq, meta = self.generate_sample(class_name)
                all_samples.append(iq)
                all_labels.append(meta['label'])
                all_metadata.append(meta)

        dataset = {
            'samples':     np.array(all_samples),
            'labels':      np.array(all_labels),
            'metadata':    all_metadata,
            'class_names': self.CLASSES,
        }
        out_path = os.path.join(output_dir, 'synthetic_dataset.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Dataset saved → {out_path}  ({len(all_samples)} samples)")


# ──────────────────────────────────────────────────────────────────────────────
# Environmental variation helper (unchanged API)
# ──────────────────────────────────────────────────────────────────────────────

def generate_environmental_variations(
    base_sample: np.ndarray,
    rain_levels: list = [0, 5, 10, 20],
    temp_range: Tuple[float, float] = (-10.0, 40.0),
) -> list:
    variations = []
    for rain in rain_levels:
        temp = np.random.uniform(*temp_range)
        pressure = np.random.uniform(980.0, 1040.0)
        modified = base_sample.copy()
        if rain > 0:
            attenuation_linear = 10 ** (-0.01 * rain / 20.0)
            modified *= attenuation_linear
            rain_clutter = 0.01 * rain * (
                np.random.randn(*base_sample.shape) +
                1j * np.random.randn(*base_sample.shape)
            )
            modified += rain_clutter
        variations.append((modified, {'rain': rain, 'temperature': temp, 'pressure': pressure}))
    return variations


if __name__ == '__main__':
    gen = SyntheticRadarGenerator(seed=42)
    for cls in SyntheticRadarGenerator.CLASSES:
        iq, meta = gen.generate_sample(cls)
        print(f"{cls:8s}: shape={iq.shape}, "
              f"vel={meta.get('velocity', 'N/A')}, "
              f"label={meta['label']}")
    gen.generate_dataset(samples_per_class=200, output_dir='data/synthetic')
