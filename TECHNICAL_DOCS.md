# DRDO-Level Radar Threat Classification System

## Technical Documentation

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Dataset Acquisition](#dataset-acquisition)
5. [Training Pipeline](#training-pipeline)
6. [Inference API](#inference-api)
7. [Production Deployment](#production-deployment)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Uncertainty Quantification](#uncertainty-quantification)
10. [Integration with DQN Agent](#integration-with-dqn-agent)

---

## System Overview

This system provides radar-based threat classification for the RL-Driven Radar Signal Inspection project. It classifies radar returns into 5 threat categories:

| Class | Threat Level | Priority |
|-------|-------------|----------|
| Drone | 0.9 | High |
| Aircraft | 0.8 | High |
| Bird | 0.2 | Low |
| Clutter | 0.1 | Minimal |
| Noise | 0.0 | None |

### Key Features

- **CNN-LSTM Hybrid Architecture**: Processes spectrograms (spatial) and Doppler sequences (temporal)
- **Environmental Factor Integration**: Rain, temperature, pressure affect classification
- **Uncertainty Estimation**: MC Dropout and temperature scaling for confidence calibration
- **Real-Time Inference**: <50ms latency for production deployment
- **MLflow Tracking**: Complete experiment reproducibility
- **Production-Grade Logging**: Audit trails for DRDO compliance

---

## Architecture

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RadarClassifier                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │ Spectrogram │    │   Doppler   │    │ Environment │            │
│  │  [1,32,128] │    │    [32]     │    │    [3]      │            │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘            │
│         │                  │                  │                    │
│         ▼                  ▼                  ▼                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │ CNN Branch  │    │ LSTM Branch │    │  FC Branch  │            │
│  │ Conv2D×3    │    │ LSTM×2      │    │ Linear      │            │
│  │ BatchNorm   │    │ hidden=64   │    │ 3 → 16      │            │
│  │ MaxPool     │    │             │    │             │            │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘            │
│         │                  │                  │                    │
│         └──────────────────┼──────────────────┘                    │
│                            │                                       │
│                            ▼                                       │
│                    ┌─────────────┐                                 │
│                    │ Concatenate │                                 │
│                    │ 2048+64+16  │                                 │
│                    └──────┬──────┘                                 │
│                           │                                        │
│                           ▼                                        │
│                    ┌─────────────┐                                 │
│                    │ Classifier  │                                 │
│                    │ FC 2128→256 │                                 │
│                    │ Dropout 0.3 │                                 │
│                    │ FC 256→5    │                                 │
│                    └──────┬──────┘                                 │
│                           │                                        │
│                           ▼                                        │
│                     [5 Classes]                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Total Parameters: ~722,453
```

### Project Structure

```
IDP/
├── configs/
│   └── config.yaml              # Configuration
│
├── data/
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Preprocessed data
│   ├── synthetic/               # Generated samples
│   └── real/                    # Real radar data
│       ├── mafat/              # MAFAT Challenge
│       ├── bistatic_uav/       # Bistatic UAV RD
│       └── dronerf/            # DroneRF
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Signal processing
│   │   ├── feature_extraction.py # Doppler, amplitude
│   │   ├── synthetic_generator.py # Generate samples
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── real_datasets/       # Real data loaders
│   │
│   ├── models/
│   │   └── cnn_lstm.py          # Model architecture
│   │
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   └── metrics.py           # Evaluation metrics
│   │
│   ├── inference/
│   │   └── predictor.py         # ThreatClassifier API
│   │
│   ├── production/
│   │   └── inference_engine.py  # Real-time inference
│   │
│   └── utils/
│       ├── logging_utils.py     # Structured logging
│       ├── experiment_tracking.py # MLflow
│       └── uncertainty.py       # Calibration
│
├── scripts/
│   ├── prepare_data.py          # Data pipeline
│   ├── train.py                 # Basic training
│   ├── train_production.py      # Production training
│   └── evaluate.py              # Evaluation
│
├── outputs/
│   ├── models/                  # Checkpoints
│   ├── mlruns/                  # MLflow experiments
│   ├── logs/                    # Training logs
│   └── model_registry/          # Model versions
│
├── DATASETS.md                  # Dataset guide
└── requirements.txt             # Dependencies
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
cd IDP

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from src.models.cnn_lstm import build_model; print('Model OK')"
```

---

## Dataset Acquisition

### Option 1: Synthetic Data (Quick Start)

```bash
python scripts/prepare_data.py --samples 1000
```

### Option 2: MAFAT Radar Challenge (Recommended)

1. Apply at: https://mafatchallenge.mod.gov.il/#ApplicationForm
2. Wait for approval (2-5 days)
3. Download from CodaLab
4. Extract to `data/real/mafat/`

### Option 3: Bistatic UAV Dataset

1. Subscribe to IEEE DataPort
2. Download from: https://ieee-dataport.org/documents/bistatic-radar-uav-target-rd-dataset
3. Extract to `data/real/bistatic_uav/`

### Dataset Instructions

```bash
python scripts/train_production.py --help-data
```

---

## Training Pipeline

### Basic Training

```bash
python scripts/train.py
```

### Production Training with MLflow

```bash
# With synthetic data
python scripts/train_production.py --epochs 50

# With real data (if available)
python scripts/train_production.py --data real --epochs 100

# Custom experiment
python scripts/train_production.py \
    --experiment_name drone_detection_v2 \
    --run_name baseline_cnn_lstm \
    --epochs 100 \
    --batch_size 64
```

### Hyperparameter Tuning

Edit `configs/config.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler:
    type: "reduce_on_plateau"
    patience: 5
    factor: 0.5
  early_stopping_patience: 15

model:
  cnn_channels: [32, 64, 128]
  lstm_hidden: 64
  lstm_layers: 2
  dropout: 0.3
```

### View Training Metrics

```bash
# Start MLflow UI
mlflow ui --backend-store-uri outputs/mlruns

# Open browser: http://localhost:5000
```

---

## Inference API

### Basic Usage (ThreatClassifier)

```python
from src.inference.predictor import ThreatClassifier

# Load model
classifier = ThreatClassifier('outputs/models/best_model.pt')

# Predict from preprocessed data
class_name, confidence, probs = classifier.predict(
    spectrogram,    # [1, 1, 32, 128]
    doppler_seq,    # [1, 32]
    env_features    # [1, 3] - [rain, temp, pressure] normalized
)

# Get threat level for RL reward
threat_level = classifier.get_threat_level(class_name)

# Get detailed threat info
threat_info = classifier.get_threat_info(class_name, confidence)
# Returns: {
#   'class': 'Drone',
#   'confidence': 0.95,
#   'base_threat_level': 0.9,
#   'adjusted_threat_level': 0.855,  # confidence * threat
#   'is_threat': True,
#   'priority': 'high'
# }
```

### Raw IQ Data Processing

```python
# Predict from raw IQ matrix
class_name, confidence, probs = classifier.predict_raw(
    iq_matrix,  # [32, 128] complex numpy array
    {'rain': 5, 'temperature': 25, 'pressure': 1013}
)
```

---

## Production Deployment

### Real-Time Inference Engine

```python
from src.production import ProductionInferenceEngine, load_production_engine

# Initialize with optimizations
engine = load_production_engine(
    'outputs/models/best_model.pt',
    optimize=True,           # JIT compilation
    enable_uncertainty=True,  # MC Dropout
    sla_latency_ms=50.0      # SLA requirement
)

# Single prediction
result = engine.predict(
    spectrogram, doppler_seq, env_features,
    input_id='scan_001',
    with_uncertainty=True
)

print(f"Class: {result.class_name}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Threat Level: {result.threat_level}")
print(f"Latency: {result.latency_ms:.2f}ms")
print(f"Reliable: {result.is_reliable}")
print(f"Uncertainty: {result.uncertainty:.3f}")

# Batch prediction (higher throughput)
results = engine.predict_batch(
    spectrograms,  # [B, 1, 32, 128]
    doppler_seqs,  # [B, 32]
    env_features   # [B, 3]
)

# Check SLA compliance
sla_report = engine.check_sla()
print(f"Meets SLA: {sla_report['meets_sla']}")
```

### Async Processing

```python
from src.production import AsyncInferenceEngine

# Start async engine
async_engine = AsyncInferenceEngine('outputs/models/best_model.pt')
async_engine.start()

# Submit requests (non-blocking)
async_engine.submit(spec, doppler, env, input_id='scan_001')
async_engine.submit(spec, doppler, env, input_id='scan_002')

# Get results
result1 = async_engine.get_result(timeout=1.0)
result2 = async_engine.get_result(timeout=1.0)

# Stop engine
async_engine.stop()
```

---

## Performance Benchmarks

### Target Specifications

| Metric | Target | Achieved |
|--------|--------|----------|
| Test Accuracy | >85% | 93.6% |
| F1 (macro) | >0.80 | 0.936 |
| Inference Latency (CPU) | <100ms | ~15ms |
| Inference Latency (GPU) | <50ms | ~5ms |
| Batch Throughput (GPU) | >100/sec | ~500/sec |
| Model Size | <50MB | 8.7MB |

### Latency Breakdown

```
CPU Inference (batch=1):
  - Preprocessing: ~2ms
  - Model Forward: ~10ms
  - Postprocessing: ~1ms
  - Total: ~13ms

GPU Inference (batch=1):
  - Preprocessing: ~2ms
  - Model Forward: ~3ms
  - Postprocessing: ~1ms
  - Total: ~6ms
```

---

## Uncertainty Quantification

### MC Dropout

```python
from src.utils.uncertainty import UncertaintyEstimator

estimator = UncertaintyEstimator(
    model,
    method='mc_dropout',
    device='cuda'
)

result = estimator.predict(
    spectrogram, doppler_seq, env_features,
    return_uncertainty=True
)

# Epistemic uncertainty (model uncertainty)
print(f"Epistemic: {result['epistemic_uncertainty']}")

# Aleatoric uncertainty (data uncertainty)
print(f"Aleatoric: {result['aleatoric_uncertainty']}")

# Total uncertainty
print(f"Total: {result['total_uncertainty']}")

# Reliability flag
print(f"Reliable: {result['is_reliable']}")
```

### Temperature Scaling (Calibration)

```python
from src.utils.uncertainty import TemperatureScaling

# Calibrate on validation set
calibrated_model = TemperatureScaling(model)
temperature = calibrated_model.calibrate(val_loader)

# Get calibrated probabilities
probs = calibrated_model.get_calibrated_probabilities(
    spectrogram, doppler_seq, env_features
)
```

---

## Integration with DQN Agent

### For Your DQN Teammate

```python
from src.inference.predictor import ThreatClassifier

class RadarEnvironment:
    def __init__(self):
        # Load the trained classifier
        self.classifier = ThreatClassifier('outputs/models/best_model.pt')

    def get_threat_assessment(self, radar_scan):
        """
        Process radar scan and return threat assessment for RL agent.

        Args:
            radar_scan: Dict with 'iq_matrix', 'env_features'

        Returns:
            threat_info: Dict for reward calculation
        """
        class_name, confidence, probs = self.classifier.predict_raw(
            radar_scan['iq_matrix'],
            radar_scan.get('env_features')
        )

        return self.classifier.get_threat_info(class_name, confidence)

    def calculate_reward(self, threat_info, action_taken):
        """
        Calculate RL reward based on threat and action.

        Higher reward for:
        - Correctly identifying high threats and taking action
        - Ignoring low-threat objects
        """
        threat = threat_info['adjusted_threat_level']

        if action_taken == 'investigate':
            # Reward investigation of threats, penalize false alarms
            reward = threat - 0.5
        elif action_taken == 'ignore':
            # Penalize ignoring threats, reward ignoring noise
            reward = 0.5 - threat
        else:
            reward = 0.0

        return reward

# Example usage in RL training loop
env = RadarEnvironment()

for episode in range(num_episodes):
    radar_scan = get_next_scan()

    # Get threat assessment from classifier
    threat_info = env.get_threat_assessment(radar_scan)

    # State for DQN includes threat info
    state = np.concatenate([
        radar_scan['features'],
        [threat_info['adjusted_threat_level']],
        threat_info['class_probabilities']
    ])

    # DQN selects action
    action = dqn.select_action(state)

    # Calculate reward
    reward = env.calculate_reward(threat_info, action)

    # RL update...
```

### Feature Vector for DQN

```python
# Get feature vector for custom processing
feature_vector = classifier.get_feature_vector(
    spectrogram, doppler_seq, env_features
)

# Returns: np.array of shape [2128] - concatenated CNN+LSTM+Env features
# Can be used directly as DQN state input
```

---

## Audit and Compliance

### Audit Logging

All predictions are logged to `outputs/logs/audit.jsonl`:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "input_id": "scan_001",
  "prediction": "Drone",
  "confidence": 0.95,
  "threat_level": 0.855,
  "latency_ms": 12.3,
  "metadata": {"uncertainty": 0.05, "is_reliable": true}
}
```

### Performance Monitoring

```python
# Get performance statistics
stats = engine.perf_monitor.get_statistics()
print(f"Mean latency: {stats['mean_latency_ms']:.2f}ms")
print(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")
print(f"Throughput: {stats['mean_throughput']:.1f} samples/sec")
```

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch_size in config.yaml
   - Use `--device cpu` flag

2. **PyTorch version mismatch**
   - Ensure `weights_only=False` in torch.load()
   - Update PyTorch: `pip install --upgrade torch`

3. **DataLoader issues on Windows**
   - Set `num_workers: 0` in config.yaml

4. **MLflow tracking errors**
   - Use `--no-tracking` flag
   - Check `outputs/mlruns/` permissions

---

## Contact

For issues or questions:
- Create issue at project repository
- Contact project maintainer

---

*Document Version: 1.0*
*Last Updated: 2024*
