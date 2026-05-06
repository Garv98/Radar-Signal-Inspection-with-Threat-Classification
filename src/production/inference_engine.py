"""
Production Real-Time Inference Engine

Optimized for low-latency (<50ms) inference with:
- Model warmup and optimization
- Batch inference support
- Async processing capability
- Performance monitoring
- Thread-safe operation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import time
import threading
from queue import Queue
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import project modules
try:
    from src.models.cnn_lstm import build_model
    from src.utils.logging_utils import PerformanceMonitor, AuditLogger
    from src.utils.uncertainty import UncertaintyEstimator
except ImportError:
    from ..models.cnn_lstm import build_model
    from .logging_utils import PerformanceMonitor, AuditLogger
    from .uncertainty import UncertaintyEstimator


@dataclass
class InferenceResult:
    """Structured inference result."""
    input_id: str
    class_name: str
    class_idx: int
    confidence: float
    probabilities: np.ndarray
    threat_level: float
    adjusted_threat_level: float
    reliability_threshold: float
    latency_ms: float
    is_reliable: bool
    uncertainty: Optional[float] = None
    requires_review: bool = False
    metadata: Optional[Dict] = None


class ProductionInferenceEngine:
    """
    Production-grade inference engine for DRDO deployment.

    Features:
    - Model optimization (JIT compilation, quantization)
    - Warmup for consistent latency
    - Performance monitoring and SLA checking
    - Audit logging for compliance
    - Thread-safe operation
    - Uncertainty estimation
    """

    CLASS_NAMES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']
    THREAT_LEVELS = {
        'Drone': 0.9,
        'Aircraft': 0.8,
        'Bird': 0.2,
        'Clutter': 0.1,
        'Noise': 0.0
    }
    DEFAULT_CONFIDENCE_THRESHOLDS = {
        'Drone': 0.80,
        'Aircraft': 0.75,
        'Bird': 0.60,
        'Clutter': 0.55,
        'Noise': 0.50
    }

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict] = None,
        device: str = 'auto',
        optimize: bool = True,
        enable_uncertainty: bool = True,
        sla_latency_ms: float = 50.0,
        audit_log: bool = True
    ):
        """
        Initialize production inference engine.

        Args:
            model_path: Path to model checkpoint
            config: Model configuration
            device: 'cuda', 'cpu', or 'auto'
            optimize: Enable model optimization
            enable_uncertainty: Enable uncertainty estimation
            sla_latency_ms: SLA latency requirement
            audit_log: Enable audit logging
        """
        self.model_path = Path(model_path)
        self.sla_latency_ms = sla_latency_ms

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing inference engine on {self.device}")

        # Load model
        self._load_model(config)

        # Optimize if requested
        if optimize:
            self._optimize_model()

        # Setup monitoring
        self.perf_monitor = PerformanceMonitor('inference')
        self.audit_logger = AuditLogger() if audit_log else None
        configured_thresholds = self.config.get('inference', {}).get('class_confidence_thresholds', {})
        self.confidence_thresholds = {
            name: float(configured_thresholds.get(name, default_threshold))
            for name, default_threshold in self.DEFAULT_CONFIDENCE_THRESHOLDS.items()
        }

        # Setup uncertainty estimation
        if enable_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                self.model, method='mc_dropout', device=str(self.device)
            )
        else:
            self.uncertainty_estimator = None

        # Thread lock for concurrent access
        self._lock = threading.Lock()

        # Warmup model
        self._warmup()

        logger.info("Inference engine ready")

    def _get_reliability_threshold(self, class_name: str) -> float:
        """Get confidence threshold used to mark predictions as reliable."""
        return float(self.confidence_thresholds.get(class_name, 0.7))

    def _load_model(self, config: Optional[Dict]):
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Get config from checkpoint or argument
        self.config = config or checkpoint.get('config', {})

        # Build and load model
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Store model info
        self.model_info = {
            'path': str(self.model_path),
            'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
            'epoch': checkpoint.get('epoch', 'N/A')
        }

        logger.info(f"Loaded model from {self.model_path}")

    def _optimize_model(self):
        """Apply model optimizations for inference."""
        try:
            # Try TorchScript compilation
            self.model = torch.jit.script(self.model)
            logger.info("Model compiled with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")
            try:
                # Fallback to trace-based compilation
                dummy_spec = torch.randn(1, 1, 32, 128).to(self.device)
                dummy_doppler = torch.randn(1, 32).to(self.device)
                dummy_env = torch.randn(1, 3).to(self.device)

                self.model = torch.jit.trace(
                    self.model, (dummy_spec, dummy_doppler, dummy_env)
                )
                logger.info("Model compiled with TorchScript trace")
            except Exception as e2:
                logger.warning(f"TorchScript trace failed: {e2}")

        # Enable cudnn benchmark for consistent CUDA performance
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            logger.info("CUDNN benchmark enabled")

    def _warmup(self, n_iterations: int = 10):
        """Warmup model for consistent latency."""
        logger.info(f"Warming up model with {n_iterations} iterations...")

        dummy_spec = torch.randn(1, 1, 32, 128).to(self.device)
        dummy_doppler = torch.randn(1, 32).to(self.device)
        dummy_env = torch.randn(1, 3).to(self.device)

        warmup_times = []
        with torch.no_grad():
            for _ in range(n_iterations):
                start = time.perf_counter()
                _ = self.model(dummy_spec, dummy_doppler, dummy_env)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                warmup_times.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(warmup_times[2:])  # Skip first few
        logger.info(f"Warmup complete. Avg latency: {avg_latency:.2f}ms")

    @torch.no_grad()
    def predict(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        doppler_seq: Union[np.ndarray, torch.Tensor],
        env_features: Union[np.ndarray, torch.Tensor],
        input_id: str = 'unknown',
        with_uncertainty: bool = False
    ) -> InferenceResult:
        """
        Make a single prediction.

        Args:
            spectrogram: Input spectrogram [1, 1, H, W] or [H, W]
            doppler_seq: Doppler sequence [1, seq_len] or [seq_len]
            env_features: Environmental features [1, 3] or [3]
            input_id: Identifier for audit logging
            with_uncertainty: Compute uncertainty estimate

        Returns:
            InferenceResult with prediction details
        """
        start_time = time.perf_counter()

        with self._lock:
            # Prepare inputs
            spec_tensor = self._prepare_tensor(spectrogram, [1, 1, 32, 128])
            doppler_tensor = self._prepare_tensor(doppler_seq, [1, 32])
            env_tensor = self._prepare_tensor(env_features, [1, 3])

            # Inference
            if with_uncertainty and self.uncertainty_estimator:
                result = self.uncertainty_estimator.predict(
                    spec_tensor, doppler_tensor, env_tensor, return_uncertainty=True
                )
                class_idx = int(result['predicted_class'][0])
                confidence = float(result['confidence'][0])
                probs = result['probabilities'][0]
                uncertainty = float(result.get('total_uncertainty', [0])[0])
                uncertainty_reliable = bool(result['is_reliable'][0])
            else:
                logits = self.model(spec_tensor, doppler_tensor, env_tensor)
                probs = F.softmax(logits, dim=1)

                confidence, class_idx = torch.max(probs, dim=1)
                class_idx = class_idx.item()
                confidence = confidence.item()
                probs = probs.cpu().numpy()[0]
                uncertainty = None
                uncertainty_reliable = True

        # Synchronize for accurate timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Get class info
        class_name = self.CLASS_NAMES[class_idx]
        threat_level = self.THREAT_LEVELS[class_name]
        adjusted_threat_level = threat_level * confidence
        reliability_threshold = self._get_reliability_threshold(class_name)
        threshold_reliable = confidence >= reliability_threshold
        is_reliable = bool(uncertainty_reliable and threshold_reliable)
        requires_review = not is_reliable

        # Record metrics
        self.perf_monitor.record(latency_ms)

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_prediction(
                input_id=input_id,
                prediction=class_name,
                confidence=confidence,
                threat_level=adjusted_threat_level,
                latency_ms=latency_ms,
                metadata={
                    'uncertainty': uncertainty,
                    'is_reliable': is_reliable,
                    'base_threat_level': threat_level,
                    'adjusted_threat_level': adjusted_threat_level,
                    'reliability_threshold': reliability_threshold,
                    'model_path': self.model_info.get('path'),
                    'model_epoch': self.model_info.get('epoch'),
                    'device': str(self.device)
                }
            )

        return InferenceResult(
            input_id=input_id,
            class_name=class_name,
            class_idx=class_idx,
            confidence=confidence,
            probabilities=probs,
            threat_level=threat_level,
            adjusted_threat_level=adjusted_threat_level,
            reliability_threshold=reliability_threshold,
            latency_ms=latency_ms,
            is_reliable=is_reliable,
            uncertainty=uncertainty,
            requires_review=requires_review
        )

    @torch.no_grad()
    def predict_batch(
        self,
        spectrograms: Union[np.ndarray, torch.Tensor],
        doppler_seqs: Union[np.ndarray, torch.Tensor],
        env_features: Union[np.ndarray, torch.Tensor]
    ) -> List[Dict]:
        """
        Batch prediction for higher throughput.

        Args:
            spectrograms: Batch of spectrograms [B, 1, H, W]
            doppler_seqs: Batch of Doppler sequences [B, seq_len]
            env_features: Batch of environmental features [B, 3]

        Returns:
            List of prediction dictionaries
        """
        start_time = time.perf_counter()

        with self._lock:
            # Prepare batch
            spec_tensor = self._prepare_tensor(spectrograms)
            doppler_tensor = self._prepare_tensor(doppler_seqs)
            env_tensor = self._prepare_tensor(env_features)

            batch_size = spec_tensor.shape[0]

            # Batch inference
            logits = self.model(spec_tensor, doppler_tensor, env_tensor)
            probs = F.softmax(logits, dim=1)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        self.perf_monitor.record(latency_ms, batch_size)

        # Process results
        probs_np = probs.cpu().numpy()
        class_indices = np.argmax(probs_np, axis=1)
        confidences = np.max(probs_np, axis=1)

        results = []
        for i in range(batch_size):
            class_name = self.CLASS_NAMES[class_indices[i]]
            base_threat_level = self.THREAT_LEVELS[class_name]
            adjusted_threat_level = base_threat_level * float(confidences[i])
            reliability_threshold = self._get_reliability_threshold(class_name)
            results.append({
                'class_name': class_name,
                'class_idx': int(class_indices[i]),
                'confidence': float(confidences[i]),
                'probabilities': probs_np[i],
                'threat_level': base_threat_level,
                'adjusted_threat_level': adjusted_threat_level,
                'reliability_threshold': reliability_threshold,
                'is_reliable': float(confidences[i]) >= reliability_threshold,
                'requires_review': float(confidences[i]) < reliability_threshold
            })

        return results

    def _prepare_tensor(
        self,
        data: Union[np.ndarray, torch.Tensor],
        target_shape: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Prepare input tensor with proper shape and device."""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        else:
            tensor = data.float()

        # Add dimensions if needed
        if target_shape:
            while tensor.dim() < len(target_shape):
                tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def check_sla(self) -> Dict:
        """Check if performance meets SLA requirements."""
        stats = self.perf_monitor.get_statistics()
        meets_sla = self.perf_monitor.check_sla(self.sla_latency_ms)

        return {
            'meets_sla': meets_sla,
            'sla_requirement_ms': self.sla_latency_ms,
            'statistics': stats
        }

    def get_performance_report(self) -> Dict:
        """Get detailed performance report."""
        self.perf_monitor.log_report()
        return self.perf_monitor.get_statistics()

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            **self.model_info,
            'device': str(self.device),
            'classes': self.CLASS_NAMES,
            'threat_levels': self.THREAT_LEVELS
        }


class AsyncInferenceEngine:
    """
    Asynchronous inference engine for high-throughput scenarios.

    Uses a request queue and worker thread for non-blocking inference.
    """

    def __init__(self, model_path: str, **kwargs):
        self.engine = ProductionInferenceEngine(model_path, **kwargs)
        self.request_queue = Queue()
        self.result_queue = Queue()
        self._running = False
        self._worker_thread = None

    def start(self):
        """Start the async worker."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Async inference engine started")

    def stop(self):
        """Stop the async worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("Async inference engine stopped")

    def _worker_loop(self):
        """Worker thread loop."""
        while self._running:
            try:
                request = self.request_queue.get(timeout=1.0)
                result = self.engine.predict(**request)
                self.result_queue.put(result)
            except Exception:
                continue

    def submit(
        self,
        spectrogram,
        doppler_seq,
        env_features,
        input_id: str = 'unknown'
    ):
        """Submit inference request (non-blocking)."""
        self.request_queue.put({
            'spectrogram': spectrogram,
            'doppler_seq': doppler_seq,
            'env_features': env_features,
            'input_id': input_id
        })

    def get_result(self, timeout: float = None) -> Optional[InferenceResult]:
        """Get inference result (blocking)."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Exception:
            return None


def load_production_engine(
    model_path: str = 'outputs/models/best_model.pt',
    **kwargs
) -> ProductionInferenceEngine:
    """
    Convenience function to load production inference engine.

    Args:
        model_path: Path to model checkpoint
        **kwargs: Additional arguments for ProductionInferenceEngine

    Returns:
        Initialized ProductionInferenceEngine
    """
    return ProductionInferenceEngine(model_path, **kwargs)
