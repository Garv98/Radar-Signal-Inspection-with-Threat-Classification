"""
Inference Module for Radar Threat Classification

This module provides the ThreatClassifier class that your DQN teammate
can use to get predictions from the trained model.

Usage Example:
    from src.inference.predictor import ThreatClassifier

    # Load the trained model
    classifier = ThreatClassifier('outputs/models/best_model.pt')

    # Make predictions
    class_name, confidence, probs = classifier.predict(spectrogram, doppler_seq, env_features)

    # Get threat level for RL reward signal
    threat_level = classifier.get_threat_level(class_name)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, Union, List
import os

# Import model - handle relative/absolute imports
try:
    from src.models.cnn_lstm import RadarClassifier, build_model
    from src.data.preprocessing import preprocess_iq_matrix, compute_spectrogram, normalize
    from src.data.feature_extraction import extract_doppler_shift
except ImportError:
    from ..models.cnn_lstm import RadarClassifier, build_model
    from ..data.preprocessing import preprocess_iq_matrix, compute_spectrogram, normalize
    from ..data.feature_extraction import extract_doppler_shift


class ThreatClassifier:
    """
    Threat Classification Interface for DQN Integration.

    This class wraps the trained RadarClassifier model and provides
    a simple interface for making predictions and getting threat levels.
    """

    # Class definitions
    CLASS_NAMES = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

    # Threat levels (0.0 = no threat, 1.0 = maximum threat)
    THREAT_LEVELS = {
        'Drone': 0.9,      # High threat - unauthorized surveillance
        'Aircraft': 0.8,   # High threat - potential collision
        'Bird': 0.2,       # Low threat - may cause false alarms
        'Clutter': 0.1,    # Minimal threat - environmental
        'Noise': 0.0       # No threat - sensor noise
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
        device: Optional[str] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize the threat classifier.

        Args:
            model_path: Path to saved model checkpoint (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
            config: Optional configuration dictionary
        """
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Get config from checkpoint or use provided
        self.config = config or checkpoint.get('config', {})
        configured_thresholds = self.config.get('inference', {}).get('class_confidence_thresholds', {})
        self.confidence_thresholds = {
            name: float(configured_thresholds.get(name, default_threshold))
            for name, default_threshold in self.DEFAULT_CONFIDENCE_THRESHOLDS.items()
        }

        # Build model
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Store model info
        self.model_info = {
            'path': model_path,
            'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
            'metrics': checkpoint.get('metrics', {})
        }

        print(f"Loaded ThreatClassifier from {model_path}")
        print(f"Running on: {self.device}")

    def preprocess(
        self,
        iq_matrix: np.ndarray,
        env_features: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess raw IQ data for inference.

        Args:
            iq_matrix: Raw IQ data [num_pulses x num_samples]
            env_features: Optional environmental features dict

        Returns:
            Tuple of (spectrogram, doppler_seq, env_tensor) ready for model
        """
        # Default config values
        fs = self.config.get('radar', {}).get('sampling_rate', 10000)
        carrier_freq = self.config.get('radar', {}).get('carrier_frequency', 24e9)
        prf = self.config.get('radar', {}).get('prf', 1000)
        target_size = self.config.get('features', {}).get('spectrogram_size', [32, 128])

        # Compute spectrogram
        avg_pulse = np.mean(iq_matrix, axis=0)
        _, _, spec = compute_spectrogram(avg_pulse, fs=fs)

        # Resize to target size
        from scipy.ndimage import zoom
        zoom_factors = (target_size[0] / spec.shape[0], target_size[1] / spec.shape[1])
        spectrogram = zoom(spec, zoom_factors, order=1)
        spectrogram, _ = normalize(spectrogram, method='minmax')

        # Extract Doppler sequence
        doppler_spec, _ = extract_doppler_shift(
            iq_matrix, carrier_freq=carrier_freq, prf=prf, num_bins=32
        )
        doppler_seq, _ = normalize(doppler_spec, method='minmax')

        # Process environmental features
        use_environmental = self.config.get('features', {}).get('use_environmental', True)
        if not use_environmental:
            env_array = np.zeros(3, dtype=np.float32)
        else:
            if env_features is None:
                env_features = {'rain': 0, 'temperature': 20, 'pressure': 1013}

            rain = np.clip(env_features.get('rain', 0) / 100.0, 0, 1)
            temp = np.clip((env_features.get('temperature', 20) + 40) / 90.0, 0, 1)
            pressure = np.clip((env_features.get('pressure', 1013) - 950) / 100.0, 0, 1)
            env_array = np.array([rain, temp, pressure], dtype=np.float32)

        # Convert to tensors with batch dimension
        spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
        doppler_tensor = torch.FloatTensor(doppler_seq).unsqueeze(0).to(self.device)
        env_tensor = torch.FloatTensor(env_array).unsqueeze(0).to(self.device)

        return spec_tensor, doppler_tensor, env_tensor

    @torch.no_grad()
    def predict(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        doppler_seq: Union[np.ndarray, torch.Tensor],
        env_features: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[str, float, np.ndarray]:
        """
        Make a prediction on preprocessed inputs.

        Args:
            spectrogram: Spectrogram tensor [1, 1, H, W] or [H, W]
            doppler_seq: Doppler sequence [1, seq_len] or [seq_len]
            env_features: Environmental features [1, 3] or [3]

        Returns:
            Tuple of (class_name, confidence, class_probabilities)
        """
        # Convert to tensors if numpy
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.FloatTensor(spectrogram)
        if isinstance(doppler_seq, np.ndarray):
            doppler_seq = torch.FloatTensor(doppler_seq)
        if isinstance(env_features, np.ndarray):
            env_features = torch.FloatTensor(env_features)

        # Add batch and channel dimensions if needed
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
        elif spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)

        if doppler_seq.dim() == 1:
            doppler_seq = doppler_seq.unsqueeze(0)

        if env_features.dim() == 1:
            env_features = env_features.unsqueeze(0)

        # Move to device
        spectrogram = spectrogram.to(self.device)
        doppler_seq = doppler_seq.to(self.device)
        env_features = env_features.to(self.device)

        # Forward pass
        logits = self.model(spectrogram, doppler_seq, env_features)
        probs = F.softmax(logits, dim=1)

        # Get prediction
        confidence, predicted_idx = torch.max(probs, dim=1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

        class_name = self.IDX_TO_CLASS[predicted_idx]
        probabilities = probs.cpu().numpy().flatten()

        return class_name, confidence, probabilities

    @torch.no_grad()
    def predict_raw(
        self,
        iq_matrix: np.ndarray,
        env_features: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float, np.ndarray]:
        """
        Make prediction directly from raw IQ data.

        Args:
            iq_matrix: Raw IQ data [num_pulses x num_samples]
            env_features: Optional dict with 'rain', 'temperature', 'pressure'

        Returns:
            Tuple of (class_name, confidence, class_probabilities)
        """
        spec, doppler, env = self.preprocess(iq_matrix, env_features)
        return self.predict(spec, doppler, env)

    @torch.no_grad()
    def predict_batch(
        self,
        spectrograms: Union[np.ndarray, torch.Tensor],
        doppler_seqs: Union[np.ndarray, torch.Tensor],
        env_features: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of inputs.

        Args:
            spectrograms: Batch of spectrograms [batch, 1, H, W]
            doppler_seqs: Batch of Doppler sequences [batch, seq_len]
            env_features: Batch of environmental features [batch, 3]

        Returns:
            Tuple of (class_names, confidences, probabilities)
        """
        # Convert to tensors
        if isinstance(spectrograms, np.ndarray):
            spectrograms = torch.FloatTensor(spectrograms)
        if isinstance(doppler_seqs, np.ndarray):
            doppler_seqs = torch.FloatTensor(doppler_seqs)
        if isinstance(env_features, np.ndarray):
            env_features = torch.FloatTensor(env_features)

        # Move to device
        spectrograms = spectrograms.to(self.device)
        doppler_seqs = doppler_seqs.to(self.device)
        env_features = env_features.to(self.device)

        # Forward pass
        logits = self.model(spectrograms, doppler_seqs, env_features)
        probs = F.softmax(logits, dim=1)

        # Get predictions
        confidences, predicted_indices = torch.max(probs, dim=1)

        class_names = [self.IDX_TO_CLASS[idx.item()] for idx in predicted_indices]
        confidences = confidences.cpu().numpy()
        probabilities = probs.cpu().numpy()

        return class_names, confidences, probabilities

    def get_threat_level(self, class_name: str) -> float:
        """
        Get threat level for a class.

        This can be used as part of the RL reward signal.

        Args:
            class_name: Predicted class name

        Returns:
            Threat level between 0.0 (no threat) and 1.0 (max threat)
        """
        return self.THREAT_LEVELS.get(class_name, 0.0)

    def is_prediction_reliable(self, class_name: str, confidence: float) -> bool:
        """Check if confidence clears the per-class reliability threshold."""
        threshold = self.confidence_thresholds.get(class_name, 0.7)
        return confidence >= threshold

    def get_threat_info(
        self,
        class_name: str,
        confidence: float
    ) -> Dict:
        """
        Get detailed threat information for RL agent.

        Args:
            class_name: Predicted class name
            confidence: Prediction confidence

        Returns:
            Dictionary with threat information
        """
        threat_level = self.get_threat_level(class_name)
        reliability_threshold = self.confidence_thresholds.get(class_name, 0.7)
        is_reliable = confidence >= reliability_threshold

        # Adjust threat level by confidence
        adjusted_threat = threat_level * confidence

        return {
            'class': class_name,
            'class_idx': self.CLASS_TO_IDX[class_name],
            'confidence': confidence,
            'base_threat_level': threat_level,
            'adjusted_threat_level': adjusted_threat,
            'reliability_threshold': reliability_threshold,
            'is_reliable': is_reliable,
            'requires_review': not is_reliable,
            'is_threat': threat_level > 0.5,
            'priority': 'high' if threat_level > 0.7 else ('medium' if threat_level > 0.3 else 'low')
        }

    def get_feature_vector(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        doppler_seq: Union[np.ndarray, torch.Tensor],
        env_features: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Extract feature vector for custom downstream processing.

        Args:
            spectrogram: Input spectrogram
            doppler_seq: Doppler sequence
            env_features: Environmental features

        Returns:
            Combined feature vector from all branches
        """
        # Convert and prepare inputs (same as predict)
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.FloatTensor(spectrogram)
        if isinstance(doppler_seq, np.ndarray):
            doppler_seq = torch.FloatTensor(doppler_seq)
        if isinstance(env_features, np.ndarray):
            env_features = torch.FloatTensor(env_features)

        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
        if doppler_seq.dim() == 1:
            doppler_seq = doppler_seq.unsqueeze(0)
        if env_features.dim() == 1:
            env_features = env_features.unsqueeze(0)

        spectrogram = spectrogram.to(self.device)
        doppler_seq = doppler_seq.to(self.device)
        env_features = env_features.to(self.device)

        with torch.no_grad():
            features = self.model.get_feature_vector(spectrogram, doppler_seq, env_features)

        return features.cpu().numpy().flatten()

    def __repr__(self) -> str:
        return f"ThreatClassifier(device={self.device}, classes={self.CLASS_NAMES})"


def load_classifier(model_path: str, device: Optional[str] = None) -> ThreatClassifier:
    """
    Convenience function to load a classifier.

    Args:
        model_path: Path to model checkpoint
        device: Device to use

    Returns:
        Initialized ThreatClassifier
    """
    return ThreatClassifier(model_path, device)


# Example usage for DQN teammate
if __name__ == "__main__":
    print("=" * 60)
    print("ThreatClassifier Usage Example")
    print("=" * 60)

    print("""
    # Load the trained model
    classifier = ThreatClassifier('outputs/models/best_model.pt')

    # Option 1: Predict from preprocessed data
    class_name, confidence, probs = classifier.predict(
        spectrogram,    # [1, 1, 32, 128] tensor
        doppler_seq,    # [1, 32] tensor
        env_features    # [1, 3] tensor
    )

    # Option 2: Predict from raw IQ data
    class_name, confidence, probs = classifier.predict_raw(
        iq_matrix,                    # [32, 128] complex numpy array
        {'rain': 5, 'temperature': 25, 'pressure': 1013}
    )

    # Get threat level for RL reward signal
    threat_level = classifier.get_threat_level(class_name)

    # Get detailed threat info
    threat_info = classifier.get_threat_info(class_name, confidence)

    # Use adjusted_threat_level as part of RL reward:
    # reward = base_reward - threat_info['adjusted_threat_level'] * penalty_factor
    """)

    # Test with dummy data if no model exists
    print("\nTesting with dummy data...")

    # Create dummy inputs
    spectrogram = torch.randn(1, 1, 32, 128)
    doppler_seq = torch.randn(1, 32)
    env_features = torch.tensor([[0.1, 0.5, 0.5]])  # normalized

    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Doppler sequence shape: {doppler_seq.shape}")
    print(f"Environmental features shape: {env_features.shape}")

    print("\nTo use the classifier:")
    print("  classifier = ThreatClassifier('outputs/models/best_model.pt')")
    print("  class_name, confidence, probs = classifier.predict(spec, doppler, env)")
