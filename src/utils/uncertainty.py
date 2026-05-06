"""
Uncertainty Estimation for Radar Threat Classification

Implements multiple uncertainty quantification methods:
1. MC Dropout - Monte Carlo Dropout for epistemic uncertainty
2. Temperature Scaling - Calibrated confidence scores
3. Ensemble - Multiple model predictions
4. Deep Ensembles - Training multiple models from scratch

Critical for DRDO deployment where confidence calibration is essential.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MCDropoutWrapper(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.

    Enables dropout during inference to sample from approximate posterior.
    Epistemic uncertainty = variance across samples.
    """

    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        """
        Args:
            model: Base model
            dropout_rate: Dropout probability for MC sampling
        """
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self._add_mc_dropout()

    def _add_mc_dropout(self):
        """Add dropout layers if not present."""
        # Enable dropout in model regardless of training mode
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate

    def forward(self, *args, **kwargs):
        """Standard forward pass."""
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        spectrogram: torch.Tensor,
        doppler_seq: torch.Tensor,
        env_features: torch.Tensor,
        n_samples: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        Make prediction with uncertainty estimation via MC Dropout.

        Args:
            spectrogram: Input spectrogram
            doppler_seq: Doppler sequence
            env_features: Environmental features
            n_samples: Number of forward passes

        Returns:
            Dictionary with predictions and uncertainties
        """
        self.model.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            logits = self.model(spectrogram, doppler_seq, env_features)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs.cpu().numpy())

        self.model.eval()

        predictions = np.array(predictions)  # [n_samples, batch, n_classes]

        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)

        # Epistemic uncertainty (model uncertainty) - variance across samples
        epistemic_var = np.var(predictions, axis=0)
        epistemic_uncertainty = np.mean(epistemic_var, axis=1)  # Per sample

        # Aleatoric uncertainty (data uncertainty) - entropy of mean prediction
        aleatoric_uncertainty = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)

        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # Predicted class
        predicted_class = np.argmax(mean_pred, axis=1)
        confidence = np.max(mean_pred, axis=1)

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': mean_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'prediction_variance': epistemic_var,
            'n_samples': n_samples
        }


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for confidence calibration.

    Learns a single temperature parameter to calibrate model outputs
    without changing predictions.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, *args, **kwargs):
        """Forward with temperature scaling."""
        logits = self.model(*args, **kwargs)
        return logits / self.temperature

    def calibrate(
        self,
        val_loader,
        device: str = 'cuda',
        max_iter: int = 100
    ) -> float:
        """
        Learn optimal temperature on validation set.

        Args:
            val_loader: Validation data loader
            device: Device to use
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        self.model.eval()
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                spectrogram, doppler_seq, env_features, labels = batch
                spectrogram = spectrogram.to(device)
                doppler_seq = doppler_seq.to(device)
                env_features = env_features.to(device)

                logits = self.model(spectrogram, doppler_seq, env_features)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Optimize temperature
        self.temperature.data = torch.ones(1)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_temp():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            loss = F.cross_entropy(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_temp)

        logger.info(f"Calibrated temperature: {self.temperature.item():.4f}")
        return self.temperature.item()

    def get_calibrated_probabilities(
        self,
        spectrogram: torch.Tensor,
        doppler_seq: torch.Tensor,
        env_features: torch.Tensor
    ) -> torch.Tensor:
        """Get calibrated probability estimates."""
        logits = self.forward(spectrogram, doppler_seq, env_features)
        return F.softmax(logits, dim=1)


class EnsembleClassifier(nn.Module):
    """
    Ensemble of models for robust predictions and uncertainty.

    Combines predictions from multiple models trained with different
    random seeds or configurations.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

    @torch.no_grad()
    def forward(
        self,
        spectrogram: torch.Tensor,
        doppler_seq: torch.Tensor,
        env_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass - average of all models."""
        all_logits = []
        for model in self.models:
            model.eval()
            logits = model(spectrogram, doppler_seq, env_features)
            all_logits.append(logits)

        # Average logits
        return torch.stack(all_logits).mean(dim=0)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        spectrogram: torch.Tensor,
        doppler_seq: torch.Tensor,
        env_features: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Predict with ensemble disagreement as uncertainty.
        """
        all_probs = []
        for model in self.models:
            model.eval()
            logits = model(spectrogram, doppler_seq, env_features)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs)  # [n_models, batch, n_classes]

        # Mean prediction
        mean_pred = np.mean(all_probs, axis=0)

        # Disagreement (variance across models)
        ensemble_variance = np.var(all_probs, axis=0)
        disagreement = np.mean(ensemble_variance, axis=1)

        # Predicted class and confidence
        predicted_class = np.argmax(mean_pred, axis=1)
        confidence = np.max(mean_pred, axis=1)

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': mean_pred,
            'ensemble_disagreement': disagreement,
            'individual_predictions': all_probs,
            'n_models': self.n_models
        }


class UncertaintyEstimator:
    """
    Unified interface for uncertainty estimation.

    Combines multiple methods and provides interpretable uncertainty scores.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = 'mc_dropout',
        device: str = 'auto'
    ):
        """
        Args:
            model: Base classification model
            method: 'mc_dropout', 'temperature', 'ensemble'
            device: Device to use
        """
        self.method = method
        self.device = torch.device(
            device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        if method == 'mc_dropout':
            self.estimator = MCDropoutWrapper(model)
        elif method == 'temperature':
            self.estimator = TemperatureScaling(model)
        elif method == 'ensemble' and isinstance(model, list):
            self.estimator = EnsembleClassifier(model)
        else:
            self.estimator = model
            logger.warning(f"Unknown method {method}, using base model")

        self.estimator = self.estimator.to(self.device)

        # Thresholds for uncertainty flags
        self.high_uncertainty_threshold = 0.5
        self.low_confidence_threshold = 0.7

    def predict(
        self,
        spectrogram: torch.Tensor,
        doppler_seq: torch.Tensor,
        env_features: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Dict:
        """
        Make prediction with optional uncertainty.

        Args:
            spectrogram: Input spectrogram
            doppler_seq: Doppler sequence
            env_features: Environmental features
            return_uncertainty: Whether to compute uncertainty

        Returns:
            Prediction dictionary
        """
        # Move to device
        spectrogram = spectrogram.to(self.device)
        doppler_seq = doppler_seq.to(self.device)
        env_features = env_features.to(self.device)

        if return_uncertainty and hasattr(self.estimator, 'predict_with_uncertainty'):
            result = self.estimator.predict_with_uncertainty(
                spectrogram, doppler_seq, env_features
            )
        else:
            self.estimator.eval()
            with torch.no_grad():
                logits = self.estimator(spectrogram, doppler_seq, env_features)
                probs = F.softmax(logits, dim=1)

            result = {
                'predicted_class': probs.argmax(dim=1).cpu().numpy(),
                'confidence': probs.max(dim=1).values.cpu().numpy(),
                'probabilities': probs.cpu().numpy()
            }

        # Add interpretable flags
        result['is_reliable'] = (
            result['confidence'] > self.low_confidence_threshold
        )

        if 'total_uncertainty' in result:
            result['is_reliable'] &= (
                result['total_uncertainty'] < self.high_uncertainty_threshold
            )
            result['requires_review'] = (
                result['total_uncertainty'] > self.high_uncertainty_threshold
            )
        else:
            result['requires_review'] = ~result['is_reliable']

        return result

    def calibrate(self, val_loader, **kwargs):
        """Calibrate the estimator if supported."""
        if isinstance(self.estimator, TemperatureScaling):
            return self.estimator.calibrate(val_loader, **kwargs)
        else:
            logger.info("Calibration not supported for this method")
            return None


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well confidence scores match actual accuracy.

    Args:
        probs: Predicted probabilities [N, n_classes]
        labels: True labels [N]
        n_bins: Number of confidence bins

    Returns:
        ECE value and per-bin statistics
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_stats = []

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

            bin_stats.append({
                'bin_idx': i,
                'lower': bin_boundaries[i],
                'upper': bin_boundaries[i + 1],
                'count': np.sum(in_bin),
                'avg_confidence': avg_confidence,
                'avg_accuracy': avg_accuracy,
                'calibration_gap': avg_accuracy - avg_confidence
            })

    return ece, {'bins': bin_stats, 'n_bins': n_bins}
