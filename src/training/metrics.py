"""
Metrics Module for Model Evaluation

Provides functions for computing classification metrics,
generating confusion matrices, and plotting results.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary containing various metrics
    """
    if class_names is None:
        class_names = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'f1_per_class': dict(zip(class_names[:len(f1_per_class)], f1_per_class)),
        'precision_per_class': dict(zip(class_names[:len(precision_per_class)], precision_per_class)),
        'recall_per_class': dict(zip(class_names[:len(recall_per_class)], recall_per_class))
    }

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        normalize: Whether to normalize by row

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
        fmt = '.2f'
        cm_display = cm_normalized
    else:
        fmt = 'd'
        cm_display = cm

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return fig


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """
    Plot training history curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)

    # Learning rate plot
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_yscale('log')
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate and print classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Classification report string
    """
    if class_names is None:
        class_names = ['Drone', 'Aircraft', 'Bird', 'Clutter', 'Noise']

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names[:len(np.unique(np.concatenate([y_true, y_pred])))],
        zero_division=0
    )

    print("\nClassification Report:")
    print("=" * 60)
    print(report)
    print("=" * 60)

    return report


def evaluate_model(
    model,
    test_loader,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        class_names: Optional class names

    Returns:
        Dictionary with all evaluation results
    """
    import torch

    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            spectrogram, doppler_seq, env_features, labels = batch
            spectrogram = spectrogram.to(device)
            doppler_seq = doppler_seq.to(device)
            env_features = env_features.to(device)

            outputs = model(spectrogram, doppler_seq, env_features)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)
    metrics['probabilities'] = all_probs
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels

    # Print report
    print_classification_report(all_labels, all_preds, class_names)

    return metrics
