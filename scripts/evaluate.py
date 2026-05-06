#!/usr/bin/env python3
"""
Evaluation Script

Evaluate trained model on test set and generate reports.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model outputs/models/best_model.pt
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import RadarDataset, create_dataloaders
from src.models.cnn_lstm import build_model
from src.training.metrics import evaluate_model, plot_confusion_matrix, print_classification_report


def main():
    parser = argparse.ArgumentParser(description='Evaluate Radar Classifier')
    parser.add_argument('--model', type=str, default='outputs/models/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create test dataloader
    dataloaders = create_dataloaders(
        data_dir='data',
        batch_size=config['training']['batch_size'],
        config=config
    )

    # Evaluate
    print("\nEvaluating on test set...")
    print("=" * 60)

    metrics = evaluate_model(
        model,
        dataloaders['test'],
        device=device,
        class_names=RadarDataset.CLASS_NAMES
    )

    # Plot confusion matrix
    cm_path = 'outputs/logs/test_confusion_matrix.png'
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        RadarDataset.CLASS_NAMES,
        save_path=cm_path
    )

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"\nPer-class F1 scores:")
    for cls, f1 in metrics['f1_per_class'].items():
        print(f"  {cls}: {f1:.4f}")


if __name__ == '__main__':
    main()
