#!/usr/bin/env python3
"""
Training Script for Radar Threat Classifier

This script handles the complete training pipeline:
1. Generate synthetic data (if not exists)
2. Create data loaders
3. Build model
4. Train with validation
5. Save best model

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --epochs 50 --batch_size 64
"""

import argparse
import os
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_generator import SyntheticRadarGenerator
from src.data.dataset import RadarDataset, create_dataloaders
from src.models.cnn_lstm import RadarClassifier, build_model, count_parameters
from src.training.trainer import Trainer, train_model
from src.training.metrics import plot_training_history, plot_confusion_matrix, evaluate_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_data(config: dict, force: bool = False):
    """Generate synthetic training data if not exists."""
    synthetic_dir = os.path.join(config['paths']['data_raw'], '..', 'synthetic')
    synthetic_path = os.path.join(synthetic_dir, 'synthetic_dataset.pkl')

    if os.path.exists(synthetic_path) and not force:
        print(f"Synthetic data already exists at {synthetic_path}")
        return

    print("Generating synthetic training data...")

    radar_config = config.get('radar', {})
    synthetic_config = config.get('synthetic', {})

    generator = SyntheticRadarGenerator(
        carrier_freq=radar_config.get('carrier_frequency', 24e9),
        sampling_rate=radar_config.get('sampling_rate', 10000),
        prf=radar_config.get('prf', 1000),
        num_pulses=radar_config.get('num_pulses', 32),
        num_samples=radar_config.get('num_samples', 128),
        seed=config.get('dataset', {}).get('random_seed', 42)
    )

    samples_per_class = synthetic_config.get('num_samples_per_class', 1000)
    generator.generate_dataset(
        samples_per_class=samples_per_class,
        output_dir=synthetic_dir
    )

    print(f"Generated {samples_per_class * 5} total samples")


def main():
    parser = argparse.ArgumentParser(description='Train Radar Threat Classifier')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--generate_data', action='store_true',
                        help='Force regenerate synthetic data')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Override samples per class for synthetic data')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.samples_per_class:
        config['synthetic']['num_samples_per_class'] = args.samples_per_class

    # Determine device
    if args.device:
        device = args.device
    elif config.get('hardware', {}).get('device') == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.get('hardware', {}).get('device', 'cpu')

    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(config['paths']['models'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)

    # Generate synthetic data if needed
    generate_data(config, force=args.generate_data)

    # Create data loaders
    print("\nCreating data loaders...")
    data_dir = 'data'
    dataloaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        config=config,
        augment_train=config.get('augmentation', {}).get('enabled', True)
    )

    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    print(f"Model parameters: {count_parameters(model):,}")

    # Create trainer and train
    print("\nStarting training...")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config,
        device=device
    )

    history = trainer.train()

    # Plot and save training history
    print("\nSaving training history...")
    history_path = os.path.join(config['paths']['logs'], 'training_history.png')
    plot_training_history(history, save_path=history_path)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    print("=" * 60)

    # Load best model for evaluation
    best_model_path = os.path.join(config['paths']['models'], 'best_model.pt')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(
        model,
        dataloaders['test'],
        device=device,
        class_names=RadarDataset.CLASS_NAMES
    )

    # Plot confusion matrix
    cm_path = os.path.join(config['paths']['logs'], 'confusion_matrix.png')
    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        RadarDataset.CLASS_NAMES,
        save_path=cm_path
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Test accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"\nModel saved to: {best_model_path}")
    print(f"Training history: {history_path}")
    print(f"Confusion matrix: {cm_path}")

    # Export model for DQN teammate
    print("\n" + "-" * 60)
    print("FOR YOUR DQN TEAMMATE:")
    print("-" * 60)
    print(f"""
    from src.inference.predictor import ThreatClassifier

    # Load the trained model
    classifier = ThreatClassifier('{best_model_path}')

    # Make predictions
    class_name, confidence, probs = classifier.predict(spectrogram, doppler_seq, env_features)

    # Get threat level for RL reward
    threat_level = classifier.get_threat_level(class_name)
    """)


if __name__ == '__main__':
    main()
