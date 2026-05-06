#!/usr/bin/env python3
"""
DRDO-Level Production Training Script

Training pipeline for radar threat classification with:
- Support for real datasets only
- MLflow experiment tracking
- Temperature scaling calibration
- Comprehensive logging and monitoring
- Model versioning and registry

Usage:
    python scripts/train_production.py --data real --epochs 100
    python scripts/train_production.py --config configs/config_production.yaml
"""

import argparse
import os
import sys
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import RadarDataset, create_dataloaders
from src.models.cnn_lstm import RadarClassifier, build_model, count_parameters
from src.training.trainer import Trainer
from src.training.metrics import (
    plot_training_history, plot_confusion_matrix, evaluate_model, compute_metrics
)
from src.utils.logging_utils import setup_logging, PerformanceMonitor
from src.utils.experiment_tracking import ExperimentTracker, ModelRegistry
from src.utils.uncertainty import TemperatureScaling, compute_expected_calibration_error

# Try to import real dataset loaders
try:
    from src.data.real_datasets import MAFATLoader, BistaticUAVLoader, print_dataset_instructions
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load and validate configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required = ['training', 'model', 'paths']
    for section in required:
        if section not in config:
            print(f"Warning: Missing config section '{section}'")

    return config


def check_real_data(config: dict) -> dict:
    """Check for available real datasets and return paths."""
    data_status = {
        'mafat': False,
        'bistatic_uav': False,
        'dronerf': False
    }

    real_data_dir = Path(config.get('paths', {}).get('data_real', 'data/real'))

    # Check MAFAT
    mafat_dir = real_data_dir / 'mafat'
    mafat_required = [
        mafat_dir / 'train' / 'MAFAT_RADAR_Train_Segments.pkl',
        mafat_dir / 'train' / 'MAFAT_RADAR_Train_Metadata.csv'
    ]
    if all(path.exists() for path in mafat_required):
        data_status['mafat'] = True
        print(f"[OK] MAFAT dataset found at {mafat_dir}")
    elif mafat_dir.exists():
        missing = [path.name for path in mafat_required if not path.exists()]
        print(f"[!] MAFAT directory found but missing required files: {missing}")

    # Check Bistatic UAV
    bistatic_dir = real_data_dir / 'bistatic_uav'
    bistatic_images = bistatic_dir / 'trainval' / 'images'
    if bistatic_images.exists() and list(bistatic_images.glob('*.png')):
        data_status['bistatic_uav'] = True
        print(f"[OK] Bistatic UAV dataset found at {bistatic_dir}")

    # Check DroneRF
    dronerf_dir = real_data_dir / 'dronerf'
    if dronerf_dir.exists() and any(dronerf_dir.rglob('*')):
        data_status['dronerf'] = True
        print(f"[OK] DroneRF dataset found at {dronerf_dir}")

    if not any([data_status['mafat'], data_status['bistatic_uav'], data_status['dronerf']]):
        print("\n[!] No real datasets found. Run with --help-data for download instructions.")

    return data_status


def enforce_real_only_policy(args: argparse.Namespace, config: dict, data_status: dict):
    """Enforce real-data-only execution policy and required dataset availability."""
    policy = config.get('data_policy', {})
    real_only = policy.get('real_only', True)

    if real_only and args.data != 'real':
        raise ValueError("This project is configured for real datasets only. Use --data real.")

    required = policy.get('required_real_datasets', ['mafat'])
    if isinstance(required, str):
        required = [required]

    missing = [name for name in required if not data_status.get(name, False)]
    if missing:
        raise FileNotFoundError(
            "Missing required real datasets: "
            f"{missing}. Place datasets under {config.get('paths', {}).get('data_real', 'data/real')} "
            "or run --help-data for download instructions."
        )


def train_with_tracking(
    config: dict,
    tracker: ExperimentTracker,
    device: str,
    dataloaders: dict
) -> tuple:
    """
    Train model with experiment tracking.

    Returns:
        Tuple of (model, history, best_val_acc)
    """
    # Build model
    print("\n" + "=" * 60)
    print("Building Model")
    print("=" * 60)

    model = build_model(config)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Log model architecture
    tracker.log_param('model.parameters', n_params)
    tracker.log_param('model.architecture', config.get('model', {}).get('architecture', 'cnn_lstm'))

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config,
        device=device
    )

    # Training loop with metric logging
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    patience = config['training'].get('early_stopping_patience', 10)
    patience_counter = 0

    epochs = config['training']['epochs']

    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc, val_metrics = trainer.validate()

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(trainer.optimizer.param_groups[0]['lr'])

        # Log metrics to tracker
        tracker.log_metrics({
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'learning_rate': trainer.optimizer.param_groups[0]['lr']
        }, step=epoch)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }

            best_model_path = os.path.join(config['paths']['models'], 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f"  -> Saved best model (val_acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Update scheduler (skip if warmup or invalid T_max)
        if trainer.scheduler is not None and epoch >= trainer.warmup_epochs:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(val_loss)
            else:
                try:
                    trainer.scheduler.step()
                except ZeroDivisionError:
                    pass  # T_max is 0, skip scheduling

    return model, history, best_val_acc


def calibrate_and_evaluate(
    model: torch.nn.Module,
    config: dict,
    dataloaders: dict,
    tracker: ExperimentTracker,
    device: str
) -> dict:
    """
    Calibrate model and perform comprehensive evaluation.
    """
    print("\n" + "=" * 60)
    print("Model Calibration & Evaluation")
    print("=" * 60)

    # Load best model
    best_model_path = os.path.join(config['paths']['models'], 'best_model.pt')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Temperature scaling calibration
    print("\nCalibrating with Temperature Scaling...")
    calibrated_model = TemperatureScaling(model)
    temperature = calibrated_model.calibrate(dataloaders['val'], device=device)
    tracker.log_param('calibration.temperature', temperature)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        calibrated_model,
        dataloaders['test'],
        device=device,
        class_names=RadarDataset.CLASS_NAMES
    )

    # Compute calibration error
    ece, ece_details = compute_expected_calibration_error(
        test_metrics['probabilities'],
        test_metrics['labels']
    )
    test_metrics['ece'] = ece
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Log final metrics
    tracker.log_metrics({
        'test/accuracy': test_metrics['accuracy'],
        'test/f1_macro': test_metrics['f1_macro'],
        'test/f1_weighted': test_metrics['f1_weighted'],
        'test/precision_macro': test_metrics['precision_macro'],
        'test/recall_macro': test_metrics['recall_macro'],
        'test/ece': ece
    })

    # Log per-class metrics
    for cls_name, f1 in test_metrics['f1_per_class'].items():
        tracker.log_metric(f'test/f1_{cls_name}', f1)

    # Save confusion matrix
    cm_path = os.path.join(config['paths']['logs'], 'confusion_matrix.png')
    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        RadarDataset.CLASS_NAMES,
        save_path=cm_path
    )
    tracker.log_artifact(cm_path, 'plots')

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='DRDO-Level Radar Threat Classifier Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train with real data (if available):
    python scripts/train_production.py --data real

  Custom configuration:
    python scripts/train_production.py --config configs/config_production.yaml --epochs 100

  Show dataset download instructions:
    python scripts/train_production.py --help-data
        """
    )

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
    parser.add_argument('--data', type=str, choices=['real'],
                        default='real', help='Data source (real-only policy)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name')
    parser.add_argument('--help-data', action='store_true',
                        help='Show dataset download instructions')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable MLflow tracking')

    args = parser.parse_args()

    # Show data instructions and exit
    if args.help_data:
        if REAL_DATA_AVAILABLE:
            print_dataset_instructions('all')
        else:
            print("Real dataset loaders not available.")
            print("See DATASETS.md for manual download instructions.")
        return

    # Setup logging
    logger = setup_logging(
        log_dir='outputs/logs',
        log_name=datetime.now().strftime('train_%Y%m%d_%H%M%S')
    )

    print("=" * 60)
    print("DRDO-Level Radar Threat Classification Training")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Configuration: {args.config}")
    print(f"Data source: {args.data}")

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Determine device
    if args.device:
        device = args.device
    elif config.get('hardware', {}).get('device') == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.get('hardware', {}).get('device', 'cpu')

    print(f"Device: {device}")

    # Create output directories
    os.makedirs(config['paths']['models'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)

    # Check data availability
    data_status = check_real_data(config)
    enforce_real_only_policy(args, config, data_status)

    # Setup experiment tracking
    if not args.no_tracking:
        experiment_name = args.experiment_name or 'radar_threat_classification'
        tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri='outputs/mlruns'
        )

        run_name = args.run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tracker.start_run(
            run_name=run_name,
            tags={
                'data_source': args.data,
                'device': device,
                'timestamp': datetime.now().isoformat()
            }
        )
        tracker.log_config(config)
    else:
        tracker = None
        print("Tracking disabled")

    try:
        # Prepare data
        data_dir = config.get('paths', {}).get('data_real', 'data/real')
        print(f"Using real dataset root: {data_dir}")

        # Create data loaders
        print("\nCreating data loaders...")
        dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=config['training']['batch_size'],
            num_workers=config.get('hardware', {}).get('num_workers', 0),
            config=config,
            augment_train=config.get('augmentation', {}).get('enabled', True)
        )

        print(f"Train samples: {len(dataloaders['train'].dataset)}")
        print(f"Val samples: {len(dataloaders['val'].dataset)}")
        print(f"Test samples: {len(dataloaders['test'].dataset)}")

        if tracker:
            tracker.log_params({
                'data.train_samples': len(dataloaders['train'].dataset),
                'data.val_samples': len(dataloaders['val'].dataset),
                'data.test_samples': len(dataloaders['test'].dataset)
            })

        # Train model
        model, history, best_val_acc = train_with_tracking(
            config=config,
            tracker=tracker if tracker else DummyTracker(),
            device=device,
            dataloaders=dataloaders
        )

        # Save training history
        history_path = os.path.join(config['paths']['logs'], 'training_history.png')
        plot_training_history(history, save_path=history_path)
        if tracker:
            tracker.log_artifact(history_path, 'plots')

        # Calibrate and evaluate
        test_metrics = calibrate_and_evaluate(
            model=model,
            config=config,
            dataloaders=dataloaders,
            tracker=tracker if tracker else DummyTracker(),
            device=device
        )

        # Register model
        registry = ModelRegistry()
        version = registry.register_model(
            model_name='radar_threat_classifier',
            model_path=os.path.join(config['paths']['models'], 'best_model.pt'),
            metrics={
                'accuracy': test_metrics['accuracy'],
                'f1_macro': test_metrics['f1_macro'],
                'ece': test_metrics['ece']
            },
            config=config,
            tags={'data_source': args.data}
        )

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Test accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"Calibration ECE: {test_metrics['ece']:.4f}")
        print(f"\nModel registered as: radar_threat_classifier {version}")
        print(f"Model saved to: {os.path.join(config['paths']['models'], 'best_model.pt')}")

        if tracker:
            tracker.end_run(status='FINISHED')

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        if tracker:
            tracker.end_run(status='FAILED')
        raise


class DummyTracker:
    """Dummy tracker when MLflow is disabled."""
    def log_params(self, *args, **kwargs): pass
    def log_param(self, *args, **kwargs): pass
    def log_metrics(self, *args, **kwargs): pass
    def log_metric(self, *args, **kwargs): pass
    def log_artifact(self, *args, **kwargs): pass
    def log_config(self, *args, **kwargs): pass


if __name__ == '__main__':
    main()
