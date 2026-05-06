"""
Training Pipeline for Radar Classifier

Handles the complete training loop with:
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- Logging and metrics tracking
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm

from .metrics import compute_metrics, AverageMeter


class Trainer:
    """Training class for radar classifier."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.lr = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 0.0001)

        # Early stopping config
        es_config = train_config.get('early_stopping', {})
        self.patience = es_config.get('patience', 15)
        self.min_delta = es_config.get('min_delta', 0.001)

        # Setup loss function with class weights
        class_weights = train_config.get('class_weights')
        if class_weights == 'balanced':
            # Get weights from dataset if available
            if hasattr(train_loader.dataset, 'get_class_weights'):
                weights = train_loader.dataset.get_class_weights().to(self.device)
            else:
                weights = None
        elif isinstance(class_weights, list):
            weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            weights = None

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Setup scheduler
        scheduler_config = train_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)

        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None

        # Warmup scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = None
        if warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )

        # Output directory
        self.output_dir = config.get('paths', {}).get('models', 'outputs/models')
        os.makedirs(self.output_dir, exist_ok=True)

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        loss_meter = AverageMeter()
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')

        for batch in pbar:
            spectrogram, doppler_seq, env_features, labels = batch
            spectrogram = spectrogram.to(self.device)
            doppler_seq = doppler_seq.to(self.device)
            env_features = env_features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectrogram, doppler_seq, env_features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            loss_meter.update(loss.item(), spectrogram.size(0))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        accuracy = 100. * correct / total
        return loss_meter.avg, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model.

        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()

        loss_meter = AverageMeter()
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc='Validating'):
            spectrogram, doppler_seq, env_features, labels = batch
            spectrogram = spectrogram.to(self.device)
            doppler_seq = doppler_seq.to(self.device)
            env_features = env_features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(spectrogram, doppler_seq, env_features)
            loss = self.criterion(outputs, labels)

            loss_meter.update(loss.item(), spectrogram.size(0))

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = compute_metrics(all_labels, all_preds)
        accuracy = metrics['accuracy'] * 100

        return loss_meter.avg, accuracy, metrics

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Training history dictionary
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 50)

        start_time = time.time()

        for epoch in range(self.epochs):
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_metrics = self.validate()

            # Learning rate scheduling
            if epoch < self.warmup_epochs:
                if self.warmup_scheduler:
                    self.warmup_scheduler.step()
            else:
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt', val_metrics)
                print("  -> Saved best model!")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Save final model
        self.save_checkpoint('final_model.pt')

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        return self.history

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            metrics: Optional metrics to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': len(self.history['train_loss']),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }

        if metrics:
            checkpoint['metrics'] = metrics

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

        print(f"Loaded checkpoint from {path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to use

    Returns:
        Tuple of (trained_model, history)
    """
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()

    return trainer.model, history
