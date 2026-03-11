"""
Training pipeline for LaneNet.

Features:
- Mixed precision training with torch.cuda.amp
- Learning rate scheduler (cosine annealing)
- TensorBoard logging
- Model checkpointing
- Early stopping
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import argparse

from model import create_lanenet
from dataset import create_dataloaders
from metrics import LaneMetrics, PerformanceProfiler


class Trainer:
    """Training manager for LaneNet."""

    def __init__(self, model: nn.Module, device: torch.device,
                 output_dir: str = './outputs', use_amp: bool = True):
        """
        Args:
            model: Model to train
            device: Device to train on
            output_dir: Output directory for checkpoints and logs
            use_amp: Use automatic mixed precision
        """
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.writer = SummaryWriter(self.output_dir / 'logs')

        # Mixed precision
        self.scaler = GradScaler() if use_amp else None

        # Best metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def setup_training(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """
        Setup optimizer and scheduler.

        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay (L2 regularization)
        """
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def setup_scheduler(self, total_epochs: int, batch_per_epoch: int):
        """
        Setup learning rate scheduler (cosine annealing).

        Args:
            total_epochs: Total number of epochs
            batch_per_epoch: Batches per epoch
        """
        total_steps = total_epochs * batch_per_epoch
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    seg_logits = outputs['seg']
                    loss = self.loss_fn(seg_logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                seg_logits = outputs['seg']
                loss = self.loss_fn(seg_logits, masks)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} LR: {lr:.2e}")

        avg_loss = total_loss / num_batches
        self.writer.add_scalar('train/loss', avg_loss, epoch)

        return {'loss': avg_loss}

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader
            epoch: Epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_f1 = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        seg_logits = outputs['seg']
                        loss = self.loss_fn(seg_logits, masks)
                else:
                    outputs = self.model(images)
                    seg_logits = outputs['seg']
                    loss = self.loss_fn(seg_logits, masks)

                total_loss += loss.item()

                # Compute metrics
                seg_pred = torch.sigmoid(seg_logits).cpu().numpy()
                masks_np = masks.cpu().numpy()

                f1 = LaneMetrics.f1_score(seg_pred[0, 0], masks_np[0, 0])
                acc = LaneMetrics.accuracy(seg_pred[0, 0], masks_np[0, 0])

                total_f1 += f1
                total_acc += acc
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_f1 = total_f1 / num_batches
        avg_acc = total_acc / num_batches

        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/f1_score', avg_f1, epoch)
        self.writer.add_scalar('val/accuracy', avg_acc, epoch)

        print(f"Validation - Loss: {avg_loss:.4f} F1: {avg_f1:.4f} Acc: {avg_acc:.4f}")

        return {
            'loss': avg_loss,
            'f1_score': avg_f1,
            'accuracy': avg_acc
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Epoch number
            metrics: Metrics dictionary
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Starting epoch
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, early_stopping_patience: int = 10):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        self.setup_scheduler(num_epochs, len(train_loader))

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_metrics = self.train_epoch(train_loader, epoch + 1)

            # Validate
            val_metrics = self.validate(val_loader, epoch + 1)

            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch + 1, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch + 1, val_metrics, is_best=False)

                if self.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                    break

        self.writer.close()
        print("\nTraining completed!")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train LaneNet')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--data-dir', type=str, default='/tmp/lane_data', help='Data directory')
    parser.add_argument('--use-mock-data', action='store_true', help='Use mock data')
    parser.add_argument('--backbone', type=str, default='efficientnet', help='Backbone')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    model = create_lanenet(backbone=args.backbone, pretrained=True)
    model = model.to(device)
    model.print_summary()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_mock_data=args.use_mock_data
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Setup trainer
    trainer = Trainer(model, device, output_dir=args.output_dir, use_amp=True)
    trainer.setup_training(learning_rate=args.lr, weight_decay=args.weight_decay)

    # Train
    trainer.train(train_loader, val_loader, num_epochs=args.epochs)


if __name__ == '__main__':
    main()
