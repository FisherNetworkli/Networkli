import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Union, Path
from tqdm import tqdm
import numpy as np
import logging
import json

from .models import NetworkModel, GNNModel
from .losses import NetworkLoss
from .data import NetworkDataModule, FeatureTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkTrainer:
    """Trainer for network models."""
    def __init__(
        self,
        model: Union[NetworkModel, GNNModel],
        data_module: NetworkDataModule,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Network model
            data_module: Data module
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize loss function
        self.criterion = NetworkLoss().to(device)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.data_module.train_dataloader():
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                batch['features'],
                batch.get('adj_matrix')
            )
            
            # Compute loss
            loss_dict = self.criterion(
                outputs['embeddings'],
                batch['label'],
                batch.get('triplets'),
                batch.get('adj_matrix')
            )
            loss = loss_dict['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
        # Compute average metrics
        metrics = {
            'train_loss': total_loss / num_batches
        }
        
        return metrics
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model.
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.data_module.val_dataloader():
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch['features'],
                    batch.get('adj_matrix')
                )
                
                # Compute loss
                loss_dict = self.criterion(
                    outputs['embeddings'],
                    batch['label'],
                    batch.get('triplets'),
                    batch.get('adj_matrix')
                )
                loss = loss_dict['total']
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
        # Compute average metrics
        metrics = {
            'val_loss': total_loss / num_batches
        }
        
        return metrics
        
    def train(
        self,
        max_epochs: int,
        early_stopping_patience: int = 10,
        checkpoint_frequency: int = 5
    ):
        """Train model.
        
        Args:
            max_epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience
            checkpoint_frequency: Checkpoint frequency in epochs
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            logger.info(
                f'Epoch {epoch + 1}/{max_epochs} - ' +
                ' - '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
            )
            
            # Save checkpoint
            if self.checkpoint_dir and (epoch + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(epoch + 1, metrics)
                
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                # Save best model
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch + 1, metrics, is_best=True)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info('Early stopping triggered')
                break
                
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model
        """
        if not self.checkpoint_dir:
            return
            
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # Save metrics
        metrics_path = self.checkpoint_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ):
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch'], checkpoint['metrics'] 