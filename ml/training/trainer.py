from typing import Dict, List, Optional, Union, Callable
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from datetime import datetime
from pathlib import Path

class Trainer:
    """Training infrastructure for model training, validation, and checkpointing."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion: Callable,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint_dir: str = "checkpoints",
                 log_dir: str = "logs"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.early_stopping_counter = 0
        self.training_history = []
        
    def train_epoch(self,
                    train_loader: DataLoader,
                    scheduler: Optional[_LRScheduler] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            scheduler: Optional learning rate scheduler
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = self.criterion(outputs, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * len(batch['y'])
            total_samples += len(batch['y'])
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss
        }
    
    def validate(self,
                 val_loader: DataLoader,
                 metrics: Optional[List[Callable]] = None) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            metrics: Optional list of metric functions
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metric_values = {}
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch)
                
                # Update metrics
                total_loss += loss.item() * len(batch['y'])
                total_samples += len(batch['y'])
                
                # Calculate additional metrics
                if metrics:
                    for metric in metrics:
                        metric_name = metric.__name__
                        if metric_name not in metric_values:
                            metric_values[metric_name] = []
                        metric_values[metric_name].append(metric(outputs, batch))
        
        # Calculate average metrics
        results = {
            'loss': total_loss / total_samples
        }
        
        for metric_name, values in metric_values.items():
            results[metric_name] = np.mean(values)
        
        return results
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              scheduler: Optional[_LRScheduler] = None,
              metrics: Optional[List[Callable]] = None,
              early_stopping_patience: int = 5,
              checkpoint_frequency: int = 1) -> Dict[str, List[float]]:
        """
        Train the model with validation and checkpointing.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
            metrics: Optional list of metric functions
            early_stopping_patience: Number of epochs to wait before early stopping
            checkpoint_frequency: Frequency of checkpointing in epochs
            
        Returns:
            Dictionary of training history
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, scheduler)
            
            # Validate
            val_metrics = self.validate(val_loader, metrics)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Checkpoint if needed
            if (epoch + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['loss'], early_stopping_patience):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        return self.training_history
    
    def _log_metrics(self,
                     epoch: int,
                     train_metrics: Dict[str, float],
                     val_metrics: Dict[str, float]) -> None:
        """Log training and validation metrics."""
        metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to history
        self.training_history.append(metrics)
        
        # Save to file
        log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print metrics
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        for metric_name, value in val_metrics.items():
            if metric_name != 'loss':
                print(f"  Val {metric_name}: {value:.4f}")
    
    def _save_checkpoint(self,
                        epoch: int,
                        metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        # Save checkpoint
        checkpoint_file = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_file)
        
        # Save best model if needed
        if metrics['loss'] < self.best_metric:
            self.best_metric = metrics['loss']
            best_model_file = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_model_file)
    
    def _check_early_stopping(self,
                             val_loss: float,
                             patience: int) -> bool:
        """Check if training should be stopped early."""
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            self.early_stopping_counter = 0
            return False
        
        self.early_stopping_counter += 1
        return self.early_stopping_counter >= patience
    
    def load_checkpoint(self,
                       checkpoint_path: Union[str, Path],
                       load_optimizer: bool = True) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if requested
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['metrics']['loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
        print(f"Best validation loss: {self.best_metric:.4f}") 