import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .models.network_encoder import NetworkEncoder
from .losses import NetworkLoss
from .config import Config

class NetworkTrainer:
    """Trainer for network encoder model."""
    def __init__(
        self,
        model: NetworkEncoder,
        config: Config,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup loss
        self.criterion = NetworkLoss(
            contrastive_weight=config.training.contrastive_weight,
            triplet_weight=config.training.triplet_weight,
            network_weight=config.training.network_weight,
            temperature=config.training.temperature,
            margin=config.training.margin,
            distance=config.training.distance
        ).to(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        loss_components = {
            'contrastive': 0,
            'triplet': 0,
            'network': 0
        }
        
        # Training loop
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move batch to device
            features = {
                k: v.to(self.device)
                for k, v in batch['features'].items()
            }
            labels = batch['labels'].to(self.device)
            triplets = tuple(
                t.to(self.device) for t in batch.get('triplets', (None, None, None))
            ) if 'triplets' in batch else None
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(features)
            losses = self.criterion(embeddings, labels, triplets)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip
                )
                
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total'].item()
            for k, v in losses.items():
                if k != 'total':
                    loss_components[k] += v.item()
                    
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}"
            })
            
        # Compute average metrics
        num_batches = len(train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()}
        }
        
        return metrics
        
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        loss_components = {
            'contrastive': 0,
            'triplet': 0,
            'network': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                features = {
                    k: v.to(self.device)
                    for k, v in batch['features'].items()
                }
                labels = batch['labels'].to(self.device)
                triplets = tuple(
                    t.to(self.device) for t in batch.get('triplets', (None, None, None))
                ) if 'triplets' in batch else None
                
                # Forward pass
                embeddings = self.model(features)
                losses = self.criterion(embeddings, labels, triplets)
                
                # Update metrics
                total_loss += losses['total'].item()
                for k, v in losses.items():
                    if k != 'total':
                        loss_components[k] += v.item()
                        
        # Compute average metrics
        num_batches = len(val_loader)
        metrics = {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_components.items()}
        }
        
        return metrics
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Train model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_contrastive': [],
            'train_triplet': [],
            'train_network': [],
            'val_contrastive': [],
            'val_triplet': [],
            'val_network': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_contrastive'].append(train_metrics['contrastive'])
            history['train_triplet'].append(train_metrics['triplet'])
            history['train_network'].append(train_metrics['network'])
            
            # Log training metrics
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}"
            )
            
            # Validate if validation loader provided
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Update history
                history['val_loss'].append(val_metrics['loss'])
                history['val_contrastive'].append(val_metrics['contrastive'])
                history['val_triplet'].append(val_metrics['triplet'])
                history['val_network'].append(val_metrics['network'])
                
                # Log validation metrics
                self.logger.info(
                    f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}"
                )
                
                # Checkpoint if best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    if checkpoint_dir is not None:
                        self.save_checkpoint(
                            Path(checkpoint_dir) / 'best_model.pt',
                            epoch,
                            val_metrics
                        )
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break
                    
        return history
        
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
    def load_checkpoint(
        self,
        path: Path
    ) -> Tuple[int, Dict[str, float]]:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Tuple of (epoch, metrics)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics'] 