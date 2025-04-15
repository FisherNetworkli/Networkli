import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging
import json
from tqdm import tqdm

from ..models.network_model import NetworkModel, NetworkLoss
from ..utils.data_loader import DataLoader as DataLoaderUtil

class NetworkTrainer:
    def __init__(
        self,
        model: NetworkModel,
        data_loader: DataLoaderUtil,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = NetworkLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        loss_components = {
            'contrastive_loss': 0,
            'triplet_loss': 0,
            'total_loss': 0
        }
        
        for batch in tqdm(train_loader, desc='Training'):
            # Move batch to device
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            
            # Forward pass
            anchor_emb = self.model(
                batch['anchor_profile'].to(self.device),
                batch['anchor_activity'].to(self.device),
                batch['anchor_network'].to(self.device),
                batch['anchor_interaction'].to(self.device)
            )
            
            positive_emb = self.model(
                batch['positive_profile'].to(self.device),
                batch['positive_activity'].to(self.device),
                batch['positive_network'].to(self.device),
                batch['positive_interaction'].to(self.device)
            )
            
            negative_emb = self.model(
                batch['negative_profile'].to(self.device),
                batch['negative_activity'].to(self.device),
                batch['negative_network'].to(self.device),
                batch['negative_interaction'].to(self.device)
            )
            
            # Compute loss
            loss, batch_components = self.criterion(
                anchor_emb,
                positive_emb,
                negative_emb
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += batch_components[key]
        
        # Average metrics
        num_batches = len(train_loader)
        for key in loss_components:
            loss_components[key] /= num_batches
            
        return loss_components
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        loss_components = {
            'contrastive_loss': 0,
            'triplet_loss': 0,
            'total_loss': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move batch to device
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                # Forward pass
                anchor_emb = self.model(
                    batch['anchor_profile'].to(self.device),
                    batch['anchor_activity'].to(self.device),
                    batch['anchor_network'].to(self.device),
                    batch['anchor_interaction'].to(self.device)
                )
                
                positive_emb = self.model(
                    batch['positive_profile'].to(self.device),
                    batch['positive_activity'].to(self.device),
                    batch['positive_network'].to(self.device),
                    batch['positive_interaction'].to(self.device)
                )
                
                negative_emb = self.model(
                    batch['negative_profile'].to(self.device),
                    batch['negative_activity'].to(self.device),
                    batch['negative_network'].to(self.device),
                    batch['negative_interaction'].to(self.device)
                )
                
                # Compute loss
                loss, batch_components = self.criterion(
                    anchor_emb,
                    positive_emb,
                    negative_emb
                )
                
                # Update metrics
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += batch_components[key]
        
        # Average metrics
        num_batches = len(val_loader)
        for key in loss_components:
            loss_components[key] /= num_batches
            
        return loss_components
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logging.info(f'Saved checkpoint to {path}')
        
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f'Loaded checkpoint from {path}')
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            logging.info(f'Epoch {epoch + 1}/{self.num_epochs}:')
            logging.info(f'Train Loss: {train_metrics["total_loss"]:.4f}')
            logging.info(f'Val Loss: {val_metrics["total_loss"]:.4f}')
            
            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['epochs'].append(epoch + 1)
            
            # Save checkpoint if validation loss improved
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(epoch + 1, val_metrics)
            
            # Save training history
            with open(self.log_dir / 'training_history.json', 'w') as f:
                json.dump(self.history, f) 