import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm

from .models import NetworkliGNN
from .data_utils import NetworkliDataset, process_user_data, generate_negative_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkliTrainer:
    """Trainer class for the Networkli recommendation model."""
    
    def __init__(self,
                 model: NetworkliGNN,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The NetworkliGNN model instance
            learning_rate: Learning rate for optimization
            weight_decay: L2 regularization factor
            device: Device to run the training on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(
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
        
    def train_epoch(self, 
                    dataset: NetworkliDataset,
                    batch_size: int = 32768) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataset: The training dataset
            batch_size: Number of edges to process at once
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        # Get the full graph
        data = dataset[0].to(self.device)
        
        # Generate negative samples for training
        neg_edge_index = generate_negative_samples(
            data.edge_index,
            num_nodes=data.x.size(0),
            num_negative=data.edge_index.size(1)
        ).to(self.device)
        
        # Process edges in batches
        num_edges = data.edge_index.size(1)
        perm = torch.randperm(num_edges)
        
        for start in range(0, num_edges, batch_size):
            self.optimizer.zero_grad()
            
            # Get batch indices
            batch_idx = perm[start:start + batch_size]
            
            # Prepare batch
            batch = {
                'x': data.x,
                'edge_index': data.edge_index,
                'pos_edge_index': data.edge_index[:, batch_idx],
                'neg_edge_index': neg_edge_index[:, batch_idx]
            }
            
            # Forward pass
            metrics = self.model.training_step(batch)
            loss = metrics['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += metrics['accuracy'].item()
            num_batches += 1
            
        return {
            'train_loss': total_loss / num_batches,
            'train_accuracy': total_accuracy / num_batches
        }
    
    @torch.no_grad()
    def validate(self,
                dataset: NetworkliDataset,
                batch_size: int = 32768) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataset: The validation dataset
            batch_size: Number of edges to process at once
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        # Get the full graph
        data = dataset[0].to(self.device)
        
        # Generate negative samples for validation
        neg_edge_index = generate_negative_samples(
            data.edge_index,
            num_nodes=data.x.size(0),
            num_negative=data.edge_index.size(1)
        ).to(self.device)
        
        # Process edges in batches
        num_edges = data.edge_index.size(1)
        
        for start in range(0, num_edges, batch_size):
            # Prepare batch
            batch = {
                'x': data.x,
                'edge_index': data.edge_index,
                'pos_edge_index': data.edge_index[:, start:start + batch_size],
                'neg_edge_index': neg_edge_index[:, start:start + batch_size]
            }
            
            # Forward pass
            metrics = self.model.validation_step(batch)
            
            # Update metrics
            total_loss += metrics['val_loss'].item()
            total_accuracy += metrics['val_accuracy'].item()
            num_batches += 1
            
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def train(self,
             train_dataset: NetworkliDataset,
             val_dataset: Optional[NetworkliDataset] = None,
             num_epochs: int = 100,
             batch_size: int = 32768,
             patience: int = 10,
             save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            num_epochs: Number of epochs to train for
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            save_dir: Optional directory to save model checkpoints
            
        Returns:
            Dictionary containing training history and best metrics
        """
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve = 0
        history = []
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_dataset, batch_size)
            
            # Validate
            if val_dataset is not None:
                val_metrics = self.validate(val_dataset, batch_size)
                metrics = {**train_metrics, **val_metrics}
                
                # Update learning rate
                self.scheduler.step(val_metrics['val_loss'])
                
                # Check for improvement
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_epoch = epoch
                    no_improve = 0
                    
                    # Save best model
                    if save_dir is not None:
                        save_path = Path(save_dir) / 'best_model.pt'
                        self.model.save(
                            metadata={
                                'epoch': epoch,
                                'metrics': metrics
                            }
                        )
                else:
                    no_improve += 1
            else:
                metrics = train_metrics
            
            # Log progress
            logger.info(f'Epoch {epoch}: {metrics}')
            history.append(metrics)
            
            # Early stopping
            if no_improve >= patience:
                logger.info(f'Early stopping after {epoch} epochs')
                break
                
        return {
            'history': history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
    
    @torch.no_grad()
    def predict(self,
               dataset: NetworkliDataset,
               candidate_edges: torch.Tensor,
               batch_size: int = 32768) -> torch.Tensor:
        """
        Generate predictions for candidate edges.
        
        Args:
            dataset: Dataset containing the graph
            candidate_edges: Edges to predict [2, num_edges]
            batch_size: Batch size for prediction
            
        Returns:
            Probability scores for each candidate edge
        """
        self.model.eval()
        
        # Get the full graph
        data = dataset[0].to(self.device)
        candidate_edges = candidate_edges.to(self.device)
        
        all_scores = []
        
        # Process edges in batches
        num_edges = candidate_edges.size(1)
        
        for start in range(0, num_edges, batch_size):
            end = min(start + batch_size, num_edges)
            
            # Prepare batch
            batch = {
                'x': data.x,
                'edge_index': data.edge_index,
                'pred_edge_index': candidate_edges[:, start:end]
            }
            
            # Get predictions
            scores = self.model.predict_step(batch)
            all_scores.append(scores)
            
        return torch.cat(all_scores) 