from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
from .base_model import BaseModel

class NetworkliGNN(BaseModel):
    """Enhanced Graph Neural Network for user matching and recommendations."""
    
    def __init__(self, 
                 input_dim: int = 768,  # BERT embedding dimension
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__(model_name="networkli_gnn", version="1.0.0")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-scale graph convolutions
        self.convs = nn.ModuleList([
            GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Attention layers
        self.attention = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Professional domain embedding
        self.domain_embedding = nn.Linear(hidden_dim, hidden_dim)
        
        # Skill compatibility scoring
        self.skill_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Output layers
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Tuple of (node embeddings, compatibility scores)
        """
        # Initial node features
        h = x
        
        # Multi-scale graph convolutions with attention
        for conv, attn in zip(self.convs, self.attention):
            # Graph convolution
            h_conv = conv(h, edge_index)
            h_conv = F.relu(h_conv)
            h_conv = F.dropout(h_conv, p=self.dropout, training=self.training)
            
            # Attention mechanism
            h_attn = attn(h_conv, edge_index)
            h_attn = F.relu(h_attn)
            h_attn = F.dropout(h_attn, p=self.dropout, training=self.training)
            
            # Combine features
            h = h_conv + h_attn
        
        # Professional domain embedding
        h_domain = self.domain_embedding(h)
        
        # Calculate skill compatibility scores
        if edge_attr is not None:
            # For each edge, concatenate source and target node features
            src, dst = edge_index
            edge_features = torch.cat([h[src], h[dst]], dim=1)
            compatibility = self.skill_scorer(edge_features)
        else:
            compatibility = None
        
        return h, compatibility
    
    def train(self, data: Dict[str, torch.Tensor], **kwargs) -> None:
        """
        Train the model on graph data.
        
        Args:
            data: Dictionary containing:
                - x: Node features
                - edge_index: Graph connectivity
                - edge_attr: Edge features
                - y: Node labels
                - batch: Batch assignment
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=kwargs.get('lr', 0.001))
        
        # Convert data to PyTorch Geometric format
        graph_data = Data(
            x=data['x'],
            edge_index=data['edge_index'],
            edge_attr=data.get('edge_attr'),
            y=data['y'],
            batch=data.get('batch')
        )
        
        # Training loop
        for epoch in range(kwargs.get('epochs', 100)):
            optimizer.zero_grad()
            
            # Forward pass
            out, compatibility = self(graph_data.x, 
                                    graph_data.edge_index,
                                    graph_data.edge_attr,
                                    graph_data.batch)
            
            # Calculate losses
            node_loss = F.binary_cross_entropy_with_logits(out, graph_data.y)
            
            if compatibility is not None and 'edge_labels' in data:
                edge_loss = F.binary_cross_entropy(compatibility, data['edge_labels'])
                loss = node_loss + edge_loss
            else:
                loss = node_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Early stopping check
            if kwargs.get('early_stopping'):
                if self._check_early_stopping(loss.item()):
                    break
    
    def predict(self, 
                data: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Make predictions on graph data.
        
        Args:
            data: Dictionary containing graph data
            
        Returns:
            Dictionary containing predictions
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            out, compatibility = self(data['x'],
                                    data['edge_index'],
                                    data.get('edge_attr'),
                                    data.get('batch'))
            
            # Convert to probabilities
            node_probs = torch.sigmoid(out)
            
            return {
                'node_predictions': node_probs,
                'compatibility_scores': compatibility
            }
    
    def evaluate(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data: Dictionary containing test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.eval()
        with torch.no_grad():
            # Get predictions
            preds = self.predict(data)
            
            # Calculate metrics
            node_preds = (preds['node_predictions'] > 0.5).float()
            node_acc = (node_preds == data['y']).float().mean().item()
            
            metrics = {
                'node_accuracy': node_acc
            }
            
            if 'edge_labels' in data and preds['compatibility_scores'] is not None:
                edge_preds = (preds['compatibility_scores'] > 0.5).float()
                edge_acc = (edge_preds == data['edge_labels']).float().mean().item()
                metrics['edge_accuracy'] = edge_acc
            
            return metrics
    
    def _check_early_stopping(self, loss: float) -> bool:
        """Check if training should be stopped early."""
        if not hasattr(self, '_best_loss'):
            self._best_loss = float('inf')
            self._patience = 5
            self._counter = 0
            return False
        
        if loss < self._best_loss:
            self._best_loss = loss
            self._counter = 0
            return False
        
        self._counter += 1
        return self._counter >= self._patience 