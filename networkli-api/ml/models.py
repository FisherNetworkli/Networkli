import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import Dict, Tuple, List, Optional
from .base_model import BaseModel

class NetworkliGNN(BaseModel):
    """
    Implementation of the Networkli recommendation algorithm.
    Combines graph neural networks with meta-learning for professional networking.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 embedding_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Multi-scale graph convolution layers
        self.conv_layers = nn.ModuleList([
            GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Attention-based aggregation
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout)
        
        # Professional domain expertise embedding
        self.domain_embedding = nn.Linear(hidden_dim * 4, embedding_dim)
        
        # Skill compatibility scoring
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode nodes using multi-scale graph convolutions and attention.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        # Multi-scale feature extraction
        hidden_states = []
        h = x
        
        for conv in self.conv_layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
            hidden_states.append(h)
        
        # Attention-based feature aggregation
        h = self.attention(h, edge_index)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([
            F.adaptive_avg_pool1d(hs.unsqueeze(2), 1).squeeze(2)
            for hs in hidden_states
        ], dim=1)
        
        # Project to final embedding space
        embeddings = self.domain_embedding(multi_scale)
        
        return embeddings
    
    def compute_similarity(self, 
                         embeddings: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute professional similarity scores between connected nodes.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Similarity scores for each edge [num_edges]
        """
        # Get embeddings for both ends of each edge
        src, dst = edge_index
        src_embeds = embeddings[src]
        dst_embeds = embeddings[dst]
        
        # Concatenate embeddings and compute compatibility score
        edge_features = torch.cat([src_embeds, dst_embeds], dim=1)
        scores = self.compatibility_scorer(edge_features).squeeze(-1)
        
        return scores
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_label_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices for message passing [2, num_edges]
            edge_label_index: Optional edges to compute scores for [2, num_edges]
            
        Returns:
            Edge scores for the provided edge_label_index
        """
        # Get node embeddings
        embeddings = self.encode_nodes(x, edge_index)
        
        # If no specific edges provided, use edge_index
        if edge_label_index is None:
            edge_label_index = edge_index
            
        # Compute similarity scores
        scores = self.compute_similarity(embeddings, edge_label_index)
        
        return scores
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing:
                - x: Node features
                - edge_index: Graph connectivity
                - pos_edge_index: Positive edges for training
                - neg_edge_index: Negative edges for training
                
        Returns:
            Dictionary with loss and metrics
        """
        # Get node embeddings
        embeddings = self.encode_nodes(batch['x'], batch['edge_index'])
        
        # Compute scores for positive and negative edges
        pos_scores = self.compute_similarity(embeddings, batch['pos_edge_index'])
        neg_scores = self.compute_similarity(embeddings, batch['neg_edge_index'])
        
        # Combine positive and negative scores and create labels
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        # Compute metrics
        with torch.no_grad():
            preds = (scores > 0).float()
            accuracy = (preds == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.
        
        Args:
            batch: Dictionary containing validation data
            
        Returns:
            Dictionary with validation metrics
        """
        with torch.no_grad():
            metrics = self.training_step(batch)
            metrics = {f'val_{k}': v for k, v in metrics.items()}
        return metrics
    
    def predict_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate predictions for new edges.
        
        Args:
            batch: Dictionary containing:
                - x: Node features
                - edge_index: Graph connectivity
                - pred_edge_index: Edges to predict
                
        Returns:
            Probability scores for the predicted edges
        """
        with torch.no_grad():
            scores = self(
                batch['x'],
                batch['edge_index'],
                batch['pred_edge_index']
            )
            probs = torch.sigmoid(scores)
        return probs 