import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class NetworkModel(nn.Module):
    """Network model for learning node embeddings."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """Initialize network model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # Feature encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.feature_encoder = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(prev_dim, output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features [num_nodes, input_dim]
            adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
            edge_index: Optional edge index [2, num_edges]
            
        Returns:
            Dictionary containing:
                - embeddings: Node embeddings [num_nodes, output_dim]
                - features: Encoded features [num_nodes, hidden_dim]
        """
        # Encode features
        features = self.feature_encoder(x)
        
        # Project to output dimension
        embeddings = self.output_proj(features)
        
        return {
            'embeddings': embeddings,
            'features': features
        }
        
class GNNLayer(nn.Module):
    """Graph neural network layer."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """Initialize GNN layer.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [num_nodes, in_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated features [num_nodes, out_dim]
        """
        # Graph convolution
        out = torch.matmul(adj_matrix, x)
        out = self.linear(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out
        
class GNNModel(nn.Module):
    """Graph neural network model."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """Initialize GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # GNN layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(GNNLayer(
                in_dim=prev_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            ))
            prev_dim = hidden_dim
            
        self.gnn_layers = nn.ModuleList(layers)
        
        # Output projection
        self.output_proj = nn.Linear(prev_dim, output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features [num_nodes, input_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            edge_index: Optional edge index [2, num_edges]
            
        Returns:
            Dictionary containing:
                - embeddings: Node embeddings [num_nodes, output_dim]
                - features: Encoded features [num_nodes, hidden_dim]
        """
        # Apply GNN layers
        features = x
        for layer in self.gnn_layers:
            features = layer(features, adj_matrix)
            
        # Project to output dimension
        embeddings = self.output_proj(features)
        
        return {
            'embeddings': embeddings,
            'features': features
        } 