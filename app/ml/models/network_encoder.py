import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class FeatureEncoder(nn.Module):
    """Encoder for a single feature type."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_rate),
                nn.ReLU() if activation == 'relu' else nn.Identity()
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

class AttentionFusion(nn.Module):
    """Attention-based feature fusion."""
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # Feature projections
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in feature_dims.items()
        })
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass."""
        # Project features
        projected = {
            name: proj(feat)
            for name, (feat, proj) in zip(
                features.keys(),
                self.projections.items()
            )
        }
        
        # Stack features
        stacked = torch.stack(list(projected.values()), dim=1)
        
        # Apply attention
        attended, _ = self.attention(
            stacked,
            stacked,
            stacked
        )
        
        # Residual connection and normalization
        fused = self.layer_norm(stacked + attended)
        
        # Output projection
        output = self.output_proj(fused)
        
        return output.mean(dim=1)  # Pool across features

class NetworkEncoder(nn.Module):
    """Network encoder model."""
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dims: List[int],
        embedding_dim: int,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_layer_norm: bool = True,
        fusion_method: str = 'concat'
    ):
        super().__init__()
        
        # Feature encoders
        self.encoders = nn.ModuleDict({
            name: FeatureEncoder(
                input_dim=dim,
                hidden_dims=hidden_dims,
                output_dim=embedding_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                use_layer_norm=use_layer_norm
            )
            for name, dim in feature_dims.items()
        })
        
        # Feature fusion
        if fusion_method == 'attention':
            self.fusion = AttentionFusion(
                feature_dims={name: embedding_dim for name in feature_dims},
                hidden_dim=embedding_dim,
                dropout_rate=dropout_rate
            )
        else:
            self.fusion = None
            
        # Output projection
        if fusion_method == 'concat':
            self.output_proj = nn.Linear(
                embedding_dim * len(feature_dims),
                embedding_dim
            )
        else:
            self.output_proj = nn.Linear(embedding_dim, embedding_dim)
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def encode_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Encode individual features."""
        return {
            name: encoder(feat)
            for name, (feat, encoder) in zip(
                features.keys(),
                self.encoders.items()
            )
        }
        
    def fuse_features(
        self,
        encoded_features: Dict[str, torch.Tensor],
        fusion_method: str = 'concat'
    ) -> torch.Tensor:
        """Fuse encoded features."""
        if fusion_method == 'attention':
            return self.fusion(encoded_features)
            
        elif fusion_method == 'concat':
            # Concatenate features
            concatenated = torch.cat(
                list(encoded_features.values()),
                dim=-1
            )
            # Project to embedding dimension
            return self.output_proj(concatenated)
            
        elif fusion_method == 'sum':
            # Sum features
            summed = sum(encoded_features.values())
            # Project to embedding dimension
            return self.output_proj(summed)
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        fusion_method: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # Encode features
        encoded = self.encode_features(features)
        
        # Fuse features
        if fusion_method is None:
            fusion_method = self.fusion_method
            
        fused = self.fuse_features(encoded, fusion_method)
        
        # Normalize output
        output = self.layer_norm(fused)
        
        return output
        
    def get_embeddings(
        self,
        features: Dict[str, torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """Get normalized embeddings."""
        embeddings = self.forward(features)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings 