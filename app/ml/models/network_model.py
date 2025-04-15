import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class NetworkModel(nn.Module):
    def __init__(
        self,
        profile_input_dim: int,
        activity_input_dim: int,
        network_input_dim: int,
        interaction_input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Feature encoders
        self.profile_encoder = FeatureEncoder(
            profile_input_dim,
            hidden_dims,
            dropout
        )
        
        self.activity_encoder = FeatureEncoder(
            activity_input_dim,
            hidden_dims,
            dropout
        )
        
        self.network_encoder = FeatureEncoder(
            network_input_dim,
            hidden_dims,
            dropout
        )
        
        self.interaction_encoder = FeatureEncoder(
            interaction_input_dim,
            hidden_dims,
            dropout
        )
        
        # Fusion layers
        fusion_input_dim = hidden_dims[-1] * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], output_dim)
        )
        
    def forward(
        self,
        profile_features: torch.Tensor,
        activity_features: torch.Tensor,
        network_features: torch.Tensor,
        interaction_features: torch.Tensor
    ) -> torch.Tensor:
        # Encode each feature type
        profile_encoded = self.profile_encoder(profile_features)
        activity_encoded = self.activity_encoder(activity_features)
        network_encoded = self.network_encoder(network_features)
        interaction_encoded = self.interaction_encoder(interaction_features)
        
        # Concatenate encoded features
        combined = torch.cat([
            profile_encoded,
            activity_encoded,
            network_encoded,
            interaction_encoded
        ], dim=1)
        
        # Fuse features
        output = self.fusion(combined)
        
        return output

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # Compute loss
        loss = -torch.log(
            torch.exp(pos_sim) / (
                torch.exp(pos_sim) + torch.exp(neg_sim)
            )
        )
        
        # Add margin to push negative pairs apart
        loss = loss + F.relu(neg_sim - pos_sim + self.margin)
        
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 1.0
    ):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()

class NetworkLoss(nn.Module):
    def __init__(
        self,
        contrastive_weight: float = 0.5,
        triplet_weight: float = 0.5
    ):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss()
        self.triplet_loss = TripletLoss()
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute individual losses
        contrastive_loss = self.contrastive_loss(anchor, positive, negative)
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        
        # Compute total loss
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.triplet_weight * triplet_loss
        )
        
        # Return loss components for logging
        loss_components = {
            'contrastive_loss': contrastive_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components 