import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning embeddings."""
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            x1: First embedding [batch_size, embedding_dim]
            x2: Second embedding [batch_size, embedding_dim]
            y: Labels (1 for positive pairs, 0 for negative pairs) [batch_size]
            
        Returns:
            Contrastive loss
        """
        # Compute squared Euclidean distance
        dist = F.pairwise_distance(x1, x2)
        
        # Compute loss
        loss = y * dist.pow(2) + (1 - y) * F.relu(self.margin - dist).pow(2)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            
class TripletLoss(nn.Module):
    """Triplet loss for learning embeddings."""
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for negative pairs
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        # Compute loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            
class NetworkLoss(nn.Module):
    """Loss functions for network models."""
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        triplet_weight: float = 0.1,
        margin: float = 1.0,
        temperature: float = 0.07
    ):
        """Initialize loss functions.
        
        Args:
            contrastive_weight: Weight for contrastive loss
            triplet_weight: Weight for triplet loss
            margin: Margin for triplet loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        self.margin = margin
        self.temperature = temperature
        
    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            labels: Node labels [num_nodes]
            
        Returns:
            Contrastive loss
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create label matrix
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Compute positive and negative masks
        pos_mask = label_matrix.float()
        neg_mask = (~label_matrix).float()
        
        # Compute log probabilities
        log_prob = F.log_softmax(sim_matrix, dim=1)
        
        # Compute loss
        loss = -torch.mean(
            pos_mask * log_prob - neg_mask * log_prob
        )
        
        return loss
        
    def triplet_loss(
        self,
        embeddings: torch.Tensor,
        triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            triplets: Tuple of (anchor, positive, negative) indices
            
        Returns:
            Triplet loss
        """
        anchor_idx, pos_idx, neg_idx = triplets
        
        # Get embeddings for triplets
        anchor_emb = embeddings[anchor_idx]
        pos_emb = embeddings[pos_idx]
        neg_emb = embeddings[neg_idx]
        
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
        neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
        
        # Compute loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return torch.mean(loss)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        triplets: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            labels: Node labels [num_nodes]
            triplets: Optional tuple of (anchor, positive, negative) indices
            adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Compute contrastive loss
        contrastive_loss = self.contrastive_loss(embeddings, labels)
        losses['contrastive'] = contrastive_loss
        
        # Compute triplet loss if provided
        if triplets is not None:
            triplet_loss = self.triplet_loss(embeddings, triplets)
            losses['triplet'] = triplet_loss
            
        # Compute total loss
        total_loss = self.contrastive_weight * contrastive_loss
        if triplets is not None:
            total_loss += self.triplet_weight * triplet_loss
            
        losses['total'] = total_loss
        
        return losses 