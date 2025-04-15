import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse

class NetworkDataset(Dataset):
    """Dataset for network data."""
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        adj_matrix: Optional[sparse.spmatrix] = None,
        triplets: Optional[List[Tuple[int, int, int]]] = None,
        transform: Optional[callable] = None
    ):
        """Initialize dataset.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            labels: Node labels [num_nodes]
            adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
            triplets: Optional list of (anchor, positive, negative) triplets
            transform: Optional transform to apply to features
        """
        self.features = features
        self.labels = labels
        self.adj_matrix = adj_matrix
        self.triplets = triplets
        self.transform = transform
        
        # Convert adj_matrix to torch tensor if provided
        if adj_matrix is not None:
            if sparse.issparse(adj_matrix):
                self.adj_matrix = torch.from_numpy(adj_matrix.toarray()).float()
            else:
                self.adj_matrix = torch.from_numpy(adj_matrix).float()
                
        # Convert triplets to tensor if provided
        if triplets is not None:
            self.triplets = torch.tensor(triplets, dtype=torch.long)
            
    def __len__(self) -> int:
        """Get dataset size.
        
        Returns:
            Number of samples
        """
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing features and labels
        """
        # Get features
        features = self.features[idx]
        
        # Apply transform if provided
        if self.transform is not None:
            features = self.transform(features)
            
        # Create sample dictionary
        sample = {
            'features': features,
            'labels': self.labels[idx]
        }
        
        # Add adj_matrix if available
        if self.adj_matrix is not None:
            sample['adj_matrix'] = self.adj_matrix[idx]
            
        # Add triplets if available
        if self.triplets is not None:
            # Get triplets where idx is the anchor
            mask = self.triplets[:, 0] == idx
            if mask.any():
                sample['triplets'] = self.triplets[mask]
                
        return sample
        
    @classmethod
    def from_numpy(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        adj_matrix: Optional[np.ndarray] = None,
        triplets: Optional[List[Tuple[int, int, int]]] = None,
        transform: Optional[callable] = None
    ) -> 'NetworkDataset':
        """Create dataset from numpy arrays.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            labels: Node labels [num_nodes]
            adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
            triplets: Optional list of (anchor, positive, negative) triplets
            transform: Optional transform to apply to features
            
        Returns:
            NetworkDataset instance
        """
        # Convert to torch tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()
        
        return cls(
            features=features,
            labels=labels,
            adj_matrix=adj_matrix,
            triplets=triplets,
            transform=transform
        )
        
    @classmethod
    def from_sparse(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        adj_matrix: sparse.spmatrix,
        triplets: Optional[List[Tuple[int, int, int]]] = None,
        transform: Optional[callable] = None
    ) -> 'NetworkDataset':
        """Create dataset from sparse adjacency matrix.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            labels: Node labels [num_nodes]
            adj_matrix: Sparse adjacency matrix [num_nodes, num_nodes]
            triplets: Optional list of (anchor, positive, negative) triplets
            transform: Optional transform to apply to features
            
        Returns:
            NetworkDataset instance
        """
        # Convert to torch tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()
        
        return cls(
            features=features,
            labels=labels,
            adj_matrix=adj_matrix,
            triplets=triplets,
            transform=transform
        )
        
    def get_triplets(
        self,
        num_triplets: Optional[int] = None,
        margin: float = 1.0
    ) -> torch.Tensor:
        """Generate triplets for training.
        
        Args:
            num_triplets: Number of triplets to generate (default: all possible)
            margin: Margin for triplet loss
            
        Returns:
            Tensor of triplets [num_triplets, 3]
        """
        if self.triplets is not None:
            if num_triplets is None:
                return self.triplets
            else:
                # Randomly sample triplets
                indices = torch.randperm(len(self.triplets))[:num_triplets]
                return self.triplets[indices]
                
        # Generate triplets from labels
        triplets = []
        labels = self.labels.numpy()
        
        for label in np.unique(labels):
            # Get indices of nodes with this label
            label_indices = np.where(labels == label)[0]
            
            # Get indices of nodes with different labels
            other_indices = np.where(labels != label)[0]
            
            # Generate triplets
            for anchor_idx in label_indices:
                # Get positive examples (same label)
                pos_indices = label_indices[label_indices != anchor_idx]
                if len(pos_indices) == 0:
                    continue
                    
                # Get negative examples (different label)
                neg_indices = other_indices
                if len(neg_indices) == 0:
                    continue
                    
                # Sample positive and negative examples
                pos_idx = np.random.choice(pos_indices)
                neg_idx = np.random.choice(neg_indices)
                
                triplets.append((anchor_idx, pos_idx, neg_idx))
                
        # Convert to tensor
        triplets = torch.tensor(triplets, dtype=torch.long)
        
        # Sample if requested
        if num_triplets is not None and len(triplets) > num_triplets:
            indices = torch.randperm(len(triplets))[:num_triplets]
            triplets = triplets[indices]
            
        return triplets 