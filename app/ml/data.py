import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class NetworkDataset(Dataset):
    """Dataset for network data."""
    def __init__(
        self,
        features: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        adj_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        triplets: Optional[Tuple[Union[torch.Tensor, np.ndarray], ...]] = None,
        transform: Optional[callable] = None
    ):
        """Initialize dataset.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            labels: Node labels [num_nodes]
            adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
            triplets: Optional tuple of (anchor, positive, negative) indices
            transform: Optional transform to apply to features
        """
        # Convert inputs to tensors
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        
        if adj_matrix is not None:
            self.adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float32)
        else:
            self.adj_matrix = None
            
        if triplets is not None:
            self.triplets = tuple(
                torch.as_tensor(t, dtype=torch.long) for t in triplets
            )
        else:
            self.triplets = None
            
        self.transform = transform
        
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - features: Node features [feature_dim]
                - label: Node label
                - adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
                - triplets: Optional tuple of (anchor, positive, negative) indices
        """
        sample = {
            'features': self.features[idx],
            'label': self.labels[idx]
        }
        
        if self.adj_matrix is not None:
            sample['adj_matrix'] = self.adj_matrix
            
        if self.triplets is not None:
            sample['triplets'] = tuple(t[idx] for t in self.triplets)
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
        
class NetworkDataModule:
    """Data module for network data."""
    def __init__(
        self,
        features: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        adj_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        triplets: Optional[Tuple[Union[torch.Tensor, np.ndarray], ...]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42
    ):
        """Initialize data module.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            labels: Node labels [num_nodes]
            adj_matrix: Optional adjacency matrix [num_nodes, num_nodes]
            triplets: Optional tuple of (anchor, positive, negative) indices
            batch_size: Batch size
            num_workers: Number of workers for data loading
            val_split: Validation split ratio
            test_split: Test split ratio
            seed: Random seed
        """
        self.features = features
        self.labels = labels
        self.adj_matrix = adj_matrix
        self.triplets = triplets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create train/val/test splits
        self._create_splits()
        
    def _create_splits(self):
        """Create train/val/test splits."""
        num_samples = len(self.features)
        indices = np.random.permutation(num_samples)
        
        # Compute split sizes
        test_size = int(num_samples * self.test_split)
        val_size = int(num_samples * self.val_split)
        train_size = num_samples - test_size - val_size
        
        # Create splits
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:train_size + val_size]
        self.test_indices = indices[train_size + val_size:]
        
    def _get_dataset(
        self,
        indices: np.ndarray,
        transform: Optional[callable] = None
    ) -> NetworkDataset:
        """Get dataset for indices.
        
        Args:
            indices: Sample indices
            transform: Optional transform to apply
            
        Returns:
            NetworkDataset
        """
        features = self.features[indices]
        labels = self.labels[indices]
        
        if self.adj_matrix is not None:
            adj_matrix = self.adj_matrix[indices][:, indices]
        else:
            adj_matrix = None
            
        if self.triplets is not None:
            triplets = tuple(t[indices] for t in self.triplets)
        else:
            triplets = None
            
        return NetworkDataset(
            features=features,
            labels=labels,
            adj_matrix=adj_matrix,
            triplets=triplets,
            transform=transform
        )
        
    def setup(self, stage: Optional[str] = None):
        """Setup data module.
        
        Args:
            stage: Optional stage ('fit' or 'test')
        """
        # Create datasets
        self.train_dataset = self._get_dataset(self.train_indices)
        self.val_dataset = self._get_dataset(self.val_indices)
        self.test_dataset = self._get_dataset(self.test_indices)
        
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
class FeatureTransform:
    """Transform for network features."""
    def __init__(
        self,
        scaler: Optional[StandardScaler] = None,
        normalize: bool = True
    ):
        """Initialize transform.
        
        Args:
            scaler: Optional StandardScaler for feature normalization
            normalize: Whether to normalize features
        """
        self.scaler = scaler
        self.normalize = normalize
        
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transform to sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Transformed sample
        """
        features = sample['features']
        
        if self.scaler is not None:
            features = torch.from_numpy(
                self.scaler.transform(features.numpy())
            ).float()
            
        if self.normalize:
            features = F.normalize(features, p=2, dim=0)
            
        sample['features'] = features
        return sample 