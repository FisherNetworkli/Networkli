import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class DataLoader:
    def __init__(
        self,
        data_dir: str = 'data',
        cache_dir: str = 'cache'
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_user_profiles(self) -> pd.DataFrame:
        """
        Load user profile data from CSV or database.
        
        Returns:
            DataFrame containing user profiles
        """
        profile_path = self.data_dir / 'user_profiles.csv'
        
        if not profile_path.exists():
            self.logger.warning(f'Profile data not found at {profile_path}')
            return pd.DataFrame()
            
        return pd.read_csv(profile_path)
        
    def load_user_activities(self) -> pd.DataFrame:
        """
        Load user activity data from CSV or database.
        
        Returns:
            DataFrame containing user activities
        """
        activity_path = self.data_dir / 'user_activities.csv'
        
        if not activity_path.exists():
            self.logger.warning(f'Activity data not found at {activity_path}')
            return pd.DataFrame()
            
        return pd.read_csv(activity_path)
        
    def load_network_data(self) -> nx.Graph:
        """
        Load network data and construct graph.
        
        Returns:
            NetworkX graph object
        """
        network_path = self.data_dir / 'network_data.csv'
        
        if not network_path.exists():
            self.logger.warning(f'Network data not found at {network_path}')
            return nx.Graph()
            
        # Load edge list
        edges_df = pd.read_csv(network_path)
        
        # Construct graph
        G = nx.from_pandas_edgelist(
            edges_df,
            source='user_id',
            target='connection_id',
            edge_attr=True
        )
        
        return G
        
    def load_interaction_data(self) -> pd.DataFrame:
        """
        Load user interaction data from CSV or database.
        
        Returns:
            DataFrame containing user interactions
        """
        interaction_path = self.data_dir / 'interactions.csv'
        
        if not interaction_path.exists():
            self.logger.warning(f'Interaction data not found at {interaction_path}')
            return pd.DataFrame()
            
        return pd.read_csv(interaction_path)
        
    def preprocess_data(
        self,
        profiles_df: pd.DataFrame,
        activities_df: pd.DataFrame,
        network: nx.Graph,
        interactions_df: pd.DataFrame
    ) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
        """
        Preprocess data for training.
        
        Args:
            profiles_df: User profile DataFrame
            activities_df: User activity DataFrame
            network: NetworkX graph
            interactions_df: User interaction DataFrame
            
        Returns:
            Tuple of (train_data, val_data)
        """
        # Merge all data
        merged_df = profiles_df.merge(
            activities_df,
            on='user_id',
            how='left'
        ).merge(
            interactions_df,
            on='user_id',
            how='left'
        )
        
        # Extract features
        features = []
        for _, row in merged_df.iterrows():
            user_id = row['user_id']
            
            # Get user's network neighbors
            neighbors = list(network.neighbors(user_id))
            
            # Get interaction features
            interactions = interactions_df[
                interactions_df['user_id'] == user_id
            ]
            
            feature_dict = {
                'user_id': user_id,
                'profile_features': {
                    'skills': row['skills'].split(',') if pd.notna(row['skills']) else [],
                    'interests': row['interests'].split(',') if pd.notna(row['interests']) else [],
                    'experience_years': row['experience_years'],
                    'connections_count': len(neighbors)
                },
                'activity_features': {
                    'login_frequency': row['login_frequency'],
                    'message_frequency': row['message_frequency'],
                    'post_frequency': row['post_frequency']
                },
                'network_features': {
                    'neighbors': neighbors,
                    'degree_centrality': nx.degree_centrality(network)[user_id]
                },
                'interaction_features': {
                    'message_count': len(interactions),
                    'response_rate': interactions['response_rate'].mean()
                }
            }
            
            features.append(feature_dict)
            
        # Split into train and validation sets
        train_data, val_data = train_test_split(
            features,
            test_size=0.2,
            random_state=42
        )
        
        return train_data, val_data
        
    def load_and_preprocess(
        self,
        cache: bool = True
    ) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
        """
        Load and preprocess all data.
        
        Args:
            cache: Whether to cache preprocessed data
            
        Returns:
            Tuple of (train_data, val_data)
        """
        cache_path = self.cache_dir / 'preprocessed_data.json'
        
        # Check cache
        if cache and cache_path.exists():
            self.logger.info('Loading preprocessed data from cache')
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data['train_data'], data['val_data']
                
        # Load raw data
        self.logger.info('Loading raw data')
        profiles_df = self.load_user_profiles()
        activities_df = self.load_user_activities()
        network = self.load_network_data()
        interactions_df = self.load_interaction_data()
        
        # Preprocess data
        self.logger.info('Preprocessing data')
        train_data, val_data = self.preprocess_data(
            profiles_df,
            activities_df,
            network,
            interactions_df
        )
        
        # Cache preprocessed data
        if cache:
            self.logger.info('Caching preprocessed data')
            with open(cache_path, 'w') as f:
                json.dump({
                    'train_data': train_data,
                    'val_data': val_data
                }, f)
                
        return train_data, val_data 

class NetworkDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        feature_dims: Dict[str, int],
        transform: Optional[callable] = None
    ):
        """
        Dataset for network data.
        
        Args:
            data: List of data samples
            feature_dims: Dictionary of feature dimensions
            transform: Optional transform to apply to features
        """
        self.data = data
        self.feature_dims = feature_dims
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Extract features
        features = {
            'profile': torch.tensor(sample['profile_features'], dtype=torch.float32),
            'activity': torch.tensor(sample['activity_features'], dtype=torch.float32),
            'network': torch.tensor(sample['network_features'], dtype=torch.float32),
            'interaction': torch.tensor(sample['interaction_features'], dtype=torch.float32)
        }
        
        # Apply transform if specified
        if self.transform:
            features = self.transform(features)
            
        return features

class TripletSampler:
    def __init__(
        self,
        dataset: NetworkDataset,
        num_negatives: int = 1,
        hard_negative_mining: bool = False
    ):
        """
        Sampler for generating triplets.
        
        Args:
            dataset: Dataset to sample from
            num_negatives: Number of negative samples per anchor-positive pair
            hard_negative_mining: Whether to use hard negative mining
        """
        self.dataset = dataset
        self.num_negatives = num_negatives
        self.hard_negative_mining = hard_negative_mining
        
        # Build similarity matrix for hard negative mining
        if hard_negative_mining:
            self.similarity_matrix = self._build_similarity_matrix()
            
    def _build_similarity_matrix(self) -> torch.Tensor:
        """Build similarity matrix for hard negative mining."""
        n = len(self.dataset)
        similarity_matrix = torch.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compute similarity between samples
                sim = self._compute_similarity(
                    self.dataset[i],
                    self.dataset[j]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
        return similarity_matrix
        
    def _compute_similarity(
        self,
        sample1: Dict[str, torch.Tensor],
        sample2: Dict[str, torch.Tensor]
    ) -> float:
        """Compute similarity between two samples."""
        # Concatenate all features
        feat1 = torch.cat([
            sample1['profile'],
            sample1['activity'],
            sample1['network'],
            sample1['interaction']
        ])
        
        feat2 = torch.cat([
            sample2['profile'],
            sample2['activity'],
            sample2['network'],
            sample2['interaction']
        ])
        
        # Compute cosine similarity
        return torch.nn.functional.cosine_similarity(
            feat1.unsqueeze(0),
            feat2.unsqueeze(0)
        ).item()
        
    def sample_triplets(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Sample triplets for training."""
        triplets = []
        
        for _ in range(batch_size):
            # Sample anchor
            anchor_idx = np.random.randint(len(self.dataset))
            anchor = self.dataset[anchor_idx]
            
            # Sample positive (similar to anchor)
            if self.hard_negative_mining:
                # Find most similar sample as positive
                similarities = self.similarity_matrix[anchor_idx]
                positive_idx = similarities.argsort(descending=True)[1].item()
            else:
                # Random positive from same class/group
                positive_idx = np.random.randint(len(self.dataset))
                while positive_idx == anchor_idx:
                    positive_idx = np.random.randint(len(self.dataset))
                    
            positive = self.dataset[positive_idx]
            
            # Sample negatives
            negatives = []
            for _ in range(self.num_negatives):
                if self.hard_negative_mining:
                    # Find least similar sample as negative
                    similarities = self.similarity_matrix[anchor_idx]
                    negative_idx = similarities.argsort()[0].item()
                else:
                    # Random negative from different class/group
                    negative_idx = np.random.randint(len(self.dataset))
                    while negative_idx in [anchor_idx, positive_idx]:
                        negative_idx = np.random.randint(len(self.dataset))
                        
                negatives.append(self.dataset[negative_idx])
                
            # Create triplet
            triplet = {
                'anchor_profile': anchor['profile'],
                'anchor_activity': anchor['activity'],
                'anchor_network': anchor['network'],
                'anchor_interaction': anchor['interaction'],
                
                'positive_profile': positive['profile'],
                'positive_activity': positive['activity'],
                'positive_network': positive['network'],
                'positive_interaction': positive['interaction'],
                
                'negative_profile': negatives[0]['profile'],
                'negative_activity': negatives[0]['activity'],
                'negative_network': negatives[0]['network'],
                'negative_interaction': negatives[0]['interaction']
            }
            
            triplets.append(triplet)
            
        return triplets

class DataLoaderUtil:
    def __init__(
        self,
        data_path: str,
        feature_dims: Dict[str, int],
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[callable] = None,
        hard_negative_mining: bool = False
    ):
        """
        Data loader utility.
        
        Args:
            data_path: Path to data directory
            feature_dims: Dictionary of feature dimensions
            batch_size: Batch size
            num_workers: Number of worker processes
            transform: Optional transform to apply to features
            hard_negative_mining: Whether to use hard negative mining
        """
        self.data_path = Path(data_path)
        self.feature_dims = feature_dims
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.hard_negative_mining = hard_negative_mining
        
        # Load data
        self.train_data = self._load_data('train')
        self.val_data = self._load_data('val')
        
        # Create datasets
        self.train_dataset = NetworkDataset(
            self.train_data,
            feature_dims,
            transform
        )
        self.val_dataset = NetworkDataset(
            self.val_data,
            feature_dims,
            transform
        )
        
        # Create samplers
        self.train_sampler = TripletSampler(
            self.train_dataset,
            hard_negative_mining=hard_negative_mining
        )
        self.val_sampler = TripletSampler(
            self.val_dataset,
            hard_negative_mining=hard_negative_mining
        )
        
    def _load_data(self, split: str) -> List[Dict]:
        """Load data from JSON file."""
        path = self.data_path / f'{split}.json'
        with open(path) as f:
            return json.load(f)
            
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=lambda x: self.train_sampler.sample_triplets(len(x))
        )
        
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda x: self.val_sampler.sample_triplets(len(x))
        ) 