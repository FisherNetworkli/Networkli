import torch
from torch_geometric.data import Data, Dataset, Batch
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import networkx as nx

class NetworkliDataset(Dataset):
    """Custom dataset for Networkli graph data."""
    
    def __init__(self, 
                 node_features: torch.Tensor,
                 edge_index: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            node_features: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity in COO format [2, num_edges]
            labels: Optional node labels [num_nodes]
            transform: Optional transform to be applied on the data
        """
        super().__init__(transform)
        self.node_features = node_features
        self.edge_index = edge_index
        self.labels = labels
        
    def len(self) -> int:
        return 1  # Single graph
        
    def get(self, idx: int) -> Data:
        """Get a single graph."""
        data = Data(
            x=self.node_features,
            edge_index=self.edge_index
        )
        if self.labels is not None:
            data.y = self.labels
        return data

def process_user_data(users: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process raw user data into graph format.
    
    Args:
        users: List of user dictionaries containing profile information
        
    Returns:
        node_features: Tensor of node features
        edge_index: Tensor of edge indices in COO format
    """
    # Create feature vectors for each user
    feature_vectors = []
    user_id_to_idx = {}
    
    for idx, user in enumerate(users):
        user_id_to_idx[user['id']] = idx
        
        # Extract numerical features
        features = [
            float(user.get('experience_years', 0)),
            len(user.get('skills', [])),
            len(user.get('interests', [])),
            user.get('connection_count', 0),
            user.get('activity_score', 0)
        ]
        
        # Add skill embeddings (placeholder - replace with actual embeddings)
        skill_embedding = torch.randn(32)  # 32-dim skill embedding
        features.extend(skill_embedding.tolist())
        
        feature_vectors.append(features)
    
    # Convert to tensor
    node_features = torch.tensor(feature_vectors, dtype=torch.float)
    
    # Create edge index (connections between users)
    edge_list = []
    for user in users:
        source_idx = user_id_to_idx[user['id']]
        for connection_id in user.get('connections', []):
            if connection_id in user_id_to_idx:
                target_idx = user_id_to_idx[connection_id]
                edge_list.append([source_idx, target_idx])
                edge_list.append([target_idx, source_idx])  # Add reverse edge
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    return node_features, edge_index

def create_similarity_matrix(node_features: torch.Tensor) -> torch.Tensor:
    """
    Create a similarity matrix between nodes based on their features.
    
    Args:
        node_features: Node feature matrix [num_nodes, num_features]
        
    Returns:
        similarity_matrix: Pairwise similarity matrix [num_nodes, num_nodes]
    """
    # Normalize features
    normalized_features = torch.nn.functional.normalize(node_features, p=2, dim=1)
    
    # Compute cosine similarity
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    
    return similarity_matrix

def generate_negative_samples(edge_index: torch.Tensor, 
                            num_nodes: int,
                            num_negative: int) -> torch.Tensor:
    """
    Generate negative edge samples for training.
    
    Args:
        edge_index: Positive edges in COO format
        num_nodes: Total number of nodes
        num_negative: Number of negative samples to generate
        
    Returns:
        negative_edges: Tensor of negative edge samples
    """
    existing_edges = set(map(tuple, edge_index.t().tolist()))
    negative_edges = []
    
    while len(negative_edges) < num_negative:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j and (i, j) not in existing_edges:
            negative_edges.append([i, j])
    
    return torch.tensor(negative_edges, dtype=torch.long).t() 