"""
User embedding module using graph neural networks.

This module handles the creation and retrieval of user embeddings
using a graph neural network (GNN) model.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from settings import settings

logger = logging.getLogger(__name__)

# Singleton to hold the loaded model
_EMBEDDING_MODEL = None

class NetworkliGNN(nn.Module):
    """Graph Neural Network model for user embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
        """
        super(NetworkliGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph convolution layers
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output embedding layer
        self.embedding_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input feature tensor
            adj: Adjacency matrix (optional)
            
        Returns:
            Embedding tensor
        """
        # Feature encoding
        h = self.feature_encoder(x)
        
        # Graph convolution (if adjacency matrix provided)
        if adj is not None:
            h = F.relu(self.conv1(torch.matmul(adj, h)))
            h = F.relu(self.conv2(torch.matmul(adj, h)))
        
        # Generate embeddings
        embeddings = self.embedding_layer(h)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def get_embedding_model() -> Optional[NetworkliGNN]:
    """Get the singleton embedding model instance, loading it if necessary."""
    global _EMBEDDING_MODEL
    
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    
    try:
        # Load the model
        model_path = settings.ML_MODEL_PATH
        gnn_model_path = settings.NETWORKLI_GNN_MODEL
        
        if not model_path or not gnn_model_path:
            logger.warning("GNN model path not configured")
            return None
        
        full_path = Path(model_path) / gnn_model_path
        
        if not full_path.exists():
            logger.warning(f"GNN model not found at: {full_path}")
            return None
        
        # Load model configuration
        config_path = full_path.parent / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "input_dim": 128,
                "hidden_dim": 64,
                "output_dim": 32
            }
        
        # Initialize model
        model = NetworkliGNN(
            input_dim=config.get("input_dim", 128),
            hidden_dim=config.get("hidden_dim", 64),
            output_dim=config.get("output_dim", 32)
        )
        
        # Load weights
        model.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        
        _EMBEDDING_MODEL = model
        logger.info(f"Loaded GNN model from {full_path}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading GNN model: {e}")
        return None

def encode_user_profile(profile: Dict[str, Any]) -> torch.Tensor:
    """
    Encode a user profile into a feature vector for the GNN model.
    
    Args:
        profile: User profile dictionary
        
    Returns:
        Feature tensor
    """
    # Extract user features
    features = []
    
    # Basic text features
    industry = profile.get("industry", "")
    location = profile.get("location", "")
    title = profile.get("title", "")
    experience = profile.get("experience_level", "")
    
    # Skills and interests
    skills = profile.get("skills", [])
    interests = profile.get("interests", [])
    
    # TODO: Implement proper feature encoding based on your schema
    # For now, use a simple placeholder embedding of zeros
    feature_vector = np.zeros(128, dtype=np.float32)
    
    return torch.tensor(feature_vector).unsqueeze(0)  # Add batch dimension

def load_user_embedding(profile: Dict[str, Any], model: NetworkliGNN) -> np.ndarray:
    """
    Generate an embedding for a user using the GNN model.
    
    Args:
        profile: User profile dictionary
        model: Loaded GNN model
        
    Returns:
        Embedding vector as numpy array
    """
    try:
        # Encode profile to feature tensor
        features = encode_user_profile(profile)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(features)
        
        # Convert to numpy array
        return embedding.squeeze().cpu().numpy()
    
    except Exception as e:
        logger.error(f"Error generating user embedding: {e}")
        return np.zeros(model.output_dim, dtype=np.float32)  # Return zero embedding on error 