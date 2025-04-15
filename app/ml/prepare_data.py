import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(data_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load raw data from directory.
    
    Args:
        data_dir: Data directory
        
    Returns:
        Dictionary containing:
            - features: Node features [num_nodes, feature_dim]
            - labels: Node labels [num_nodes]
            - adj_matrix: Adjacency matrix [num_nodes, num_nodes]
    """
    data_dir = Path(data_dir)
    
    # Load features
    features = np.load(data_dir / 'features.npy')
    logger.info(f'Loaded features with shape {features.shape}')
    
    # Load labels
    labels = np.load(data_dir / 'labels.npy')
    logger.info(f'Loaded labels with shape {labels.shape}')
    
    # Load adjacency matrix
    adj_matrix = np.load(data_dir / 'adj_matrix.npy')
    logger.info(f'Loaded adjacency matrix with shape {adj_matrix.shape}')
    
    return {
        'features': features,
        'labels': labels,
        'adj_matrix': adj_matrix
    }

def generate_triplets(
    labels: np.ndarray,
    num_triplets: int = 1000000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate triplets for triplet loss.
    
    Args:
        labels: Node labels [num_nodes]
        num_triplets: Number of triplets to generate
        seed: Random seed
        
    Returns:
        Tuple of (anchor, positive, negative) indices
    """
    np.random.seed(seed)
    
    num_nodes = len(labels)
    triplets = []
    
    # Generate triplets
    while len(triplets) < num_triplets:
        # Sample anchor
        anchor_idx = np.random.randint(num_nodes)
        anchor_label = labels[anchor_idx]
        
        # Sample positive
        pos_mask = labels == anchor_label
        pos_mask[anchor_idx] = False
        if not pos_mask.any():
            continue
        pos_idx = np.random.choice(np.where(pos_mask)[0])
        
        # Sample negative
        neg_mask = labels != anchor_label
        if not neg_mask.any():
            continue
        neg_idx = np.random.choice(np.where(neg_mask)[0])
        
        triplets.append((anchor_idx, pos_idx, neg_idx))
        
    # Convert to arrays
    triplets = np.array(triplets)
    
    return (
        triplets[:, 0],
        triplets[:, 1],
        triplets[:, 2]
    )

def preprocess_data(
    data: Dict[str, np.ndarray],
    normalize: bool = True,
    num_triplets: int = 1000000,
    seed: int = 42
) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
    """Preprocess data.
    
    Args:
        data: Raw data dictionary
        normalize: Whether to normalize features
        num_triplets: Number of triplets to generate
        seed: Random seed
        
    Returns:
        Dictionary of preprocessed data
    """
    # Extract data
    features = data['features']
    labels = data['labels']
    adj_matrix = data['adj_matrix']
    
    # Normalize features
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        logger.info('Normalized features')
        
    # Generate triplets
    triplets = generate_triplets(
        labels,
        num_triplets=num_triplets,
        seed=seed
    )
    logger.info(f'Generated {len(triplets[0])} triplets')
    
    # Convert to tensors
    processed_data = {
        'features': torch.from_numpy(features).float(),
        'labels': torch.from_numpy(labels).long(),
        'adj_matrix': torch.from_numpy(adj_matrix).float(),
        'triplets': tuple(torch.from_numpy(t).long() for t in triplets)
    }
    
    return processed_data

def save_data(
    data: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    output_dir: Union[str, Path]
):
    """Save preprocessed data.
    
    Args:
        data: Preprocessed data dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tensors
    for name, tensor in data.items():
        if isinstance(tensor, tuple):
            # Save triplet tensors
            for i, t in enumerate(tensor):
                torch.save(t, output_dir / f'{name}_{i}.pt')
        else:
            # Save single tensor
            torch.save(tensor, output_dir / f'{name}.pt')
            
    logger.info(f'Saved preprocessed data to {output_dir}')

def main():
    """Run data preparation."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Prepare network data')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Raw data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize features')
    parser.add_argument('--num_triplets', type=int, default=1000000,
                      help='Number of triplets to generate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    args = parser.parse_args()
    
    # Load raw data
    logger.info('Loading raw data...')
    raw_data = load_raw_data(args.data_dir)
    
    # Preprocess data
    logger.info('Preprocessing data...')
    processed_data = preprocess_data(
        raw_data,
        normalize=args.normalize,
        num_triplets=args.num_triplets,
        seed=args.seed
    )
    
    # Save data
    logger.info('Saving preprocessed data...')
    save_data(processed_data, args.output_dir)
    
    logger.info('Data preparation complete')

if __name__ == '__main__':
    main() 