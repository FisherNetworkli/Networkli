import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .models import NetworkModel, GNNModel
from .data import NetworkDataModule
from .train import NetworkTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_embeddings(
    model: Union[NetworkModel, GNNModel],
    data_module: NetworkDataModule,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute model embeddings.
    
    Args:
        model: Network model
        data_module: Data module
        device: Device to use
        
    Returns:
        Tuple of (embeddings, labels)
    """
    model = model.to(device)
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                batch['features'],
                batch.get('adj_matrix')
            )
            
            # Collect embeddings
            embeddings.append(outputs['embeddings'].cpu().numpy())
            labels.append(batch['label'].cpu().numpy())
            
    # Concatenate embeddings
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    return embeddings, labels

def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    output_dir: Optional[Union[str, Path]] = None
):
    """Visualize embeddings.
    
    Args:
        embeddings: Model embeddings
        labels: Node labels
        method: Visualization method ('tsne' or 'pca')
        n_components: Number of components
        output_dir: Output directory
    """
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        reducer = PCA(n_components=n_components)
        
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot embeddings
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6
    )
    
    # Add legend
    plt.colorbar(scatter)
    
    # Add labels
    plt.title(f'{method.upper()} visualization of network embeddings')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    
    # Save figure
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / f'embeddings_{method}.png')
        logger.info(f'Saved visualization to {output_dir / f"embeddings_{method}.png"}')
        
    plt.close()
    
def visualize_adjacency(
    adj_matrix: np.ndarray,
    labels: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None
):
    """Visualize adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix
        labels: Node labels
        output_dir: Output directory
    """
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot adjacency matrix
    sns.heatmap(
        adj_matrix,
        cmap='YlOrRd',
        xticklabels=False,
        yticklabels=False
    )
    
    # Add labels
    plt.title('Adjacency matrix')
    plt.xlabel('Node index')
    plt.ylabel('Node index')
    
    # Save figure
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / 'adjacency.png')
        logger.info(f'Saved visualization to {output_dir / "adjacency.png"}')
        
    plt.close()
    
def main():
    """Run visualization."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize network model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Data directory')
    parser.add_argument('--model_type', type=str, default='network',
                      choices=['network', 'gnn'],
                      help='Model type')
    parser.add_argument('--input_dim', type=int, required=True,
                      help='Input feature dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                      default=[256, 128],
                      help='Hidden layer dimensions')
    parser.add_argument('--output_dim', type=int, default=64,
                      help='Output embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers')
    parser.add_argument('--method', type=str, default='tsne',
                      choices=['tsne', 'pca'],
                      help='Visualization method')
    parser.add_argument('--n_components', type=int, default=2,
                      help='Number of components')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    
    # Load data
    data_module = NetworkDataModule(
        features=torch.load(Path(args.data_dir) / 'features.pt'),
        labels=torch.load(Path(args.data_dir) / 'labels.pt'),
        adj_matrix=torch.load(Path(args.data_dir) / 'adj_matrix.pt'),
        triplets=torch.load(Path(args.data_dir) / 'triplets.pt'),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    if args.model_type == 'network':
        model = NetworkModel(
            input_dim=args.input_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim
        )
    else:
        model = GNNModel(
            input_dim=args.input_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Loaded checkpoint from {args.model_path}')
    
    # Create trainer
    trainer = NetworkTrainer(
        model=model,
        data_module=data_module,
        device=device
    )
    
    # Compute embeddings
    logger.info('Computing embeddings...')
    embeddings, labels = compute_embeddings(model, data_module, device)
    
    # Visualize embeddings
    logger.info('Visualizing embeddings...')
    visualize_embeddings(
        embeddings,
        labels,
        method=args.method,
        n_components=args.n_components,
        output_dir=output_dir
    )
    
    # Visualize adjacency matrix
    logger.info('Visualizing adjacency matrix...')
    visualize_adjacency(
        data_module.adj_matrix.numpy(),
        labels,
        output_dir=output_dir
    )
    
    logger.info('Visualization complete')

if __name__ == '__main__':
    main() 