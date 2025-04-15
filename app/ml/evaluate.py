import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from .models import NetworkModel, GNNModel
from .data import NetworkDataModule
from .train import NetworkTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        probabilities: Optional prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted')
    }
    
    if probabilities is not None:
        metrics.update({
            'roc_auc': roc_auc_score(labels, probabilities, multi_class='ovr'),
            'pr_auc': average_precision_score(labels, probabilities)
        })
        
    return metrics

def evaluate_model(
    model: Union[NetworkModel, GNNModel],
    data_module: NetworkDataModule,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Dict[str, float]]:
    """Evaluate model.
    
    Args:
        model: Network model
        data_module: Data module
        device: Device to use
        
    Returns:
        Dictionary of metrics for each split
    """
    model = model.to(device)
    model.eval()
    
    metrics = {}
    
    # Evaluate on each split
    for split, dataloader in [
        ('train', data_module.train_dataloader()),
        ('val', data_module.val_dataloader()),
        ('test', data_module.test_dataloader())
    ]:
        predictions = []
        labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    batch['features'],
                    batch.get('adj_matrix')
                )
                
                # Get predictions
                logits = outputs['embeddings']
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect predictions
                predictions.append(preds.cpu().numpy())
                labels.append(batch['label'].cpu().numpy())
                probabilities.append(probs.cpu().numpy())
                
        # Concatenate predictions
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        probabilities = np.concatenate(probabilities)
        
        # Compute metrics
        metrics[split] = compute_metrics(
            predictions,
            labels,
            probabilities
        )
        
    return metrics

def main():
    """Run evaluation."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate network model')
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
    
    # Evaluate model
    logger.info('Evaluating model...')
    metrics = evaluate_model(model, data_module, device)
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Saved metrics to {metrics_path}')
    
    # Print metrics
    for split, split_metrics in metrics.items():
        logger.info(f'\n{split.upper()} metrics:')
        for name, value in split_metrics.items():
            logger.info(f'{name}: {value:.4f}')
            
    logger.info('Evaluation complete')

if __name__ == '__main__':
    main() 