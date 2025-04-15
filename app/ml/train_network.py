import torch
import argparse
from pathlib import Path
import logging
from typing import Optional

from .models import NetworkModel, GNNModel
from .data import NetworkDataModule, FeatureTransform
from .train import NetworkTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train network model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='network',
                      choices=['network', 'gnn'],
                      help='Model type')
    parser.add_argument('--input_dim', type=int, required=True,
                      help='Input feature dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                      default=[256, 128],
                      help='Hidden layer dimensions')
    parser.add_argument('--output_dim', type=int, default=64,
                      help='Output embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--use_batch_norm', action='store_true',
                      help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='Early stopping patience')
    parser.add_argument('--checkpoint_frequency', type=int, default=5,
                      help='Checkpoint frequency in epochs')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Data directory')
    parser.add_argument('--val_split', type=float, default=0.1,
                      help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                      help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory')
    
    return parser.parse_args()

def main():
    """Run training."""
    # Parse arguments
    args = parse_args()
    
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
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Create model
    if args.model == 'network':
        model = NetworkModel(
            input_dim=args.input_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm
        )
    else:
        model = GNNModel(
            input_dim=args.input_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm
        )
    
    # Create trainer
    trainer = NetworkTrainer(
        model=model,
        data_module=data_module,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_dir=output_dir
    )
    
    # Train model
    trainer.train(
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_frequency=args.checkpoint_frequency
    )
    
    logger.info('Training complete')

if __name__ == '__main__':
    main() 