import torch
import argparse
from pathlib import Path
import logging
import json
from typing import Dict, Any

from .models import NetworkliGNN
from .trainer import NetworkliTrainer
from .data_utils import NetworkliDataset, process_user_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from a JSON file."""
    with open(config_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Train the Networkli recommendation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    logger.info('Loading data...')
    data_dir = Path(args.data_dir)
    
    # Load users from JSON
    with open(data_dir / 'users.json') as f:
        users = json.load(f)
    
    # Process data
    node_features, edge_index = process_user_data(users)
    
    # Create dataset
    dataset = NetworkliDataset(
        node_features=node_features,
        edge_index=edge_index
    )
    
    # Create model
    logger.info('Creating model...')
    model = NetworkliGNN(
        input_dim=node_features.size(1),
        hidden_dim=config['model']['hidden_dim'],
        embedding_dim=config['model']['embedding_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    # Create trainer
    trainer = NetworkliTrainer(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Train model
    logger.info('Starting training...')
    history = trainer.train(
        train_dataset=dataset,
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        patience=config['training']['patience'],
        save_dir=str(save_dir)
    )
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info('Training complete!')
    logger.info(f'Best validation loss: {history["best_val_loss"]:.4f}')
    logger.info(f'Best epoch: {history["best_epoch"]}')

if __name__ == '__main__':
    main() 