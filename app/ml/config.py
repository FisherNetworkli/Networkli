from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout: float = 0.1
    activation: str = 'relu'
    batch_norm: bool = True
    residual: bool = True

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    
    # Loss weights
    contrastive_weight: float = 1.0
    triplet_weight: float = 0.5
    network_weight: float = 0.3
    
    # Loss parameters
    temperature: float = 0.07
    margin: float = 1.0
    distance: str = 'euclidean'

@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Config object
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data'])
        )
        
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

@dataclass
class ModelConfig:
    """Model configuration."""
    # Feature dimensions
    profile_dim: int = 128
    activity_dim: int = 64
    network_dim: int = 256
    interaction_dim: int = 32
    
    # Encoder dimensions
    hidden_dims: List[int] = (512, 256)
    embedding_dim: int = 128
    
    # Dropout rates
    dropout_rate: float = 0.3
    
    # Activation function
    activation: str = 'relu'
    
    # Layer normalization
    use_layer_norm: bool = True
    
    # Feature fusion
    fusion_method: str = 'concat'  # Options: 'concat', 'attention', 'sum'
    
    @property
    def feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions."""
        return {
            'profile': self.profile_dim,
            'activity': self.activity_dim,
            'network': self.network_dim,
            'interaction': self.interaction_dim
        }

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizer
    optimizer: str = 'adam'  # Options: 'adam', 'sgd', 'adamw'
    scheduler: str = 'cosine'  # Options: 'cosine', 'step', 'plateau'
    
    # Loss weights
    contrastive_weight: float = 1.0
    triplet_weight: float = 0.5
    network_weight: float = 0.3
    
    # Triplet sampling
    num_negatives: int = 1
    hard_negative_mining: bool = True
    margin: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_frequency: int = 5  # Save every N epochs
    
    # Logging
    log_dir: str = 'logs'
    log_frequency: int = 100  # Log every N batches
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision
    
    # Gradient clipping
    gradient_clip_val: Optional[float] = 1.0

@dataclass
class DataConfig:
    """Data configuration."""
    # Data paths
    data_dir: str = 'data'
    train_file: str = 'train.json'
    val_file: str = 'val.json'
    test_file: str = 'test.json'
    
    # Data preprocessing
    normalize_features: bool = True
    augment_data: bool = True
    
    # Feature selection
    use_profile_features: bool = True
    use_activity_features: bool = True
    use_network_features: bool = True
    use_interaction_features: bool = True
    
    # Data splitting
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

@dataclass
class Config:
    """Main configuration."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': {
                'profile_dim': self.model.profile_dim,
                'activity_dim': self.model.activity_dim,
                'network_dim': self.model.network_dim,
                'interaction_dim': self.model.interaction_dim,
                'hidden_dims': self.model.hidden_dims,
                'embedding_dim': self.model.embedding_dim,
                'dropout_rate': self.model.dropout_rate,
                'activation': self.model.activation,
                'use_layer_norm': self.model.use_layer_norm,
                'fusion_method': self.model.fusion_method
            },
            'training': {
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'optimizer': self.training.optimizer,
                'scheduler': self.training.scheduler,
                'contrastive_weight': self.training.contrastive_weight,
                'triplet_weight': self.training.triplet_weight,
                'network_weight': self.training.network_weight,
                'num_negatives': self.training.num_negatives,
                'hard_negative_mining': self.training.hard_negative_mining,
                'margin': self.training.margin,
                'early_stopping_patience': self.training.early_stopping_patience,
                'early_stopping_min_delta': self.training.early_stopping_min_delta,
                'checkpoint_dir': self.training.checkpoint_dir,
                'save_frequency': self.training.save_frequency,
                'log_dir': self.training.log_dir,
                'log_frequency': self.training.log_frequency,
                'device': self.training.device,
                'num_workers': self.training.num_workers,
                'use_amp': self.training.use_amp,
                'gradient_clip_val': self.training.gradient_clip_val
            },
            'data': {
                'data_dir': self.data.data_dir,
                'train_file': self.data.train_file,
                'val_file': self.data.val_file,
                'test_file': self.data.test_file,
                'normalize_features': self.data.normalize_features,
                'augment_data': self.data.augment_data,
                'use_profile_features': self.data.use_profile_features,
                'use_activity_features': self.data.use_activity_features,
                'use_network_features': self.data.use_network_features,
                'use_interaction_features': self.data.use_interaction_features,
                'val_split': self.data.val_split,
                'test_split': self.data.test_split,
                'random_seed': self.data.random_seed
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        data_config = DataConfig(**config_dict['data'])
        return cls(model=model_config, training=training_config, data=data_config) 