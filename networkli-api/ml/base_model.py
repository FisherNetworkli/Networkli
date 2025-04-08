import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .model_utils import save_model, load_model

class BaseModel(nn.Module):
    """Base class for all models in the project."""
    
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
    
    def save(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save the model with optional metadata.
        
        Args:
            metadata: Optional dictionary of metadata to save with the model
            
        Returns:
            str: Path to the saved model file
        """
        return save_model(self, self.model_name, metadata)
    
    def load(self, model_path: str) -> Dict[str, Any]:
        """
        Load model weights from a file.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Dict containing the loaded metadata
        """
        return load_model(self, model_path)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get the number of trainable and total parameters in the model.
        
        Returns:
            Dict containing counts of trainable and total parameters
        """
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable_parameters': trainable_params,
            'total_parameters': total_params
        }
    
    def training_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            batch: The input batch of data
            
        Returns:
            Dict containing the loss and any other metrics
        """
        raise NotImplementedError(
            "Subclasses must implement training_step"
        )
    
    def validation_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.
        
        Args:
            batch: The input batch of data
            
        Returns:
            Dict containing the loss and any other metrics
        """
        raise NotImplementedError(
            "Subclasses must implement validation_step"
        )
    
    def predict_step(self, batch: Any) -> Any:
        """
        Perform a single prediction step.
        
        Args:
            batch: The input batch of data
            
        Returns:
            Model predictions
        """
        raise NotImplementedError(
            "Subclasses must implement predict_step"
        ) 