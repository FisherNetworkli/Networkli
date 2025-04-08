import os
import torch
from typing import Any, Dict
from datetime import datetime

def save_model(model: torch.nn.Module, model_name: str, metadata: Dict[str, Any] = None) -> str:
    """
    Save a PyTorch model to the models directory with metadata.
    
    Args:
        model: The PyTorch model to save
        model_name: Name of the model
        metadata: Optional dictionary containing model metadata
        
    Returns:
        str: Path to the saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp for unique model versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}.pt"
    save_path = os.path.join('models', filename)
    
    # Prepare save dictionary with model state and metadata
    save_dict = {
        'model_state': model.state_dict(),
        'model_name': model_name,
        'saved_at': timestamp
    }
    
    if metadata:
        save_dict['metadata'] = metadata
        
    # Save the model
    torch.save(save_dict, save_path)
    return save_path

def load_model(model: torch.nn.Module, model_path: str) -> Dict[str, Any]:
    """
    Load a saved PyTorch model and its metadata.
    
    Args:
        model: The PyTorch model instance to load weights into
        model_path: Path to the saved model file
        
    Returns:
        Dict containing the loaded metadata
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the saved dictionary
    checkpoint = torch.load(model_path)
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state'])
    
    # Return metadata
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state'}
    return metadata

def list_saved_models() -> list:
    """
    List all saved models in the models directory.
    
    Returns:
        List of model filenames
    """
    if not os.path.exists('models'):
        return []
    
    return [f for f in os.listdir('models') if f.endswith('.pt')] 