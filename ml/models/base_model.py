from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

class BaseModel(ABC):
    """Base class for all ML models in the Networkli platform."""
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> None:
        """Train the model on the provided data."""
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Evaluate the model's performance."""
        pass
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update the model's metadata."""
        self.metadata.update(metadata)
        self.last_updated = datetime.now()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get the model's metadata."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            **self.metadata
        } 