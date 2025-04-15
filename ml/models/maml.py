from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base_model import BaseModel

class MAML(BaseModel):
    """Model-Agnostic Meta-Learning for few-shot user matching."""
    
    def __init__(self,
                 base_model: nn.Module,
                 inner_lr: float = 0.01,
                 meta_lr: float = 0.001,
                 num_adaptation_steps: int = 1):
        super().__init__(model_name="maml", version="1.0.0")
        
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_adaptation_steps = num_adaptation_steps
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=meta_lr)
        
    def adapt(self, 
              support_data: Dict[str, torch.Tensor],
              num_steps: Optional[int] = None) -> nn.Module:
        """
        Adapt the model to a new task using support data.
        
        Args:
            support_data: Dictionary containing support set data
            num_steps: Number of adaptation steps (defaults to self.num_adaptation_steps)
            
        Returns:
            Adapted model
        """
        adapted_model = deepcopy(self.base_model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        num_steps = num_steps or self.num_adaptation_steps
        
        # Adaptation loop
        for _ in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = adapted_model(**support_data)
            
            # Calculate loss
            loss = self._compute_loss(outputs, support_data)
            
            # Update model
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def train(self, 
              tasks: List[Dict[str, Dict[str, torch.Tensor]]],
              **kwargs) -> None:
        """
        Meta-train the model on multiple tasks.
        
        Args:
            tasks: List of task dictionaries, each containing:
                - support: Support set data
                - query: Query set data
        """
        self.base_model.train()
        
        for epoch in range(kwargs.get('epochs', 100)):
            self.meta_optimizer.zero_grad()
            
            meta_loss = 0.0
            for task in tasks:
                # Adapt to the task
                adapted_model = self.adapt(task['support'])
                
                # Evaluate on query set
                with torch.no_grad():
                    query_outputs = adapted_model(**task['query'])
                    query_loss = self._compute_loss(query_outputs, task['query'])
                
                meta_loss += query_loss
            
            # Average meta-loss
            meta_loss /= len(tasks)
            
            # Meta-update
            meta_loss.backward()
            self.meta_optimizer.step()
            
            # Early stopping check
            if kwargs.get('early_stopping'):
                if self._check_early_stopping(meta_loss.item()):
                    break
    
    def predict(self,
                support_data: Dict[str, torch.Tensor],
                query_data: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Make predictions on new tasks.
        
        Args:
            support_data: Support set data for adaptation
            query_data: Query set data for prediction
            
        Returns:
            Dictionary containing predictions
        """
        # Adapt to the task
        adapted_model = self.adapt(support_data)
        
        # Make predictions
        adapted_model.eval()
        with torch.no_grad():
            predictions = adapted_model(**query_data)
        
        return predictions
    
    def evaluate(self,
                 tasks: List[Dict[str, Dict[str, torch.Tensor]]],
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance on multiple tasks.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            Dictionary containing evaluation metrics
        """
        total_accuracy = 0.0
        total_tasks = len(tasks)
        
        for task in tasks:
            # Adapt and predict
            predictions = self.predict(task['support'], task['query'])
            
            # Calculate accuracy
            accuracy = self._compute_accuracy(predictions, task['query'])
            total_accuracy += accuracy
        
        return {
            'average_accuracy': total_accuracy / total_tasks
        }
    
    def _compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the loss for the current task."""
        # This should be implemented based on the specific task
        raise NotImplementedError
    
    def _compute_accuracy(self,
                         predictions: Dict[str, torch.Tensor],
                         data: Dict[str, torch.Tensor]) -> float:
        """Compute the accuracy for the current task."""
        # This should be implemented based on the specific task
        raise NotImplementedError
    
    def _check_early_stopping(self, loss: float) -> bool:
        """Check if training should be stopped early."""
        if not hasattr(self, '_best_loss'):
            self._best_loss = float('inf')
            self._patience = 5
            self._counter = 0
            return False
        
        if loss < self._best_loss:
            self._best_loss = loss
            self._counter = 0
            return False
        
        self._counter += 1
        return self._counter >= self._patience 