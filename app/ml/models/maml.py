import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from .networkli_gnn import NetworkliGNN

class MAML:
    def __init__(
        self,
        model: NetworkliGNN,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_adaptation_steps: int = 1,
        similarity_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_adaptation_steps = num_adaptation_steps
        self.optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        self.similarity_weights = similarity_weights or {
            'bio_similarity': 0.2,
            'expertise_needs_match': 0.3,
            'needs_expertise_match': 0.3,
            'goals_alignment': 0.2
        }
        
    def adapt(
        self,
        support_data: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt the model to a new task using support data.
        
        Args:
            support_data: Dictionary containing:
                - x: Node features
                - edge_index: Graph connectivity
                - domain_ids: Domain IDs
                - skill_ids_1: First set of skill IDs
                - skill_ids_2: Second set of skill IDs
                - bio_embeddings: BERT embeddings of bios
                - expertise_embeddings: BERT embeddings of expertise
                - needs_embeddings: BERT embeddings of needs
                - goals_embeddings: BERT embeddings of goals
                - edge_attr: Edge features
                - labels: Ground truth labels
            num_steps: Number of adaptation steps
            
        Returns:
            Dictionary of adapted model parameters
        """
        num_steps = num_steps or self.num_adaptation_steps
        adapted_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for _ in range(num_steps):
            # Forward pass on support data
            embeddings, compatibility, similarities = self.model(
                support_data['x'],
                support_data['edge_index'],
                support_data['domain_ids'],
                support_data['skill_ids_1'],
                support_data['skill_ids_2'],
                support_data['bio_embeddings'],
                support_data['expertise_embeddings'],
                support_data['needs_embeddings'],
                support_data['goals_embeddings'],
                support_data.get('edge_attr')
            )
            
            # Compute match scores
            match_scores = self.model.compute_match_score(
                similarities,
                weights=self.similarity_weights
            )
            
            # Compute loss
            loss = self._compute_loss(
                embeddings=embeddings,
                compatibility_scores=compatibility,
                similarity_scores=similarities,
                match_scores=match_scores,
                labels=support_data['labels']
            )
            
            # Compute gradients
            grads = torch.autograd.grad(loss, self.model.parameters())
            
            # Update parameters
            for param, grad in zip(adapted_params.values(), grads):
                param.sub_(self.inner_lr * grad)
                
        return adapted_params
    
    def meta_update(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        num_adaptation_steps: Optional[int] = None
    ) -> float:
        """
        Perform meta-update using multiple tasks.
        
        Args:
            tasks: List of task data dictionaries
            num_adaptation_steps: Number of adaptation steps per task
            
        Returns:
            Average meta-loss across tasks
        """
        meta_loss = 0.0
        
        for task in tasks:
            # Adapt to the task
            adapted_params = self.adapt(task['support'], num_adaptation_steps)
            
            # Save original parameters
            original_params = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            # Load adapted parameters
            self.model.load_state_dict(adapted_params)
            
            # Compute loss on query set
            outputs = self.model(**task['query'])
            task_loss = self._compute_loss(outputs, task['query'])
            meta_loss += task_loss
            
            # Restore original parameters
            self.model.load_state_dict(original_params)
            
        # Average meta-loss
        meta_loss /= len(tasks)
        
        # Meta-optimization step
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        return meta_loss.item()
    
    def _compute_loss(
        self,
        embeddings: torch.Tensor,
        compatibility_scores: torch.Tensor,
        similarity_scores: Dict[str, torch.Tensor],
        match_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the combined loss for meta-learning.
        
        Args:
            embeddings: Node embeddings
            compatibility_scores: Skill compatibility scores
            similarity_scores: Dictionary of similarity scores
            match_scores: Final match scores
            labels: Ground truth labels
            
        Returns:
            Combined loss value
        """
        # Embedding loss (contrastive)
        embedding_loss = self._contrastive_loss(embeddings, labels)
        
        # Compatibility loss
        compatibility_loss = F.binary_cross_entropy(
            compatibility_scores,
            labels.float()
        )
        
        # Similarity losses
        similarity_loss = sum(
            self.similarity_weights[key] * F.binary_cross_entropy(
                torch.sigmoid(sim),
                labels.float()
            )
            for key, sim in similarity_scores.items()
        )
        
        # Match score loss
        match_loss = F.binary_cross_entropy(
            match_scores,
            labels.float()
        )
        
        # Combine losses
        total_loss = (
            0.3 * embedding_loss +
            0.2 * compatibility_loss +
            0.3 * similarity_loss +
            0.2 * match_loss
        )
        
        return total_loss
    
    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute contrastive loss for embeddings.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            labels: Node labels [num_nodes]
            
        Returns:
            Contrastive loss value
        """
        if labels is None:
            return torch.tensor(0.0, device=embeddings.device)
            
        # Compute pairwise distances
        dist = torch.cdist(embeddings, embeddings)
        
        # Create positive/negative masks
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        neg_mask = 1 - pos_mask
        
        # Compute loss
        pos_loss = (dist * pos_mask).sum() / (pos_mask.sum() + 1e-6)
        neg_loss = F.relu(1 - dist) * neg_mask
        neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-6)
        
        return pos_loss + neg_loss 