import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import List, Tuple, Optional, Dict

class MultiScaleGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout

        # Multi-scale GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(
                in_channels=input_dim if i == 0 else hidden_dims[i-1],
                out_channels=hidden_dims[i],
                dropout=dropout
            ) for i in range(len(hidden_dims))
        ])

        # Attention layers
        self.attention_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dims[i],
                out_channels=hidden_dims[i] // num_heads,
                heads=num_heads,
                dropout=dropout
            ) for i in range(len(hidden_dims))
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dims[i])
            for i in range(len(hidden_dims))
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the multi-scale GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Node embeddings [num_nodes, hidden_dims[-1]]
        """
        for i in range(len(self.hidden_dims)):
            # GCN layer
            x_gcn = self.gcn_layers[i](x, edge_index)
            
            # Attention layer
            x_att = self.attention_layers[i](x_gcn, edge_index)
            
            # Layer normalization and residual connection
            x = self.layer_norms[i](x_att + x_gcn)
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return x

class ProfessionalDomainEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_domains: int
    ):
        super().__init__()
        self.domain_embeddings = nn.Embedding(num_domains, embedding_dim)
        self.domain_projection = nn.Linear(input_dim, embedding_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Project features into domain-specific embedding space.
        
        Args:
            x: Input features [batch_size, input_dim]
            domain_ids: Domain IDs [batch_size]
            
        Returns:
            Domain-aware embeddings [batch_size, embedding_dim]
        """
        # Get domain embeddings
        domain_emb = self.domain_embeddings(domain_ids)
        
        # Project input features
        x_proj = self.domain_projection(x)
        
        # Combine with domain embeddings
        return x_proj + domain_emb

class SkillCompatibilityScoring(nn.Module):
    def __init__(
        self,
        skill_dim: int,
        hidden_dim: int,
        num_skills: int
    ):
        super().__init__()
        self.skill_embeddings = nn.Embedding(num_skills, skill_dim)
        self.compatibility_net = nn.Sequential(
            nn.Linear(skill_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        skill_ids_1: torch.Tensor,
        skill_ids_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute compatibility score between two sets of skills.
        
        Args:
            skill_ids_1: First set of skill IDs [batch_size, num_skills_1]
            skill_ids_2: Second set of skill IDs [batch_size, num_skills_2]
            
        Returns:
            Compatibility scores [batch_size, 1]
        """
        # Get skill embeddings
        skills_1 = self.skill_embeddings(skill_ids_1)
        skills_2 = self.skill_embeddings(skill_ids_2)
        
        # Compute average embeddings
        skills_1_avg = skills_1.mean(dim=1)
        skills_2_avg = skills_2.mean(dim=1)
        
        # Concatenate and compute compatibility
        combined = torch.cat([skills_1_avg, skills_2_avg], dim=1)
        return self.compatibility_net(combined)

class NetworkliGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        skill_dim: int,
        num_skills: int,
        num_domains: int,
        text_embedding_dim: int = 768,  # BERT embedding dimension
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Multi-scale GNN for graph structure
        self.gnn = MultiScaleGNN(
            input_dim=input_dim + text_embedding_dim * 4,  # Add dimensions for text embeddings
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Professional domain embeddings
        self.domain_embedding = ProfessionalDomainEmbedding(
            input_dim=hidden_dims[-1],
            embedding_dim=hidden_dims[-1],
            num_domains=num_domains
        )
        
        # Skill compatibility scoring
        self.skill_compatibility = SkillCompatibilityScoring(
            skill_dim=skill_dim,
            hidden_dim=hidden_dims[-1],
            num_skills=num_skills
        )

        # Text feature projections
        self.bio_projection = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.expertise_projection = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.needs_projection = nn.Linear(text_embedding_dim, text_embedding_dim)
        self.goals_projection = nn.Linear(text_embedding_dim, text_embedding_dim)
        
        # Attention for combining different aspects
        self.profile_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        domain_ids: torch.Tensor,
        skill_ids_1: torch.Tensor,
        skill_ids_2: torch.Tensor,
        bio_embeddings: torch.Tensor,
        expertise_embeddings: torch.Tensor,
        needs_embeddings: torch.Tensor,
        goals_embeddings: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the complete NetworkliGNN model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            domain_ids: Domain IDs [num_nodes]
            skill_ids_1: First set of skill IDs [batch_size, num_skills_1]
            skill_ids_2: Second set of skill IDs [batch_size, num_skills_2]
            bio_embeddings: BERT embeddings of user bios [num_nodes, text_embedding_dim]
            expertise_embeddings: BERT embeddings of expertise [num_nodes, text_embedding_dim]
            needs_embeddings: BERT embeddings of needs [num_nodes, text_embedding_dim]
            goals_embeddings: BERT embeddings of meaningful goals [num_nodes, text_embedding_dim]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Tuple of:
            - Node embeddings [num_nodes, hidden_dims[-1]]
            - Skill compatibility scores [batch_size, 1]
            - Similarity scores for different aspects
        """
        # Project text embeddings
        bio_features = self.bio_projection(bio_embeddings)
        expertise_features = self.expertise_projection(expertise_embeddings)
        needs_features = self.needs_projection(needs_embeddings)
        goals_features = self.goals_projection(goals_embeddings)
        
        # Concatenate all features
        combined_features = torch.cat([
            x,
            bio_features,
            expertise_features,
            needs_features,
            goals_features
        ], dim=1)
        
        # Get node embeddings from GNN
        node_embeddings = self.gnn(combined_features, edge_index, edge_attr)
        
        # Add domain-specific embeddings
        domain_embeddings = self.domain_embedding(node_embeddings, domain_ids)
        
        # Compute skill compatibility
        compatibility_scores = self.skill_compatibility(skill_ids_1, skill_ids_2)
        
        # Calculate similarity scores for different aspects
        similarity_scores = {}
        
        # Bio similarity
        bio_sim = F.cosine_similarity(
            bio_features[edge_index[0]],
            bio_features[edge_index[1]]
        )
        
        # Expertise-Needs matching (bidirectional)
        expertise_needs_sim = F.cosine_similarity(
            expertise_features[edge_index[0]],
            needs_features[edge_index[1]]
        )
        needs_expertise_sim = F.cosine_similarity(
            needs_features[edge_index[0]],
            expertise_features[edge_index[1]]
        )
        
        # Goals alignment
        goals_sim = F.cosine_similarity(
            goals_features[edge_index[0]],
            goals_features[edge_index[1]]
        )
        
        similarity_scores.update({
            'bio_similarity': bio_sim,
            'expertise_needs_match': expertise_needs_sim,
            'needs_expertise_match': needs_expertise_sim,
            'goals_alignment': goals_sim
        })
        
        return domain_embeddings, compatibility_scores, similarity_scores
        
    def compute_match_score(
        self,
        similarity_scores: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute the final match score using weighted similarities.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            weights: Optional dictionary of weights for each score type
            
        Returns:
            Final match scores [num_edges]
        """
        if weights is None:
            weights = {
                'bio_similarity': 0.2,
                'expertise_needs_match': 0.3,
                'needs_expertise_match': 0.3,
                'goals_alignment': 0.2
            }
            
        final_score = sum(
            weights[key] * similarity_scores[key]
            for key in weights
        )
        
        return torch.sigmoid(final_score) 