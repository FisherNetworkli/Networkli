import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import networkx as nx

class FeatureEngineer:
    def __init__(
        self,
        skill_dim: int = 100,
        interest_dim: int = 50,
        max_skills: int = 20,
        max_interests: int = 10
    ):
        self.skill_dim = skill_dim
        self.interest_dim = interest_dim
        self.max_skills = max_skills
        self.max_interests = max_interests
        
        # Initialize vectorizers
        self.skill_vectorizer = TfidfVectorizer(
            max_features=skill_dim,
            stop_words='english'
        )
        self.interest_vectorizer = TfidfVectorizer(
            max_features=interest_dim,
            stop_words='english'
        )
        
        # Initialize scalers
        self.profile_scaler = StandardScaler()
        self.activity_scaler = StandardScaler()
        
    def process_profile(
        self,
        profile_data: Dict[str, any]
    ) -> torch.Tensor:
        """
        Process user profile data into features.
        
        Args:
            profile_data: Dictionary containing profile information
            
        Returns:
            Profile features tensor
        """
        # Extract text features
        skills_text = ' '.join(profile_data.get('skills', []))
        interests_text = ' '.join(profile_data.get('interests', []))
        
        # Transform text to vectors
        skills_vec = self.skill_vectorizer.transform([skills_text]).toarray()
        interests_vec = self.interest_vectorizer.transform([interests_text]).toarray()
        
        # Extract numerical features
        numerical_features = [
            profile_data.get('experience_years', 0),
            len(profile_data.get('connections', [])),
            len(profile_data.get('events_attended', [])),
            profile_data.get('response_rate', 0),
            profile_data.get('profile_completeness', 0)
        ]
        
        # Combine features
        features = np.concatenate([
            skills_vec.flatten(),
            interests_vec.flatten(),
            numerical_features
        ])
        
        # Scale features
        features = self.profile_scaler.fit_transform(features.reshape(1, -1))
        
        return torch.FloatTensor(features)
    
    def process_activity(
        self,
        activity_data: Dict[str, any]
    ) -> torch.Tensor:
        """
        Process user activity data into features.
        
        Args:
            activity_data: Dictionary containing activity information
            
        Returns:
            Activity features tensor
        """
        # Extract activity metrics
        activity_features = [
            activity_data.get('login_frequency', 0),
            activity_data.get('message_frequency', 0),
            activity_data.get('profile_view_frequency', 0),
            activity_data.get('event_attendance_rate', 0),
            activity_data.get('content_creation_rate', 0),
            activity_data.get('connection_growth_rate', 0)
        ]
        
        # Scale features
        features = self.activity_scaler.fit_transform(
            np.array(activity_features).reshape(1, -1)
        )
        
        return torch.FloatTensor(features)
    
    def process_network(
        self,
        network_data: Dict[str, any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process network data into graph structure.
        
        Args:
            network_data: Dictionary containing network information
            
        Returns:
            Tuple of (node features, edge index)
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for user in network_data['users']:
            G.add_node(
                user['id'],
                features=self.process_profile(user['profile'])
            )
            
        # Add edges
        for connection in network_data['connections']:
            G.add_edge(
                connection['user1_id'],
                connection['user2_id'],
                weight=connection['interaction_strength']
            )
            
        # Convert to PyTorch format
        edge_index = torch.LongTensor(list(G.edges())).t()
        node_features = torch.stack([
            G.nodes[node]['features']
            for node in G.nodes()
        ])
        
        return node_features, edge_index
    
    def process_interaction(
        self,
        interaction_data: Dict[str, any]
    ) -> torch.Tensor:
        """
        Process interaction data into features.
        
        Args:
            interaction_data: Dictionary containing interaction information
            
        Returns:
            Interaction features tensor
        """
        # Extract interaction metrics
        interaction_features = [
            interaction_data.get('message_count', 0),
            interaction_data.get('profile_views', 0),
            interaction_data.get('event_interactions', 0),
            interaction_data.get('content_interactions', 0),
            interaction_data.get('response_time', 0),
            interaction_data.get('interaction_frequency', 0)
        ]
        
        # Scale features
        features = self.activity_scaler.fit_transform(
            np.array(interaction_features).reshape(1, -1)
        )
        
        return torch.FloatTensor(features)
    
    def process_batch(
        self,
        batch_data: List[Dict[str, any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of data.
        
        Args:
            batch_data: List of data dictionaries
            
        Returns:
            Dictionary of processed tensors
        """
        profiles = []
        activities = []
        interactions = []
        
        for data in batch_data:
            profiles.append(self.process_profile(data['profile']))
            activities.append(self.process_activity(data['activity']))
            interactions.append(self.process_interaction(data['interaction']))
            
        return {
            'profiles': torch.stack(profiles),
            'activities': torch.stack(activities),
            'interactions': torch.stack(interactions)
        } 