from typing import Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from datetime import datetime
import logging

from ..models.networkli_gnn import NetworkliGNN
from ..models.maml import MAML
from ..pipeline.feature_engineering import FeatureEngineer

class Recommender:
    """Recommendation system for user matching, events, and groups."""
    
    def __init__(self,
                 feature_engineer: FeatureEngineer,
                 user_matcher: Optional[NetworkliGNN] = None,
                 event_matcher: Optional[NetworkliGNN] = None,
                 group_matcher: Optional[NetworkliGNN] = None,
                 maml: Optional[MAML] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.feature_engineer = feature_engineer
        self.user_matcher = user_matcher
        self.event_matcher = event_matcher
        self.group_matcher = group_matcher
        self.maml = maml
        self.device = device
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def recommend_users(self,
                       user_id: str,
                       user_data: Dict[str, Union[str, List[str], float]],
                       candidate_users: List[Dict[str, Union[str, List[str], float]]],
                       top_k: int = 10,
                       filters: Optional[Dict[str, Union[str, List[str], bool]]] = None) -> List[Dict[str, Union[str, float, List[str]]]]:
        """
        Recommend users to connect with.
        
        Args:
            user_id: ID of the user to get recommendations for
            user_data: Dictionary containing user profile data
            candidate_users: List of candidate users to match against
            top_k: Number of recommendations to return
            filters: Optional filters to apply (industry, location, etc.)
            
        Returns:
            List of recommended users with match scores
        """
        if self.user_matcher is None:
            self.logger.error("User matcher model not initialized")
            return []
        
        # Process user data
        user_features = self.feature_engineer.process_user_data(user_data)
        
        # Process candidate users
        candidate_features = []
        for candidate in candidate_users:
            features = self.feature_engineer.process_user_data(candidate)
            candidate_features.append(features)
        
        # Create graph data
        num_candidates = len(candidate_users)
        x = torch.cat([user_features['features'].unsqueeze(0)] + 
                     [f['features'].unsqueeze(0) for f in candidate_features], dim=0)
        
        # Create edge index for all pairs
        edge_index = []
        for i in range(num_candidates + 1):
            for j in range(i + 1, num_candidates + 1):
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Get predictions
        self.user_matcher.eval()
        with torch.no_grad():
            _, compatibility = self.user_matcher(x, edge_index)
        
        # Extract compatibility scores for the user
        user_compatibility = compatibility[edge_index[0] == 0].squeeze()
        
        # Apply filters if provided
        if filters:
            filtered_indices = []
            for i, candidate in enumerate(candidate_users):
                # Check industry match
                if 'industry' in filters and candidate.get('industry') != filters['industry']:
                    continue
                
                # Check location match
                if 'location' in filters and candidate.get('location') != filters['location']:
                    continue
                
                # Check remote preference
                if filters.get('remote') and not candidate.get('remote_available', False):
                    continue
                
                filtered_indices.append(i)
            
            if not filtered_indices:
                return []
            
            # Get top k from filtered candidates
            filtered_compatibility = user_compatibility[filtered_indices]
            top_k_indices = torch.argsort(filtered_compatibility, descending=True)[:top_k]
            top_k_candidates = [candidate_users[filtered_indices[i]] for i in top_k_indices]
            top_k_scores = filtered_compatibility[top_k_indices].tolist()
        else:
            # Get top k candidates
            top_k_indices = torch.argsort(user_compatibility, descending=True)[:top_k]
            top_k_candidates = [candidate_users[i] for i in top_k_indices]
            top_k_scores = user_compatibility[top_k_indices].tolist()
        
        # Format recommendations
        recommendations = []
        for candidate, score in zip(top_k_candidates, top_k_scores):
            # Calculate skill match
            user_skills = set(user_data.get('skills', []))
            candidate_skills = set(candidate.get('skills', []))
            skill_match = len(user_skills & candidate_skills) / max(len(user_skills), 1)
            
            recommendations.append({
                'user_id': candidate.get('id', ''),
                'name': candidate.get('name', ''),
                'title': candidate.get('title', ''),
                'company': candidate.get('company', ''),
                'match_score': float(score),
                'skill_match': float(skill_match),
                'common_skills': list(user_skills & candidate_skills)
            })
        
        return recommendations
    
    def recommend_events(self,
                        user_id: str,
                        user_data: Dict[str, Union[str, List[str], float]],
                        events: List[Dict[str, Union[str, List[str], float, datetime]]],
                        top_k: int = 5,
                        filters: Optional[Dict[str, Union[str, List[str], bool]]] = None) -> List[Dict[str, Union[str, float, datetime]]]:
        """
        Recommend events to attend.
        
        Args:
            user_id: ID of the user to get recommendations for
            user_data: Dictionary containing user profile data
            events: List of available events
            top_k: Number of recommendations to return
            filters: Optional filters to apply (industry, location, etc.)
            
        Returns:
            List of recommended events with relevance scores
        """
        if self.event_matcher is None:
            self.logger.error("Event matcher model not initialized")
            return []
        
        # Process user data
        user_features = self.feature_engineer.process_user_data(user_data)
        
        # Process events
        event_features = []
        for event in events:
            features = self.feature_engineer.process_job_data(event)
            event_features.append(features)
        
        # Create graph data
        num_events = len(events)
        x = torch.cat([user_features['features'].unsqueeze(0)] + 
                     [f['features'].unsqueeze(0) for f in event_features], dim=0)
        
        # Create edge index for all pairs
        edge_index = []
        for i in range(num_events + 1):
            for j in range(i + 1, num_events + 1):
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Get predictions
        self.event_matcher.eval()
        with torch.no_grad():
            _, compatibility = self.event_matcher(x, edge_index)
        
        # Extract compatibility scores for the user
        user_compatibility = compatibility[edge_index[0] == 0].squeeze()
        
        # Apply filters if provided
        if filters:
            filtered_indices = []
            for i, event in enumerate(events):
                # Check industry match
                if 'industry' in filters and event.get('industry') != filters['industry']:
                    continue
                
                # Check location match
                if 'location' in filters and event.get('location') != filters['location']:
                    continue
                
                # Check format preference
                if 'format' in filters and event.get('format') != filters['format']:
                    continue
                
                # Check date range
                if 'start_date' in filters and event.get('start_date') < filters['start_date']:
                    continue
                if 'end_date' in filters and event.get('end_date') > filters['end_date']:
                    continue
                
                filtered_indices.append(i)
            
            if not filtered_indices:
                return []
            
            # Get top k from filtered events
            filtered_compatibility = user_compatibility[filtered_indices]
            top_k_indices = torch.argsort(filtered_compatibility, descending=True)[:top_k]
            top_k_events = [events[filtered_indices[i]] for i in top_k_indices]
            top_k_scores = filtered_compatibility[top_k_indices].tolist()
        else:
            # Get top k events
            top_k_indices = torch.argsort(user_compatibility, descending=True)[:top_k]
            top_k_events = [events[i] for i in top_k_indices]
            top_k_scores = user_compatibility[top_k_indices].tolist()
        
        # Format recommendations
        recommendations = []
        for event, score in zip(top_k_events, top_k_scores):
            # Calculate skill relevance
            user_skills = set(user_data.get('skills', []))
            event_skills = set(event.get('required_skills', []))
            skill_relevance = len(user_skills & event_skills) / max(len(event_skills), 1)
            
            recommendations.append({
                'event_id': event.get('id', ''),
                'title': event.get('title', ''),
                'description': event.get('description', ''),
                'start_date': event.get('start_date'),
                'end_date': event.get('end_date'),
                'location': event.get('location', ''),
                'format': event.get('format', ''),
                'relevance_score': float(score),
                'skill_relevance': float(skill_relevance)
            })
        
        return recommendations
    
    def recommend_groups(self,
                        user_id: str,
                        user_data: Dict[str, Union[str, List[str], float]],
                        groups: List[Dict[str, Union[str, List[str], float, int]]],
                        top_k: int = 5,
                        filters: Optional[Dict[str, Union[str, List[str], bool]]] = None) -> List[Dict[str, Union[str, float, int]]]:
        """
        Recommend groups to join.
        
        Args:
            user_id: ID of the user to get recommendations for
            user_data: Dictionary containing user profile data
            groups: List of available groups
            top_k: Number of recommendations to return
            filters: Optional filters to apply (industry, focus area, etc.)
            
        Returns:
            List of recommended groups with relevance scores
        """
        if self.group_matcher is None:
            self.logger.error("Group matcher model not initialized")
            return []
        
        # Process user data
        user_features = self.feature_engineer.process_user_data(user_data)
        
        # Process groups
        group_features = []
        for group in groups:
            features = self.feature_engineer.process_job_data(group)
            group_features.append(features)
        
        # Create graph data
        num_groups = len(groups)
        x = torch.cat([user_features['features'].unsqueeze(0)] + 
                     [f['features'].unsqueeze(0) for f in group_features], dim=0)
        
        # Create edge index for all pairs
        edge_index = []
        for i in range(num_groups + 1):
            for j in range(i + 1, num_groups + 1):
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Get predictions
        self.group_matcher.eval()
        with torch.no_grad():
            _, compatibility = self.group_matcher(x, edge_index)
        
        # Extract compatibility scores for the user
        user_compatibility = compatibility[edge_index[0] == 0].squeeze()
        
        # Apply filters if provided
        if filters:
            filtered_indices = []
            for i, group in enumerate(groups):
                # Check industry match
                if 'industry' in filters and group.get('industry') != filters['industry']:
                    continue
                
                # Check focus area match
                if 'focus_area' in filters and group.get('focus_area') != filters['focus_area']:
                    continue
                
                # Check size constraints
                if 'min_size' in filters and group.get('member_count', 0) < filters['min_size']:
                    continue
                if 'max_size' in filters and group.get('member_count', 0) > filters['max_size']:
                    continue
                
                filtered_indices.append(i)
            
            if not filtered_indices:
                return []
            
            # Get top k from filtered groups
            filtered_compatibility = user_compatibility[filtered_indices]
            top_k_indices = torch.argsort(filtered_compatibility, descending=True)[:top_k]
            top_k_groups = [groups[filtered_indices[i]] for i in top_k_indices]
            top_k_scores = filtered_compatibility[top_k_indices].tolist()
        else:
            # Get top k groups
            top_k_indices = torch.argsort(user_compatibility, descending=True)[:top_k]
            top_k_groups = [groups[i] for i in top_k_indices]
            top_k_scores = user_compatibility[top_k_indices].tolist()
        
        # Format recommendations
        recommendations = []
        for group, score in zip(top_k_groups, top_k_scores):
            # Calculate skill development opportunity
            user_skills = set(user_data.get('skills', []))
            group_skills = set(group.get('focus_skills', []))
            skill_opportunity = len(group_skills - user_skills) / max(len(group_skills), 1)
            
            # Calculate network value
            network_value = min(group.get('member_count', 0) / 100, 1.0)  # Normalize to 0-1
            
            recommendations.append({
                'group_id': group.get('id', ''),
                'name': group.get('name', ''),
                'description': group.get('description', ''),
                'industry': group.get('industry', ''),
                'focus_area': group.get('focus_area', ''),
                'member_count': group.get('member_count', 0),
                'relevance_score': float(score),
                'skill_opportunity': float(skill_opportunity),
                'network_value': float(network_value)
            })
        
        return recommendations
    
    def get_personalized_recommendations(self,
                                        user_id: str,
                                        user_data: Dict[str, Union[str, List[str], float]],
                                        candidates: Dict[str, List[Dict[str, Union[str, List[str], float, datetime, int]]]],
                                        top_k: Dict[str, int] = {'users': 10, 'events': 5, 'groups': 5},
                                        filters: Optional[Dict[str, Dict[str, Union[str, List[str], bool, datetime, int]]]] = None) -> Dict[str, List[Dict[str, Union[str, float, List[str], datetime, int]]]]:
        """
        Get personalized recommendations for users, events, and groups.
        
        Args:
            user_id: ID of the user to get recommendations for
            user_data: Dictionary containing user profile data
            candidates: Dictionary containing candidate users, events, and groups
            top_k: Dictionary specifying number of recommendations for each type
            filters: Optional filters for each recommendation type
            
        Returns:
            Dictionary containing recommendations for each type
        """
        recommendations = {}
        
        # Get user recommendations
        if 'users' in candidates and self.user_matcher is not None:
            user_filters = filters.get('users') if filters else None
            recommendations['users'] = self.recommend_users(
                user_id, user_data, candidates['users'], 
                top_k.get('users', 10), user_filters
            )
        
        # Get event recommendations
        if 'events' in candidates and self.event_matcher is not None:
            event_filters = filters.get('events') if filters else None
            recommendations['events'] = self.recommend_events(
                user_id, user_data, candidates['events'], 
                top_k.get('events', 5), event_filters
            )
        
        # Get group recommendations
        if 'groups' in candidates and self.group_matcher is not None:
            group_filters = filters.get('groups') if filters else None
            recommendations['groups'] = self.recommend_groups(
                user_id, user_data, candidates['groups'], 
                top_k.get('groups', 5), group_filters
            )
        
        return recommendations
    
    def adapt_to_user_preferences(self,
                                 user_id: str,
                                 user_data: Dict[str, Union[str, List[str], float]],
                                 feedback_data: List[Dict[str, Union[str, float, bool]]]) -> None:
        """
        Adapt recommendations to user preferences using MAML.
        
        Args:
            user_id: ID of the user
            user_data: Dictionary containing user profile data
            feedback_data: List of feedback items with ratings or interactions
        """
        if self.maml is None:
            self.logger.error("MAML model not initialized")
            return
        
        # Process user data
        user_features = self.feature_engineer.process_user_data(user_data)
        
        # Process feedback data
        support_data = {
            'x': user_features['features'].unsqueeze(0),
            'y': torch.tensor([item.get('rating', 0.0) for item in feedback_data], dtype=torch.float)
        }
        
        # Adapt model to user preferences
        adapted_model = self.maml.adapt(support_data)
        
        # Update the appropriate matcher based on feedback type
        feedback_type = feedback_data[0].get('type', '') if feedback_data else ''
        
        if feedback_type == 'user':
            self.user_matcher = adapted_model
        elif feedback_type == 'event':
            self.event_matcher = adapted_model
        elif feedback_type == 'group':
            self.group_matcher = adapted_model
        
        self.logger.info(f"Adapted {feedback_type} matcher to user {user_id} preferences") 