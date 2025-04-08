"""Service for generating predictions and recommendations."""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import logging
from .ml.models import NetworkliGNN

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model: NetworkliGNN = None):
        """Initialize the prediction service."""
        self.model = model
        self._cache = {}
        self._cache_ttl = timedelta(hours=1)
    
    def _get_user_vector(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Convert user data into a feature vector."""
        # Combine skills and interests
        features = set(user_data.get('skills', []) + user_data.get('interests', []))
        
        # Add professional features
        if user_data.get('experience'):
            features.update(exp['title'] for exp in user_data['experience'])
            features.update(exp['company'] for exp in user_data['experience'])
        
        if user_data.get('education'):
            features.update(edu['field'] for edu in user_data['education'])
            features.update(edu['school'] for edu in user_data['education'])
        
        return np.array(list(features))
    
    def predict_matches(self, 
                       user_id: str,
                       user_data: Dict[str, Any],
                       potential_matches: List[Dict[str, Any]],
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Predict the best professional matches for a user.
        
        Args:
            user_id: ID of the user
            user_data: Data about the user
            potential_matches: List of potential matches
            top_k: Number of matches to return
            
        Returns:
            List of recommended matches with scores
        """
        cache_key = f"matches_{user_id}"
        if cache_key in self._cache:
            cache_time, cached_results = self._cache[cache_key]
            if datetime.now() - cache_time < self._cache_ttl:
                return cached_results
        
        try:
            # Get embeddings from the GNN model if available
            if self.model:
                scores = self.model.predict_connections(user_id, [m['id'] for m in potential_matches])
            else:
                # Fallback to simpler similarity matching
                user_vector = self._get_user_vector(user_data)
                match_vectors = [self._get_user_vector(m) for m in potential_matches]
                
                similarities = [
                    cosine_similarity(user_vector.reshape(1, -1), 
                                   mv.reshape(1, -1))[0][0]
                    for mv in match_vectors
                ]
                scores = np.array(similarities)
            
            # Sort and get top matches
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = [
                {
                    **potential_matches[i],
                    'match_score': float(scores[i]),
                    'match_reasons': self._get_match_reasons(
                        user_data,
                        potential_matches[i]
                    )
                }
                for i in top_indices
            ]
            
            # Cache results
            self._cache[cache_key] = (datetime.now(), results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting matches: {e}")
            return []
    
    def recommend_events(self,
                        user_data: Dict[str, Any],
                        available_events: List[Dict[str, Any]],
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend professional events based on user interests.
        
        Args:
            user_data: Data about the user
            available_events: List of available events
            top_k: Number of events to recommend
            
        Returns:
            List of recommended events with scores
        """
        try:
            user_interests = set(user_data.get('interests', []))
            user_skills = set(user_data.get('skills', []))
            
            scored_events = []
            for event in available_events:
                score = 0
                reasons = []
                
                # Match event topics with user interests
                event_topics = set(event.get('topics', []))
                matching_interests = user_interests & event_topics
                if matching_interests:
                    score += len(matching_interests) * 2
                    reasons.append(f"Matches your interests: {', '.join(matching_interests)}")
                
                # Match required skills
                required_skills = set(event.get('required_skills', []))
                matching_skills = user_skills & required_skills
                if matching_skills:
                    score += len(matching_skills)
                    reasons.append(f"Relevant to your skills: {', '.join(matching_skills)}")
                
                # Consider event format preference
                if event.get('format') == user_data.get('preferred_event_format'):
                    score += 1
                    reasons.append("Matches your preferred event format")
                
                scored_events.append({
                    **event,
                    'match_score': score,
                    'match_reasons': reasons
                })
            
            # Sort by score and return top k
            return sorted(
                scored_events,
                key=lambda x: x['match_score'],
                reverse=True
            )[:top_k]
            
        except Exception as e:
            logger.error(f"Error recommending events: {e}")
            return []
    
    def recommend_groups(self,
                        user_data: Dict[str, Any],
                        available_groups: List[Dict[str, Any]],
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend professional groups based on user profile.
        
        Args:
            user_data: Data about the user
            available_groups: List of available groups
            top_k: Number of groups to recommend
            
        Returns:
            List of recommended groups with scores
        """
        try:
            user_interests = set(user_data.get('interests', []))
            user_skills = set(user_data.get('skills', []))
            user_industry = user_data.get('industry')
            
            scored_groups = []
            for group in available_groups:
                score = 0
                reasons = []
                
                # Match group focus areas with user interests
                group_focus = set(group.get('focus_areas', []))
                matching_interests = user_interests & group_focus
                if matching_interests:
                    score += len(matching_interests) * 2
                    reasons.append(f"Aligns with your interests: {', '.join(matching_interests)}")
                
                # Match industry
                if group.get('industry') == user_industry:
                    score += 3
                    reasons.append(f"Relevant to your industry: {user_industry}")
                
                # Consider skill development opportunities
                group_skills = set(group.get('relevant_skills', []))
                skill_matches = user_skills & group_skills
                if skill_matches:
                    score += len(skill_matches)
                    reasons.append(f"Matches your skill areas: {', '.join(skill_matches)}")
                
                # Consider network value
                if group.get('member_count', 0) > 100:
                    score += 1
                    reasons.append("Active community with many professionals")
                
                scored_groups.append({
                    **group,
                    'match_score': score,
                    'match_reasons': reasons
                })
            
            # Sort by score and return top k
            return sorted(
                scored_groups,
                key=lambda x: x['match_score'],
                reverse=True
            )[:top_k]
            
        except Exception as e:
            logger.error(f"Error recommending groups: {e}")
            return []
    
    def _get_match_reasons(self,
                          user_data: Dict[str, Any],
                          match_data: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasons for a match."""
        reasons = []
        
        # Check common skills
        user_skills = set(user_data.get('skills', []))
        match_skills = set(match_data.get('skills', []))
        common_skills = user_skills & match_skills
        if common_skills:
            reasons.append(f"Shares {len(common_skills)} skills with you")
        
        # Check common interests
        user_interests = set(user_data.get('interests', []))
        match_interests = set(match_data.get('interests', []))
        common_interests = user_interests & match_interests
        if common_interests:
            reasons.append(f"Has {len(common_interests)} common interests")
        
        # Check industry alignment
        if user_data.get('industry') == match_data.get('industry'):
            reasons.append("Works in the same industry")
        
        # Check mutual connections
        mutual_connections = len(
            set(user_data.get('connections', [])) &
            set(match_data.get('connections', []))
        )
        if mutual_connections > 0:
            reasons.append(f"Has {mutual_connections} mutual connections")
        
        return reasons 