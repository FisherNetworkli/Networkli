"""
Simple Recommendation Service

This module provides a lightweight attribute-matching algorithm for recommending users
without requiring machine learning models.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from collections import Counter
from supabase import Client

logger = logging.getLogger(__name__)

class SimpleRecommendationService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
        
    async def get_user_recommendations(
        self, 
        user_id: str, 
        limit: int = 10, 
        exclude_connected: bool = True,
        include_reason: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get user recommendations using a simple attribute-matching algorithm.
        
        Args:
            user_id: The ID of the user to get recommendations for
            limit: Maximum number of recommendations to return
            exclude_connected: Whether to exclude users already connected
            include_reason: Whether to include a reason for each recommendation
            
        Returns:
            List of recommended profiles with similarity scores and reasons
        """
        try:
            # Step 1: Get the user's profile
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                self.logger.error(f"User profile not found for user_id: {user_id}")
                return []
            
            # Step 2: Get the user's skills
            user_skills = await self._get_user_skills(user_id)
            
            # Step 3: Get the user's interests
            user_interests = await self._get_user_interests(user_id)
            
            # Step 4: Determine connected users to exclude (if needed)
            excluded_users = {user_id}  # Always exclude the user themselves
            if exclude_connected:
                connections = await self._get_user_connections(user_id)
                excluded_users.update(connections)
            
            # Step 5: Get candidate profiles (exclude connected users)
            candidates = await self._get_candidate_profiles(excluded_users)
            
            # Step 6: Calculate similarity scores for each candidate
            recommendations = []
            for candidate in candidates:
                candidate_id = candidate.get('id')
                
                # Skip if candidate is somehow in excluded users 
                # (shouldn't happen but added as a safeguard)
                if candidate_id in excluded_users:
                    continue
                
                # Calculate similarity and gather matching attributes
                similarity_score, matching_attributes = await self._calculate_similarity(
                    user_profile, user_skills, user_interests,
                    candidate, 
                    await self._get_user_skills(candidate_id),
                    await self._get_user_interests(candidate_id)
                )
                
                # Create recommendation with score and matching attributes
                recommendation = {
                    **candidate,
                    "similarity_score": similarity_score,
                }
                
                # Add recommendation reason if requested
                if include_reason:
                    recommendation["reason"] = self._generate_recommendation_reason(matching_attributes)
                
                recommendations.append(recommendation)
            
            # Step 7: Sort by similarity score and return top recommendations
            recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
            return recommendations[:limit]
        
        except Exception as e:
            self.logger.error(f"Error getting user recommendations: {e}", exc_info=True)
            return []
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's profile from the database."""
        try:
            result = self.supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            if result.error:
                self.logger.error(f"Error retrieving profile for {user_id}: {result.error}")
                return None
            return result.data
        except Exception as e:
            self.logger.error(f"Exception retrieving profile for {user_id}: {e}")
            return None
    
    async def _get_user_skills(self, user_id: str) -> List[str]:
        """Get a user's skills from the database."""
        try:
            result = self.supabase.table("user_skills").select("skill_name").eq("user_id", user_id).execute()
            if result.error:
                self.logger.error(f"Error retrieving skills for {user_id}: {result.error}")
                return []
            return [item.get("skill_name") for item in result.data]
        except Exception as e:
            self.logger.error(f"Exception retrieving skills for {user_id}: {e}")
            return []
    
    async def _get_user_interests(self, user_id: str) -> List[str]:
        """Get a user's interests from the database."""
        try:
            # Join user_interests with interests to get interest names
            result = self.supabase.from_("user_interests").select("""
                interest_id,
                interests!inner(name)
            """).eq("user_id", user_id).execute()
            
            if result.error:
                self.logger.error(f"Error retrieving interests for {user_id}: {result.error}")
                return []
            
            # Extract interest names from the result
            interests = []
            for item in result.data:
                if "interests" in item and "name" in item["interests"]:
                    interests.append(item["interests"]["name"])
            return interests
        except Exception as e:
            self.logger.error(f"Exception retrieving interests for {user_id}: {e}")
            return []
    
    async def _get_user_connections(self, user_id: str) -> Set[str]:
        """Get IDs of users that are connected to the given user."""
        try:
            # Get outgoing connections
            outgoing = self.supabase.table("connections").select("receiver_id").eq("requester_id", user_id).execute()
            
            # Get incoming connections
            incoming = self.supabase.table("connections").select("requester_id").eq("receiver_id", user_id).execute()
            
            # Combine and return unique IDs
            connected_ids = set()
            if not outgoing.error:
                connected_ids.update([item.get("receiver_id") for item in outgoing.data])
            if not incoming.error:
                connected_ids.update([item.get("requester_id") for item in incoming.data])
            
            return connected_ids
        except Exception as e:
            self.logger.error(f"Exception retrieving connections for {user_id}: {e}")
            return set()
    
    async def _get_candidate_profiles(self, exclude_ids: Set[str]) -> List[Dict[str, Any]]:
        """Get candidate profiles for recommendations, excluding specified user IDs."""
        try:
            # Get all profiles except excluded ones
            # Note: For large user bases, this would need pagination or other optimization
            result = self.supabase.table("profiles").select("*").execute()
            
            if result.error:
                self.logger.error(f"Error retrieving candidate profiles: {result.error}")
                return []
            
            # Filter out excluded users
            return [profile for profile in result.data if profile.get("id") not in exclude_ids]
        except Exception as e:
            self.logger.error(f"Exception retrieving candidate profiles: {e}")
            return []
    
    async def _calculate_similarity(
        self,
        user_profile: Dict[str, Any],
        user_skills: List[str],
        user_interests: List[str],
        candidate_profile: Dict[str, Any],
        candidate_skills: List[str],
        candidate_interests: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate similarity score between a user and a candidate.
        
        Returns:
            Tuple of (similarity_score, matching_attributes)
        """
        # Initialize weights for different attributes
        weights = {
            "industry": 0.2,
            "skills": 0.3,
            "interests": 0.2,
            "location": 0.1,
            "experience": 0.1,
            "domain": 0.1
        }
        
        matching_attributes = {}
        weighted_score = 0.0
        
        # 1. Industry match
        if user_profile.get("industry") and candidate_profile.get("industry"):
            if user_profile["industry"].lower() == candidate_profile["industry"].lower():
                matching_attributes["industry"] = candidate_profile["industry"]
                weighted_score += weights["industry"]
        
        # 2. Skills match (Jaccard similarity)
        if user_skills and candidate_skills:
            user_skills_set = set([s.lower() for s in user_skills])
            candidate_skills_set = set([s.lower() for s in candidate_skills])
            
            if user_skills_set and candidate_skills_set:
                intersection = user_skills_set.intersection(candidate_skills_set)
                union = user_skills_set.union(candidate_skills_set)
                
                if union:
                    skill_similarity = len(intersection) / len(union)
                    weighted_score += weights["skills"] * skill_similarity
                    
                    if intersection:
                        matching_attributes["skills"] = list(intersection)
        
        # 3. Interests match (Jaccard similarity)
        if user_interests and candidate_interests:
            user_interests_set = set([i.lower() for i in user_interests])
            candidate_interests_set = set([i.lower() for i in candidate_interests])
            
            if user_interests_set and candidate_interests_set:
                intersection = user_interests_set.intersection(candidate_interests_set)
                union = user_interests_set.union(candidate_interests_set)
                
                if union:
                    interest_similarity = len(intersection) / len(union)
                    weighted_score += weights["interests"] * interest_similarity
                    
                    if intersection:
                        matching_attributes["interests"] = list(intersection)
        
        # 4. Location match
        if user_profile.get("location") and candidate_profile.get("location"):
            if user_profile["location"].lower() == candidate_profile["location"].lower():
                matching_attributes["location"] = candidate_profile["location"]
                weighted_score += weights["location"]
        
        # 5. Experience level similarity
        if user_profile.get("experience_level") and candidate_profile.get("experience_level"):
            if user_profile["experience_level"] == candidate_profile["experience_level"]:
                matching_attributes["experience_level"] = candidate_profile["experience_level"]
                weighted_score += weights["experience"]
        
        # 6. Domain ID match
        if user_profile.get("domain_id") is not None and candidate_profile.get("domain_id") is not None:
            if user_profile["domain_id"] == candidate_profile["domain_id"]:
                matching_attributes["domain_id"] = candidate_profile["domain_id"]
                weighted_score += weights["domain"]
        
        return weighted_score, matching_attributes
    
    def _generate_recommendation_reason(self, matching_attributes: Dict[str, Any]) -> str:
        """Generate a human-readable reason for the recommendation based on matching attributes."""
        if not matching_attributes:
            return "This professional would be a valuable addition to your network."
        
        reasons = []
        
        if "industry" in matching_attributes:
            reasons.append(f"Professional in the {matching_attributes['industry']} industry")
        
        if "skills" in matching_attributes:
            skills = matching_attributes["skills"]
            if len(skills) == 1:
                reasons.append(f"Has expertise in {skills[0]} like you")
            elif len(skills) == 2:
                reasons.append(f"Shares professional expertise in {skills[0]} and {skills[1]}")
            elif len(skills) > 2:
                reasons.append(f"Matches {len(skills)} professional skills including {skills[0]} and {skills[1]}")
        
        if "interests" in matching_attributes:
            interests = matching_attributes["interests"]
            if len(interests) == 1:
                reasons.append(f"Professional interest in {interests[0]}")
            elif len(interests) == 2:
                reasons.append(f"Common professional focus on {interests[0]} and {interests[1]}")
            elif len(interests) > 2:
                reasons.append(f"Shares {len(interests)} professional interests with you")
        
        if "location" in matching_attributes:
            reasons.append(f"Located in {matching_attributes['location']} for potential local networking")
        
        if "experience_level" in matching_attributes:
            reasons.append(f"Has {matching_attributes['experience_level']} experience in their field")
        
        if not reasons:
            return "This professional would be a valuable addition to your network."
        
        # Return the top 2 reasons to keep it concise
        if len(reasons) > 2:
            return f"{reasons[0]}. {reasons[1]}."
        else:
            return ". ".join(reasons) + "."

# Factory function to create recommendation service
def create_recommendation_service(supabase_client: Client) -> SimpleRecommendationService:
    """Create and return a SimpleRecommendationService instance."""
    return SimpleRecommendationService(supabase_client) 