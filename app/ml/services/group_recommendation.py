"""
Group Recommendation Service

This module provides algorithms for recommending groups to users based on 
their profile, interests, and interaction history.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from supabase import Client

logger = logging.getLogger(__name__)

class GroupRecommendationService:
    """Service for generating group recommendations."""
    
    def __init__(self, supabase_client: Client):
        """Initialize the group recommendation service.
        
        Args:
            supabase_client: Supabase client for database access
        """
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
    
    async def get_group_recommendations(
        self, 
        user_id: str, 
        limit: int = 10, 
        exclude_joined: bool = True,
        include_reason: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get group recommendations for a user based on their profile and activity.
        
        Args:
            user_id: The ID of the user to get recommendations for
            limit: Maximum number of recommendations to return
            exclude_joined: Whether to exclude groups the user has already joined
            include_reason: Whether to include a reason for each recommendation
            
        Returns:
            List of recommended groups with similarity scores and reasons
        """
        try:
            # Step 1: Get the user's profile
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                self.logger.error(f"User profile not found for user_id: {user_id}")
                return []
            
            # Step 2: Get the user's interests
            user_interests = await self._get_user_interests(user_id)
            
            # Step 3: Get the user's skills
            user_skills = await self._get_user_skills(user_id)
            
            # Step 4: If excluding joined groups, get the groups the user has already joined
            excluded_group_ids = set()
            if exclude_joined:
                joined_groups = await self._get_joined_groups(user_id)
                excluded_group_ids.update(joined_groups)
            
            # Step 5: Get candidate groups (exclude joined groups if requested)
            candidate_groups = await self._get_candidate_groups(excluded_group_ids)
            
            # Step 6: Get groups that user's connections have joined
            connection_groups = await self._get_connection_joined_groups(user_id)
            
            # Step 7: Calculate relevance scores for each group
            recommendations = []
            for group in candidate_groups:
                group_id = group.get('id')
                
                # Calculate relevance and gather matching attributes
                relevance_score, matching_attributes = self._calculate_group_relevance(
                    user_profile=user_profile,
                    user_interests=user_interests,
                    user_skills=user_skills,
                    group=group,
                    connection_groups=connection_groups
                )
                
                # Add recommendation with score and matching attributes
                recommendation = {
                    **group,
                    "relevance_score": relevance_score
                }
                
                # Add recommendation reason if requested
                if include_reason:
                    recommendation["reason"] = self._generate_group_recommendation_reason(matching_attributes)
                
                recommendations.append(recommendation)
            
            # Sort by relevance score and return top recommendations
            recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting group recommendations: {e}", exc_info=True)
            return []
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's profile from the database."""
        try:
            result = await self.supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            return result.data
        except Exception as e:
            self.logger.error(f"Exception retrieving profile for {user_id}: {e}")
            return None
    
    async def _get_user_interests(self, user_id: str) -> List[str]:
        """Get a user's interests from the database."""
        try:
            result = await self.supabase.table("user_preferences").select("interests").eq("user_id", user_id).single().execute()
            if result.data and "interests" in result.data:
                return result.data["interests"] or []
            return []
        except Exception as e:
            self.logger.error(f"Exception retrieving interests for {user_id}: {e}")
            return []
    
    async def _get_user_skills(self, user_id: str) -> List[str]:
        """Get a user's skills from the database."""
        try:
            result = await self.supabase.table("user_preferences").select("skills").eq("user_id", user_id).single().execute()
            if result.data and "skills" in result.data:
                return result.data["skills"] or []
            return []
        except Exception as e:
            self.logger.error(f"Exception retrieving skills for {user_id}: {e}")
            return []
    
    async def _get_joined_groups(self, user_id: str) -> Set[str]:
        """Get IDs of groups that the user has already joined."""
        try:
            result = await self.supabase.table("group_members").select("group_id").eq("member_id", user_id).execute()
            return {item.get("group_id") for item in result.data if item.get("group_id")}
        except Exception as e:
            self.logger.error(f"Exception retrieving joined groups for {user_id}: {e}")
            return set()
    
    async def _get_candidate_groups(self, exclude_ids: Set[str]) -> List[Dict[str, Any]]:
        """Get candidate groups for recommendations, excluding specified group IDs."""
        try:
            result = await self.supabase.table("groups").select("*").execute()
            return [group for group in result.data if group.get("id") not in exclude_ids]
        except Exception as e:
            self.logger.error(f"Exception retrieving candidate groups: {e}")
            return []
    
    async def _get_connection_joined_groups(self, user_id: str) -> Dict[str, int]:
        """Get groups that user's connections have joined, with count of connections in each group."""
        try:
            # Get user's connections
            connections_result = await self.supabase.table("connections").select("receiver_id, requester_id").or_(f"requester_id.eq.{user_id},receiver_id.eq.{user_id}").eq("status", "accepted").execute()
            
            if not connections_result.data:
                return {}
            
            # Extract connection IDs
            connection_ids = set()
            for conn in connections_result.data:
                if conn.get("requester_id") == user_id:
                    connection_ids.add(conn.get("receiver_id"))
                else:
                    connection_ids.add(conn.get("requester_id"))
            
            if not connection_ids:
                return {}
            
            # Get groups joined by connections
            groups_result = await self.supabase.table("group_members").select("group_id").in_("member_id", list(connection_ids)).execute()
            
            # Count occurrences of each group
            group_counts = {}
            for item in groups_result.data:
                group_id = item.get("group_id")
                if group_id:
                    group_counts[group_id] = group_counts.get(group_id, 0) + 1
            
            return group_counts
        except Exception as e:
            self.logger.error(f"Exception retrieving connection groups for {user_id}: {e}")
            return {}
    
    def _calculate_group_relevance(
        self,
        user_profile: Dict[str, Any],
        user_interests: List[str],
        user_skills: List[str],
        group: Dict[str, Any],
        connection_groups: Dict[str, int]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate relevance score between a user and a group.
        
        Returns:
            Tuple of (relevance_score, matching_attributes)
        """
        # Initialize weights for different attributes
        weights = {
            "category_match": 0.3,
            "interest_match": 0.25,
            "connection_match": 0.25,
            "location_match": 0.15,
            "industry_match": 0.05
        }
        
        # Initialize score and matching attributes
        score = 0.0
        matching_attributes = {}
        
        # 1. Category match (direct match between group category and user interests)
        group_category = group.get("category", "").lower()
        if group_category and any(interest.lower() == group_category for interest in user_interests):
            score += weights["category_match"]
            matching_attributes["category"] = group_category
        
        # 2. Interest match (group topics vs user interests)
        group_topics = group.get("topics", [])
        if isinstance(group_topics, str):
            # Handle case where topics might be stored as comma-separated string
            group_topics = [topic.strip() for topic in group_topics.split(",")]
        
        matching_interests = []
        for interest in user_interests:
            if any(topic.lower() == interest.lower() for topic in group_topics):
                matching_interests.append(interest)
        
        if matching_interests:
            interest_score = min(len(matching_interests) * 0.1, weights["interest_match"])
            score += interest_score
            matching_attributes["interests"] = matching_interests
        
        # 3. Connection match (how many connections are in this group)
        group_id = group.get("id")
        if group_id in connection_groups:
            connection_count = connection_groups[group_id]
            # Scale by number of connections, up to max weight
            connection_score = min(connection_count * 0.05, weights["connection_match"])
            score += connection_score
            matching_attributes["connection_count"] = connection_count
        
        # 4. Location match
        user_location = user_profile.get("location", "").lower()
        group_location = group.get("location", "").lower()
        
        if user_location and group_location and (
            user_location == group_location or
            user_location in group_location or
            group_location in user_location
        ):
            score += weights["location_match"]
            matching_attributes["location"] = group_location
        
        # 5. Industry match
        user_industry = user_profile.get("industry", "").lower()
        group_industry = group.get("industry", "").lower()
        
        if user_industry and group_industry and user_industry == group_industry:
            score += weights["industry_match"]
            matching_attributes["industry"] = group_industry
        
        return score, matching_attributes
    
    def _generate_group_recommendation_reason(self, matching_attributes: Dict[str, Any]) -> str:
        """Generate a human-readable reason for a group recommendation."""
        if not matching_attributes:
            return "This group might be interesting for your professional development."
        
        reasons = []
        
        # Category match
        if "category" in matching_attributes:
            reasons.append(f"This group is about {matching_attributes['category']}, which aligns with your interests")
        
        # Interest matches
        if "interests" in matching_attributes and matching_attributes["interests"]:
            if len(matching_attributes["interests"]) == 1:
                reasons.append(f"The group focuses on {matching_attributes['interests'][0]}, which you're interested in")
            else:
                interests_text = ", ".join(matching_attributes["interests"][:2])
                reasons.append(f"The group covers topics like {interests_text} that match your interests")
        
        # Connection matches
        if "connection_count" in matching_attributes:
            count = matching_attributes["connection_count"]
            if count == 1:
                reasons.append("One of your connections is a member of this group")
            else:
                reasons.append(f"{count} of your connections are members of this group")
        
        # Location match
        if "location" in matching_attributes:
            reasons.append(f"This group is located in {matching_attributes['location']}, where you are")
        
        # Industry match
        if "industry" in matching_attributes:
            reasons.append(f"This group is focused on the {matching_attributes['industry']} industry, which matches your profile")
        
        # Build the final reason string
        if reasons:
            return reasons[0] + (f". {reasons[1]}" if len(reasons) > 1 else "")
        else:
            return "This group might be interesting for your professional development."

# Factory function for consistent initialization
def create_group_recommendation_service(supabase_client: Client) -> GroupRecommendationService:
    """Create and return a GroupRecommendationService instance."""
    return GroupRecommendationService(supabase_client) 