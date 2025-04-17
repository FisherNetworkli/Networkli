"""
Member Alignment Module

This module provides algorithms and utilities for calculating alignment/similarity
between users in groups and events, identifying potential connections.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from supabase import Client

from app.ml.embedding import load_user_embedding, get_embedding_model
from app.ml.utils import cosine_similarity, interests_similarity, skills_similarity

logger = logging.getLogger(__name__)

class MemberAlignmentService:
    """Service for calculating alignment between members."""
    
    def __init__(self, supabase_client: Client):
        """Initialize the member alignment service.
        
        Args:
            supabase_client: Supabase client for database access
        """
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
    
    async def get_aligned_members(
        self, 
        user_id: str,
        entity_type: str,
        entity_id: str,
        limit: int = 5,
        min_similarity: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Find members in a group/event most aligned with the given user.
        
        Args:
            user_id: The ID of the user to find alignments for
            entity_type: Type of entity ("group" or "event")
            entity_id: ID of the group or event
            limit: Maximum number of aligned members to return
            min_similarity: Minimum similarity score to include in results
            
        Returns:
            List of aligned members with similarity scores and reasons
        """
        try:
            # 1. Get the user's profile and embedding
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                self.logger.error(f"User profile not found for user_id: {user_id}")
                return []
            
            user_embedding = await self._get_user_embedding(user_id)
            if user_embedding is None:
                self.logger.warning(f"No embedding found for user: {user_id}")
                # Continue with other similarity methods
            
            # 2. Get the members of the entity (group or event)
            members = await self._get_entity_members(entity_type, entity_id)
            
            # 3. Calculate alignment with each member
            alignment_results = []
            for member_id in members:
                if member_id == user_id:  # Skip the user themselves
                    continue
                
                # Get member profile
                member_profile = await self._get_user_profile(member_id)
                if not member_profile:
                    continue
                
                # Calculate similarity and matching attributes
                similarity_score, matching_attributes = await self._calculate_member_similarity(
                    user_profile=user_profile,
                    user_embedding=user_embedding,
                    member_profile=member_profile,
                    member_id=member_id
                )
                
                # Include if similarity is above threshold
                if similarity_score >= min_similarity:
                    alignment_results.append({
                        **member_profile,
                        "similarity_score": similarity_score,
                        "alignment_reason": self._generate_alignment_reason(matching_attributes)
                    })
            
            # Sort by similarity and return top results
            alignment_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            return alignment_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting aligned members: {e}", exc_info=True)
            return []
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's profile from the database."""
        try:
            result = await self.supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            return result.data
        except Exception as e:
            self.logger.error(f"Exception retrieving profile for {user_id}: {e}")
            return None
    
    async def _get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get a user's embedding vector."""
        try:
            # Try to load from user_embeddings table first
            result = await self.supabase.table("user_embeddings").select("embedding").eq("user_id", user_id).single().execute()
            if result.data and result.data.get("embedding"):
                return np.array(result.data.get("embedding"))
            
            # If not found, try to generate using the embedding model
            self.logger.info(f"No stored embedding found for {user_id}, generating new embedding")
            user_profile = await self._get_user_profile(user_id)
            if user_profile:
                embedding_model = get_embedding_model()
                if embedding_model:
                    return load_user_embedding(user_profile, embedding_model)
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting embedding for user {user_id}: {e}")
            return None
    
    async def _get_entity_members(self, entity_type: str, entity_id: str) -> List[str]:
        """Get the member IDs of a group or event."""
        try:
            if entity_type == "group":
                result = await self.supabase.table("group_members").select("member_id").eq("group_id", entity_id).execute()
                return [item.get("member_id") for item in result.data if item.get("member_id")]
            elif entity_type == "event":
                result = await self.supabase.table("event_attendance").select("user_id").eq("event_id", entity_id).execute()
                return [item.get("user_id") for item in result.data if item.get("user_id")]
            else:
                self.logger.error(f"Invalid entity type: {entity_type}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting members for {entity_type} {entity_id}: {e}")
            return []
    
    async def _calculate_member_similarity(
        self,
        user_profile: Dict[str, Any],
        user_embedding: Optional[np.ndarray],
        member_profile: Dict[str, Any],
        member_id: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate similarity between a user and another member using multiple methods.
        
        Returns:
            Tuple of (similarity_score, matching_attributes)
        """
        similarity_scores = {}
        matching_attributes = {}
        
        # 1. GNN Embedding Similarity (if available)
        if user_embedding is not None:
            member_embedding = await self._get_user_embedding(member_id)
            if member_embedding is not None:
                embedding_similarity = cosine_similarity(user_embedding, member_embedding)
                similarity_scores["embedding"] = embedding_similarity * 0.5  # High weight for GNN embeddings
        
        # 2. Profile attribute similarities
        
        # 2.1 Industry match
        user_industry = user_profile.get("industry", "")
        member_industry = member_profile.get("industry", "")
        if user_industry and member_industry and user_industry.lower() == member_industry.lower():
            similarity_scores["industry"] = 0.15
            matching_attributes["industry"] = member_industry
        
        # 2.2 Location match
        user_location = user_profile.get("location", "")
        member_location = member_profile.get("location", "")
        if user_location and member_location:
            if user_location.lower() == member_location.lower():
                similarity_scores["location"] = 0.1
                matching_attributes["location"] = member_location
            # Partial location match (city or state)
            elif (user_location.lower() in member_location.lower() or 
                  member_location.lower() in user_location.lower()):
                similarity_scores["location"] = 0.05
                matching_attributes["location"] = member_location
        
        # 2.3 Skills match
        user_skills = user_profile.get("skills", [])
        member_skills = member_profile.get("skills", [])
        if user_skills and member_skills:
            skill_sim, matching_skills = skills_similarity(user_skills, member_skills)
            if skill_sim > 0:
                similarity_scores["skills"] = skill_sim * 0.2
                matching_attributes["skills"] = matching_skills
        
        # 2.4 Interests match
        user_interests = user_profile.get("interests", [])
        member_interests = member_profile.get("interests", [])
        if user_interests and member_interests:
            interest_sim, matching_interests = interests_similarity(user_interests, member_interests)
            if interest_sim > 0:
                similarity_scores["interests"] = interest_sim * 0.25
                matching_attributes["interests"] = matching_interests
        
        # 2.5 Experience level match
        user_experience = user_profile.get("experience_level", "")
        member_experience = member_profile.get("experience_level", "")
        if user_experience and member_experience and user_experience == member_experience:
            similarity_scores["experience"] = 0.05
            matching_attributes["experience"] = member_experience
        
        # Calculate total similarity (weighted average of all scores)
        total_similarity = sum(similarity_scores.values())
        
        # Normalize to 0-1 range and ensure minimum similarity score
        normalized_similarity = min(1.0, total_similarity)
        
        return normalized_similarity, matching_attributes
    
    def _generate_alignment_reason(self, matching_attributes: Dict[str, Any]) -> str:
        """Generate a human-readable reason for the alignment."""
        if not matching_attributes:
            return "You may find value connecting with this person for professional networking."
        
        reasons = []
        
        if "industry" in matching_attributes:
            reasons.append(f"You both work in the {matching_attributes['industry']} industry")
        
        if "location" in matching_attributes:
            reasons.append(f"You're both located in {matching_attributes['location']}")
        
        if "skills" in matching_attributes:
            skills = matching_attributes["skills"]
            if len(skills) == 1:
                reasons.append(f"You both have expertise in {skills[0]}")
            else:
                skill_text = ", ".join(skills[:2])
                if len(skills) > 2:
                    skill_text += f" and {len(skills) - 2} other skills"
                reasons.append(f"You share skills in {skill_text}")
        
        if "interests" in matching_attributes:
            interests = matching_attributes["interests"]
            if len(interests) == 1:
                reasons.append(f"You're both interested in {interests[0]}")
            else:
                interest_text = ", ".join(interests[:2])
                if len(interests) > 2:
                    interest_text += f" and {len(interests) - 2} other topics"
                reasons.append(f"You share interests in {interest_text}")
        
        if "experience" in matching_attributes:
            reasons.append(f"You're both at a {matching_attributes['experience']} level in your careers")
        
        # Combine reasons into a compelling statement
        if len(reasons) >= 3:
            return f"{reasons[0]}. {reasons[1]}. Additionally, {reasons[2].lower()}"
        elif len(reasons) == 2:
            return f"{reasons[0]}. {reasons[1]}"
        elif reasons:
            return reasons[0]
        else:
            return "You may find value connecting with this person for professional networking."

def create_member_alignment_service(supabase_client: Client) -> MemberAlignmentService:
    """Create a member alignment service instance."""
    return MemberAlignmentService(supabase_client) 