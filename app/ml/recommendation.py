"""
Networkli Recommendation System

This module serves as the main entry point for the recommendation system.
It provides a unified API for generating recommendations regardless of the
underlying implementation (simple attribute matching or machine learning).
"""

import logging
from typing import List, Dict, Any, Optional
from supabase import Client
import random

# Import the simple recommendation service using a relative path
from .services.simple_recommendation import create_recommendation_service

# Import the ML-based recommendation if available using a relative path
try:
    from .services.prediction_service import PredictionService
    ml_available = True
except ImportError:
    ml_available = False
    logging.warning("PredictionService not found or could not be imported. ML recommendations disabled.")

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Unified recommendation engine that selects the appropriate algorithm."""
    
    def __init__(self, supabase_client: Client):
        """Initialize the recommendation engine.
        
        Args:
            supabase_client: Supabase client for database access
        """
        self.supabase = supabase_client
        
        # Initialize the simple recommendation service
        self.simple_recommender = create_recommendation_service(supabase_client)
        
        # Initialize the ML-based recommendation service if available
        self.ml_recommender = None
        if ml_available:
            try:
                self.ml_recommender = PredictionService(supabase_client)
                logger.info("ML-based recommendation system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ML-based recommendation: {e}")
    
    async def _is_demo_mode_enabled(self) -> bool:
        """Check if demo mode is enabled in system settings."""
        try:
            result = await self.supabase.table("system_settings").select("*").eq("key", "demo_mode_enabled").single().execute()
            if result.data and result.data.get("value") == "true":
                return True
            return False
        except Exception as e:
            logger.warning(f"Error checking demo mode status: {e}")
            return False
    
    async def _get_celebrity_recommendations(self, user_id: str, limit: int, include_reason: bool = True) -> List[Dict[str, Any]]:
        """Get recommendations prioritizing celebrity profiles for demo mode.
        
        Args:
            user_id: ID of the user to get recommendations for
            limit: Maximum number of recommendations to return
            include_reason: Whether to include a reason for each recommendation
            
        Returns:
            List of celebrity profile recommendations
        """
        try:
            # Get user profile to match with celebrities
            user_profile_result = await self.supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            user_profile = user_profile_result.data if user_profile_result.data else {}
            
            # Get celebrity profiles
            celebrity_result = await self.supabase.table("profiles").select("*").eq("is_celebrity", True).limit(limit).execute()
            celebrities = celebrity_result.data if celebrity_result.data else []
            
            if not celebrities:
                logger.warning("No celebrity profiles found for demo mode")
                return []
            
            # Add personalized reasons based on user attributes
            if include_reason and user_profile:
                for celebrity in celebrities:
                    # Find matching attributes
                    matching_attrs = {}
                    
                    # Check for matching industry
                    if user_profile.get("industry") and celebrity.get("industry"):
                        if user_profile["industry"].lower() == celebrity["industry"].lower():
                            matching_attrs["industry"] = celebrity["industry"]
                    
                    # Check for matching location
                    if user_profile.get("location") and celebrity.get("location"):
                        if user_profile["location"].lower() == celebrity["location"].lower():
                            matching_attrs["location"] = celebrity["location"]
                    
                    # Check for matching skills or interests
                    if user_profile.get("skills") and celebrity.get("skills"):
                        user_skills = set(user_profile["skills"]) if isinstance(user_profile["skills"], list) else set()
                        celeb_skills = set(celebrity["skills"]) if isinstance(celebrity["skills"], list) else set()
                        matching_skills = user_skills.intersection(celeb_skills)
                        if matching_skills:
                            skill = next(iter(matching_skills))
                            matching_attrs["skill"] = skill
                    
                    # Generate personalized reason
                    if matching_attrs:
                        celebrity["reason"] = self.simple_recommender._generate_recommendation_reason(matching_attrs)
                    else:
                        celebrity["reason"] = f"Connect with {celebrity['full_name']}, a leader in {celebrity['industry']}."
                        
                    # Add demo flag
                    celebrity["is_demo"] = True
                    
            # Shuffle to get different celebrities each time
            random.shuffle(celebrities)
            
            return celebrities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting celebrity recommendations: {e}", exc_info=True)
            return []
    
    async def get_recommendations(self, 
                                user_id: str, 
                                algorithm: str = "auto",
                                limit: int = 10, 
                                exclude_connected: bool = True,
                                include_reason: bool = True) -> Dict[str, Any]:
        """Get recommendations for a user.
        
        Args:
            user_id: ID of the user to get recommendations for
            algorithm: Which algorithm to use ('simple', 'ml', or 'auto')
            limit: Maximum number of recommendations to return
            exclude_connected: Whether to exclude already connected users
            include_reason: Whether to include a reason for each recommendation
            
        Returns:
            Dictionary with recommendations and metadata
        """
        # Check if demo mode is enabled
        demo_mode = await self._is_demo_mode_enabled()
        
        # In demo mode, prioritize celebrity recommendations
        if demo_mode:
            logger.info(f"Demo mode enabled, prioritizing celebrity recommendations for user {user_id}")
            
            # Get celebrity recommendations
            celebrity_recommendations = await self._get_celebrity_recommendations(
                user_id=user_id,
                limit=limit,
                include_reason=include_reason
            )
            
            if celebrity_recommendations:
                # Ensure at least 60% of recommendations are celebrities in demo mode
                celebrity_count = max(min(int(limit * 0.6), len(celebrity_recommendations)), 2)
                regular_count = limit - celebrity_count
                
                # Get regular recommendations for the remaining slots
                regular_recommendations = []
                if regular_count > 0:
                    # Determine which algorithm to use for regular recommendations
                    use_ml = False
                    if algorithm == "ml" and self.ml_recommender is not None:
                        use_ml = True
                    elif algorithm == "auto" and self.ml_recommender is not None:
                        use_ml = True
                    
                    try:
                        if use_ml:
                            regular_recommendations = await self.ml_recommender.get_recommendations(
                                user_id=user_id,
                                limit=regular_count,
                                exclude_connected=exclude_connected
                            )
                            
                            if include_reason and regular_recommendations:
                                regular_recommendations = await self._add_reasons_to_ml_recommendations(
                                    user_id, regular_recommendations
                                )
                        else:
                            regular_recommendations = await self.simple_recommender.get_user_recommendations(
                                user_id=user_id,
                                limit=regular_count,
                                exclude_connected=exclude_connected,
                                include_reason=include_reason
                            )
                    except Exception as e:
                        logger.error(f"Error getting regular recommendations in demo mode: {e}", exc_info=True)
                
                # Combine celebrity and regular recommendations
                combined_recommendations = celebrity_recommendations[:celebrity_count]
                
                # If we have regular recommendations, add them
                if regular_recommendations:
                    combined_recommendations.extend(regular_recommendations)
                
                # Shuffle slightly but ensure first 1-2 are celebrities
                first_celebs = combined_recommendations[:2]
                rest = combined_recommendations[2:] if len(combined_recommendations) > 2 else []
                random.shuffle(rest)
                final_recommendations = first_celebs + rest
                
                return {
                    "recommendations": final_recommendations,
                    "count": len(final_recommendations),
                    "algorithm_version": "demo-mode-celebrity-priority",
                    "demo_mode": True
                }
        
        # Continue with normal recommendation process if not in demo mode or no celebrities
        # Determine which algorithm to use
        use_ml = False
        algorithm_version = "simple-attribute-matching-v1"
        
        if algorithm == "ml" and self.ml_recommender is not None:
            use_ml = True
            algorithm_version = "ml-graph-neural-network-v1"
        elif algorithm == "auto" and self.ml_recommender is not None:
            # Use ML algorithm when 'auto' is selected and ML is available
            use_ml = True
            algorithm_version = "ml-graph-neural-network-v1"
            # Future: Could add more sophisticated logic here to choose 
            # based on data availability, A/B testing flags, etc.
        
        # Get recommendations using the selected algorithm
        recommendations = []
        
        try:
            if use_ml:
                recommendations = await self.ml_recommender.get_recommendations(
                    user_id=user_id,
                    limit=limit,
                    exclude_connected=exclude_connected
                )
                
                # Add reasons if requested
                if include_reason and recommendations:
                    # ML model might not provide reasons, so generate them
                    recommendations = await self._add_reasons_to_ml_recommendations(
                        user_id, recommendations
                    )
            else:
                recommendations = await self.simple_recommender.get_user_recommendations(
                    user_id=user_id,
                    limit=limit,
                    exclude_connected=exclude_connected,
                    include_reason=include_reason
                )
                
            return {
                "recommendations": recommendations,
                "count": len(recommendations),
                "algorithm_version": algorithm_version,
                "demo_mode": demo_mode
            }
        except Exception as e:
            logger.error(f"Error getting recommendations for {user_id}: {e}", exc_info=True)
            # Fallback to simple recommendations if ML fails
            if use_ml:
                logger.info(f"Falling back to simple recommendations for {user_id}")
                recommendations = await self.simple_recommender.get_user_recommendations(
                    user_id=user_id,
                    limit=limit,
                    exclude_connected=exclude_connected,
                    include_reason=include_reason
                )
                
                return {
                    "recommendations": recommendations,
                    "count": len(recommendations),
                    "algorithm_version": "simple-attribute-matching-v1 (fallback)",
                    "demo_mode": demo_mode
                }
            else:
                return {
                    "recommendations": [],
                    "count": 0,
                    "algorithm_version": algorithm_version,
                    "demo_mode": demo_mode,
                    "error": str(e)
                }
    
    async def _add_reasons_to_ml_recommendations(self,
                                              user_id: str,
                                              recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add human-readable reasons to ML-generated recommendations.
        
        Args:
            user_id: ID of the user receiving recommendations
            recommendations: List of recommendations from ML model
            
        Returns:
            Recommendations with reasons added
        """
        try:
            # Get user profile
            user_profile_result = await self.supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            user_profile = user_profile_result.data if user_profile_result.data else {}
            
            # For each recommendation, find matching attributes to generate a reason
            for recommendation in recommendations:
                profile_id = recommendation.get("id")
                if not profile_id:
                    continue
                
                # Find matching attributes (industry, location, etc.)
                matching_attrs = {}
                
                if user_profile.get("industry") and recommendation.get("industry"):
                    if user_profile["industry"].lower() == recommendation["industry"].lower():
                        matching_attrs["industry"] = recommendation["industry"]
                
                if user_profile.get("location") and recommendation.get("location"):
                    if user_profile["location"].lower() == recommendation["location"].lower():
                        matching_attrs["location"] = recommendation["location"]
                
                # Generate reason text
                if matching_attrs:
                    reason_text = self.simple_recommender._generate_recommendation_reason(matching_attrs)
                    recommendation["reason"] = reason_text
                else:
                    recommendation["reason"] = "This professional may be a valuable connection for your career growth."
            
            return recommendations
        except Exception as e:
            logger.warning(f"Error generating reasons for ML recommendations: {e}")
            # Return the original recommendations
            return recommendations

# Factory function for consistent initialization
def create_recommendation_engine(supabase_client: Client) -> RecommendationEngine:
    """Create and return a RecommendationEngine instance."""
    return RecommendationEngine(supabase_client) 