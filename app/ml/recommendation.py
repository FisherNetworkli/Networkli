"""
Networkli Recommendation System

This module serves as the main entry point for the recommendation system.
It provides a unified API for generating recommendations regardless of the
underlying implementation (simple attribute matching or machine learning).
"""

import logging
from typing import List, Dict, Any, Optional
from supabase import Client

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
        # Determine which algorithm to use
        use_ml = False
        algorithm_version = "simple-attribute-matching-v1"
        
        if algorithm == "ml" and self.ml_recommender is not None:
            use_ml = True
            algorithm_version = "ml-graph-neural-network-v1"
        elif algorithm == "auto" and self.ml_recommender is not None:
            # Check if we have enough data for ML
            # For now, always default to simple matching
            use_ml = False
        
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
                "algorithm_version": algorithm_version
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
                    "algorithm_version": "simple-attribute-matching-v1 (fallback)"
                }
            else:
                return {
                    "recommendations": [],
                    "count": 0,
                    "algorithm_version": algorithm_version,
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