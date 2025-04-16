"""
Recommendation service modules for Networkli.

This package contains different recommendation algorithms:
- SimpleRecommendationService: Lightweight attribute-matching algorithm
- PredictionService: ML-based recommendation algorithm
"""

from .simple_recommendation import create_recommendation_service, SimpleRecommendationService

# Import ML-based services conditionally
try:
    from .prediction_service import PredictionService
    ml_available = True
except ImportError:
    ml_available = False 