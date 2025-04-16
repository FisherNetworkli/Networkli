# Networkli Recommendation Engine

This directory contains the recommendation algorithms used in the Networkli platform. The system is designed with a modular architecture to support both simple attribute-matching recommendations and more advanced machine learning-based approaches.

## Directory Structure

```
app/ml/
├── recommendation.py       # Main entry point for the recommendation engine
├── services/
│   └── simple_recommendation.py  # Simple attribute-matching algorithm
├── models/                 # ML models directory
├── utils/                  # Utility functions
├── preprocessing/          # Data preprocessing modules
└── training/               # Model training scripts
```

## Usage

The recommendation engine provides a unified API through the `RecommendationEngine` class in `recommendation.py`:

```python
from app.ml.recommendation import create_recommendation_engine

# Initialize the engine with a Supabase client
recommendation_engine = create_recommendation_engine(supabase_client)

# Get recommendations
recommendations = await recommendation_engine.get_recommendations(
    user_id="user123",
    algorithm="auto",  # Options: "simple", "ml", "auto"
    limit=10,
    exclude_connected=True,
    include_reason=True
)
```

## Recommendation Algorithms

### Simple Attribute Matching

The simple attribute-matching algorithm (`SimpleRecommendationService`) uses a weighted similarity calculation based on profile attributes:

- Industry match (20%)
- Skills match - Jaccard similarity (30%)
- Interests match - Jaccard similarity (20%) 
- Location match (10%)
- Experience level match (10%)
- Domain match (10%)

Each recommendation includes a human-readable reason explaining why the match was made.

### Machine Learning Model (Advanced)

The ML-based recommendation (`NetworkliGNN` and related models) uses graph neural networks to learn user representations from profile attributes and connection patterns. Features include:

- Embedding users in a professional networking space
- Capturing domain expertise and skill compatibility
- Learning from user interaction signals
- Providing personalized recommendations

## Integration with Main API

The recommendation engine is integrated with the FastAPI backend through the `/recommendations/{profile_id}` endpoint, which accepts parameters for customizing the recommendation process.

## Development

To extend the recommendation system:

1. For simple improvements, modify the attribute weights or similarity calculation in `simple_recommendation.py`
2. For ML-based improvements, update the models in the `models` directory
3. To add a new algorithm, create a new service class and integrate it in `recommendation.py` 