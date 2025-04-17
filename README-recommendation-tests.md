# Networkli Recommendation Testing

This directory contains scripts for testing and evaluating the recommendation algorithms used in Networkli.

## Test Scripts

### 1. Comprehensive Testing (`test_recommendation.py`)

This script generates synthetic test data (users, groups, events, connections, etc.) and tests the recommendation algorithms on that data. It calculates various metrics including precision, recall, diversity, and more.

#### Features:
- Generate test users, groups, events, and interactions
- Test user recommendations with different algorithms (simple, ML, auto)
- Test group recommendations 
- Test event recommendations
- Calculate and compare metrics across different recommendation types
- Analyze relevance score distributions
- Compare results from multiple algorithms

#### Usage:
```bash
# Basic usage with default settings
python test_recommendation.py

# Test with specific parameters
python test_recommendation.py --num-users 20 --test-size 5 --algorithms simple,ml,auto

# Test only group and event recommendations
python test_recommendation.py --rec-types group,event

# Run comprehensive comparison of all recommendation types
python test_recommendation.py --compare-all

# Save results to a JSON file
python test_recommendation.py --save-results

# Use actual API endpoints instead of simulation
python test_recommendation.py --use-api
```

### 2. API Endpoint Testing (`test_api_recommendations.py`)

This script directly tests the recommendation API endpoints with a real user ID. It's useful for testing in staging or production environments.

#### Features:
- Test all recommendation endpoints (user, group, event)
- Test different algorithms
- Measure response times
- Display sample recommendations
- Save results to a JSON file

#### Usage:
```bash
# Basic usage with required user ID
python test_api_recommendations.py --user-id your-user-id

# Specify API URL and key
python test_api_recommendations.py --user-id your-user-id --api-url https://api.example.com --api-key your-api-key

# Specify number of recommendations and algorithms
python test_api_recommendations.py --user-id your-user-id --limit 20 --algorithms simple,ml
```

## Environment Variables

Both scripts can use the following environment variables:
- `API_BASE_URL`: Base URL for the API (default: `http://localhost:8000`)
- `SUPABASE_KEY`: API key for authentication

## Metrics Explained

- **Precision@k**: The proportion of recommended items in the top-k that are relevant
- **Recall@k**: The proportion of relevant items that are recommended in the top-k
- **Diversity**: A measure of how diverse the recommendations are (0-1)
- **Explanation Quality**: A measure of how good the recommendation explanations are (0-1)

## Recommendation Types

1. **User Recommendations**: Recommendations for users to connect with
2. **Group Recommendations**: Recommendations for groups to join
3. **Event Recommendations**: Recommendations for events to attend 