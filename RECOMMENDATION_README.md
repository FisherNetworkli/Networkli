# Networkli Recommendation System Testing Suite

This repository contains tools and documentation for testing and evaluating the Networkli recommendation system. Our recommendation system provides personalized user-to-user recommendations for professional networking.

## Components

- **test_recommendation.py**: Main testing script that simulates and evaluates recommendation algorithms
- **RECOMMENDATION_TESTING.md**: Documentation on how to use the testing script
- **RECOMMENDATION_TESTING_RESULTS.md**: Summary of latest test results

## About the Recommendation System

The Networkli recommendation system uses multiple algorithms to suggest relevant professional connections:

1. **Simple Attribute-Matching**: Uses profile attributes (industry, location, skills, interests) to find similar users. Fast and explainable.

2. **ML-Based Algorithm**: Combines content-based and graph-based features to find deeper patterns in user relationships. Provides higher-quality recommendations but is more computationally intensive.

3. **Auto Selection**: Dynamically chooses between the simple and ML algorithms based on available data and specific use case.

## Evaluation Metrics

We evaluate our recommendation algorithms using the following metrics:

- **Precision@k**: What percentage of the top k recommendations are relevant
- **Recall@k**: What percentage of all relevant items appear in the top k
- **Diversity**: How diverse the recommendations are (by industry and location)
- **Explanation Quality**: How specific and meaningful the recommendation reasons are
- **Response Time**: How quickly the algorithm can generate recommendations

## Running Tests

### Basic Usage

```bash
python test_recommendation.py
```

### Advanced Options

```bash
python test_recommendation.py --num-users 50 --test-size 10 --algorithms simple,ml
```

See [RECOMMENDATION_TESTING.md](RECOMMENDATION_TESTING.md) for detailed instructions.

## Implementation Details

The testing framework works by:

1. Generating synthetic user profiles with realistic attributes
2. Creating test connections and interaction history between users
3. Simulating the recommendation algorithms with characteristics matching our production system
4. Calculating performance metrics and comparing algorithms

## Integration with Production System

To integrate this testing framework with the actual Networkli API:

1. Configure the environment variables:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   API_BASE_URL=your_api_url
   ```

2. Update the test script to use real API calls instead of simulations:
   - Uncomment the API request code in `test_recommendation_algorithm`
   - Comment out the simulation code

3. For database testing, use a separate test database or create isolated test data.

## Current Results

As of our latest testing (June 2024), the "auto" algorithm provides the best overall performance. See [RECOMMENDATION_TESTING_RESULTS.md](RECOMMENDATION_TESTING_RESULTS.md) for detailed results.

## Next Steps

- Implement A/B testing for algorithm variations in production
- Add more metrics like serendipity and coverage
- Improve ground truth generation for more accurate evaluation
- Create visualization tools for recommendation quality metrics over time

## Contributing

To contribute to the recommendation testing framework:

1. Add additional metrics in the testing script
2. Improve the simulation accuracy to better match production behavior
3. Extend the testing to include more edge cases and user types
4. Add automated regression testing to prevent performance degradation

## Dependencies

- Python 3.7+
- Required packages: requests
- Optional packages for visualization: matplotlib, seaborn (if implementing visualization) 