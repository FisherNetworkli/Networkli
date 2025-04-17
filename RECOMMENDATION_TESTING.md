# Networkli Recommendation Algorithm Testing

This document describes how to use the recommendation algorithm testing script.

## Overview

The `test_recommendation.py` script is designed to evaluate and compare different recommendation algorithms in the Networkli platform. It generates synthetic test data and measures the performance of each algorithm using various metrics.

## Features

- Generate test users with varying profiles (industry, skills, location, etc.)
- Create synthetic connections and interactions between users
- Test multiple recommendation algorithms (simple, ML-based, auto)
- Calculate performance metrics:
  - Precision@k and Recall@k
  - Diversity (industry/location variety)
  - Explanation quality
  - Response time

## Requirements

- Python 3.7+
- Required Python packages:
  ```
  requests
  ```

## Usage

### Basic Usage

```bash
python test_recommendation.py
```

This will run the test with default settings (20 test users, testing 5 users, all algorithms).

### Advanced Options

```bash
python test_recommendation.py --num-users 50 --test-size 10 --algorithms simple,ml
```

#### Available Options:

- `--num-users`: Number of synthetic users to generate (default: 20)
- `--test-size`: Number of users to test recommendations for (default: 5)
- `--algorithms`: Comma-separated list of algorithms to test (default: simple,ml,auto)

## Interpreting Results

The script outputs a comparison table with the following metrics:

- **P@5**: Precision at 5 (what percentage of the top 5 recommendations are relevant)
- **R@5**: Recall at 5 (what percentage of all relevant items appear in the top 5)
- **Diversity**: How diverse the recommendations are (by industry and location)
- **Expl Quality**: Quality of recommendation explanations (specific vs. generic)
- **Avg Recs**: Average number of recommendations per user
- **Resp Time**: Average response time in seconds

Example output:
```
=== Algorithm Comparison ===
Algorithm  P@5      R@5      Diversity  Expl Quality   Avg Recs   Resp Time  
-----------------------------------------------------------------
simple     0.720    0.360    0.350      0.850          8.5        0.042s
ml         0.680    0.340    0.700      0.550          9.2        0.125s
auto       0.700    0.350    0.650      0.750          8.8        0.085s

Recommended algorithm: simple
```

## How It Works

1. **Data Generation**: Creates synthetic user profiles with realistic attributes
2. **Ground Truth**: Establishes "ground truth" for relevant recommendations based on shared industries and interests
3. **Simulation**: Simulates API calls to the recommendation endpoint (can be replaced with actual API calls)
4. **Metrics Calculation**: Computes performance metrics for each algorithm
5. **Comparison**: Compares algorithms and suggests the best one based on a weighted combination of metrics

## Customization

For integration with the actual Supabase database and API, you'll need to:

1. Update the Supabase configuration at the top of the script:
   ```python
   SUPABASE_URL = os.environ.get("SUPABASE_URL", "your_supabase_url")
   SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your_supabase_anon_key")
   ```

2. Replace the simulated API call with a real call to your recommendation endpoint in the `test_recommendation_algorithm` function:
   ```python
   response = requests.get(
       f"{API_BASE_URL}/recommendations/{user_id}",
       params={"algorithm": algorithm, "limit": limit},
       headers={"Authorization": f"Bearer {your_jwt_token}"}
   )
   recommendations = response.json().get("recommendations", [])
   ```

## Extending the Tests

To add more metrics or test different aspects:

1. Add new metric functions similar to `precision_at_k` and `diversity_score`
2. Update the `test_recommendation_algorithm` function to calculate your new metrics
3. Include the new metrics in the comparison table output 