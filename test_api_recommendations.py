#!/usr/bin/env python
"""
Networkli API Recommendation Test

This script tests the recommendation API endpoints by:
1. Making direct calls to the API endpoints
2. Measuring response times and number of recommendations
3. Comparing the results across different recommendation types

Usage:
  python test_api_recommendations.py --user-id your-user-id
"""

import os
import sys
import json
import time
import random
import asyncio
import argparse
from typing import List, Dict, Any
import requests
from datetime import datetime

# API configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def format_duration(duration_sec: float) -> str:
    """Format duration in seconds to a readable string"""
    if duration_sec < 0.001:
        return f"{duration_sec * 1000000:.0f}μs"
    elif duration_sec < 1:
        return f"{duration_sec * 1000:.0f}ms"
    else:
        return f"{duration_sec:.2f}s"

async def test_user_recommendation_api(user_id: str, algorithm: str = "auto", limit: int = 10) -> Dict[str, Any]:
    """Test the user recommendation API endpoint"""
    print(f"\nTesting User Recommendations API (algorithm={algorithm})")
    
    url = f"{API_BASE_URL}/recommendations/{user_id}?algorithm={algorithm}&limit={limit}"
    
    start_time = time.time()
    try:
        response = requests.get(url, headers={"apikey": SUPABASE_KEY})
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        recommendations = data.get("recommendations", [])
        
        print(f"Got {len(recommendations)} recommendations in {format_duration(end_time - start_time)}")
        
        # Print a sample recommendation
        if recommendations:
            sample = random.choice(recommendations)
            print("\nSample recommendation:")
            print(f"User: {sample.get('full_name', 'Unknown')}")
            print(f"Title: {sample.get('title', 'Unknown')}")
            print(f"Relevance: {sample.get('relevance_score', 0):.2f}")
            print(f"Reason: {sample.get('reason', 'Unknown')}")
        
        return {
            "count": len(recommendations),
            "response_time": end_time - start_time,
            "success": True
        }
    except Exception as e:
        end_time = time.time()
        print(f"Error: {e}")
        return {
            "count": 0,
            "response_time": end_time - start_time,
            "success": False,
            "error": str(e)
        }

async def test_group_recommendation_api(user_id: str, limit: int = 10) -> Dict[str, Any]:
    """Test the group recommendation API endpoint"""
    print(f"\nTesting Group Recommendations API")
    
    url = f"{API_BASE_URL}/group-recommendations/{user_id}?limit={limit}"
    
    start_time = time.time()
    try:
        response = requests.get(url, headers={"apikey": SUPABASE_KEY})
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        recommendations = data.get("group_recommendations", [])
        
        print(f"Got {len(recommendations)} recommendations in {format_duration(end_time - start_time)}")
        
        # Print a sample recommendation
        if recommendations:
            sample = random.choice(recommendations)
            print("\nSample recommendation:")
            print(f"Group: {sample.get('name', 'Unknown')}")
            print(f"Category: {sample.get('category', 'Unknown')}")
            print(f"Relevance: {sample.get('relevance_score', 0):.2f}")
            print(f"Reason: {sample.get('reason', 'Unknown')}")
        
        return {
            "count": len(recommendations),
            "response_time": end_time - start_time,
            "success": True
        }
    except Exception as e:
        end_time = time.time()
        print(f"Error: {e}")
        return {
            "count": 0,
            "response_time": end_time - start_time,
            "success": False,
            "error": str(e)
        }

async def test_event_recommendation_api(user_id: str, limit: int = 10) -> Dict[str, Any]:
    """Test the event recommendation API endpoint"""
    print(f"\nTesting Event Recommendations API")
    
    url = f"{API_BASE_URL}/event-recommendations/{user_id}?limit={limit}"
    
    start_time = time.time()
    try:
        response = requests.get(url, headers={"apikey": SUPABASE_KEY})
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        recommendations = data.get("event_recommendations", [])
        
        print(f"Got {len(recommendations)} recommendations in {format_duration(end_time - start_time)}")
        
        # Print a sample recommendation
        if recommendations:
            sample = random.choice(recommendations)
            print("\nSample recommendation:")
            print(f"Event: {sample.get('name', 'Unknown')}")
            print(f"Category: {sample.get('category', 'Unknown')}")
            print(f"When: {sample.get('start_time', 'Unknown')}")
            print(f"Relevance: {sample.get('relevance_score', 0):.2f}")
            print(f"Reason: {sample.get('reason', 'Unknown')}")
        
        return {
            "count": len(recommendations),
            "response_time": end_time - start_time,
            "success": True
        }
    except Exception as e:
        end_time = time.time()
        print(f"Error: {e}")
        return {
            "count": 0,
            "response_time": end_time - start_time,
            "success": False,
            "error": str(e)
        }

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Networkli API Recommendations")
    parser.add_argument("--user-id", required=True, help="User ID to test recommendations for")
    parser.add_argument("--api-url", help="Base URL for the API (defaults to env var API_BASE_URL)")
    parser.add_argument("--api-key", help="API key (defaults to env var SUPABASE_KEY)")
    parser.add_argument("--limit", type=int, default=10, help="Number of recommendations to request")
    parser.add_argument("--algorithms", default="simple,ml,auto", help="Comma-separated list of algorithms to test")
    args = parser.parse_args()
    
    global API_BASE_URL, SUPABASE_KEY
    
    if args.api_url:
        API_BASE_URL = args.api_url
    
    if args.api_key:
        SUPABASE_KEY = args.api_key
    
    if not API_BASE_URL:
        print("Error: API_BASE_URL is not set. Use --api-url or set the API_BASE_URL environment variable.")
        return
    
    if not SUPABASE_KEY:
        print("Warning: SUPABASE_KEY is not set. API calls may fail if authentication is required.")
    
    print(f"Testing recommendations for user: {args.user_id}")
    print(f"API URL: {API_BASE_URL}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "user_id": args.user_id,
        "api_url": API_BASE_URL,
        "limit": args.limit,
        "results": {}
    }
    
    # Test user recommendations with different algorithms
    for algorithm in args.algorithms.split(","):
        print(f"\n===== Testing {algorithm.upper()} Algorithm =====")
        result = await test_user_recommendation_api(args.user_id, algorithm, args.limit)
        results["results"][f"user_{algorithm}"] = result
    
    # Test group recommendations
    result = await test_group_recommendation_api(args.user_id, args.limit)
    results["results"]["group"] = result
    
    # Test event recommendations
    result = await test_event_recommendation_api(args.user_id, args.limit)
    results["results"]["event"] = result
    
    # Print summary
    print("\n===== Recommendation API Test Summary =====")
    print(f"User ID: {args.user_id}")
    print(f"Timestamp: {results['timestamp']}")
    print("\nResults:")
    for rec_type, result in results["results"].items():
        status = "✅ Success" if result.get("success", False) else "❌ Failed"
        count = result.get("count", 0)
        time_str = format_duration(result.get("response_time", 0))
        print(f"{rec_type.ljust(12)}: {status} | {count} recommendations in {time_str}")
    
    # Save results
    filename = f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 