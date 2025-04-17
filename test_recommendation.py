#!/usr/bin/env python
"""
Networkli Recommendation Algorithm Test

This script tests the recommendation algorithms by:
1. Generating test users with various profiles
2. Creating test data for connections and interactions
3. Calling the recommendation endpoint with different algorithms
4. Measuring the quality of recommendations with multiple metrics

Usage:
  python test_recommendation.py --num-users 10 --algorithms simple,ml,auto
"""

import os
import sys
import json
import time
import random
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
import requests
from uuid import uuid4

# Replace with your actual Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# These are sample values for synthetic test data
INDUSTRIES = [
    "Technology", "Healthcare", "Finance", "Education", "Marketing",
    "Manufacturing", "Retail", "Consulting", "Media", "Energy"
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", 
    "Boston, MA", "Chicago, IL", "Denver, CO", "Los Angeles, CA",
    "Miami, FL", "Atlanta, GA"
]

SKILLS = [
    "Python", "JavaScript", "React", "Machine Learning", "Data Science",
    "Project Management", "Communication", "Leadership", "Marketing",
    "Design", "UX/UI", "DevOps", "Cloud Computing", "Database Management",
    "Full Stack Development", "Content Creation", "SEO", "Sales", "Accounting"
]

INTERESTS = [
    "AI/ML", "Blockchain", "Startups", "Venture Capital", "Remote Work",
    "Sustainability", "Healthcare Tech", "Fintech", "EdTech", "IoT",
    "Cybersecurity", "Cloud Computing", "Big Data", "Digital Marketing",
    "E-commerce", "Green Energy", "Social Impact", "Diversity & Inclusion"
]

# Add group and event related constants
GROUP_CATEGORIES = [
    "Tech", "Finance", "Marketing", "Design", "Leadership", 
    "Entrepreneurship", "Data Science", "HR", "Sales"
]

GROUP_TOPICS = [
    "AI/ML", "Blockchain", "UX Design", "Digital Marketing", "Venture Capital",
    "Career Growth", "Leadership Development", "Remote Work", "Cybersecurity",
    "Frontend Development", "Backend Development", "DevOps", "Product Management"
]

EVENT_FORMATS = ["online", "in-person", "hybrid"]

EVENT_CATEGORIES = [
    "Conference", "Workshop", "Networking", "Hackathon", "Webinar",
    "Panel Discussion", "Training", "Meetup", "Job Fair"
]

# Evaluation metrics
def precision_at_k(relevant_items: Set[str], recommended_items: List[str], k: int) -> float:
    """Calculate precision@k metric"""
    if not recommended_items or k == 0:
        return 0.0
    k = min(k, len(recommended_items))
    return len(relevant_items.intersection(recommended_items[:k])) / k

def recall_at_k(relevant_items: Set[str], recommended_items: List[str], k: int) -> float:
    """Calculate recall@k metric"""
    if not relevant_items or not recommended_items or k == 0:
        return 0.0
    k = min(k, len(recommended_items))
    return len(relevant_items.intersection(recommended_items[:k])) / len(relevant_items)

def diversity_score(recommendations: List[Dict[str, Any]]) -> float:
    """Calculate diversity of recommendations based on industries and locations"""
    if not recommendations:
        return 0.0
    industries = set(r.get('industry', '') for r in recommendations if r.get('industry'))
    locations = set(r.get('location', '') for r in recommendations if r.get('location'))
    return (len(industries) / len(recommendations)) * 0.5 + (len(locations) / len(recommendations)) * 0.5

def explanation_quality(recommendations: List[Dict[str, Any]]) -> float:
    """Measure the quality of recommendation explanations"""
    if not recommendations:
        return 0.0
    
    # Check for non-generic explanations
    total_score = 0.0
    for r in recommendations:
        reason = r.get('reason', '')
        # Higher score for specific reasons vs generic ones
        if reason and not reason.startswith("This professional may be"):
            total_score += 1.0
        elif reason:
            total_score += 0.3
    
    return total_score / len(recommendations)

# Test data generation
def generate_test_users(num_users: int) -> List[Dict[str, Any]]:
    """Generate test user profiles with varying attributes"""
    users = []
    for i in range(num_users):
        user_id = str(uuid4())
        test_user = {
            "id": user_id,
            "email": f"test{i}@example.com",
            "first_name": f"TestUser{i}",
            "last_name": f"LastName{i}",
            "full_name": f"TestUser{i} LastName{i}",
            "avatar_url": f"https://randomuser.me/api/portraits/lego/{i}.jpg",
            "title": random.choice([
                "Software Engineer", "Product Manager", "Data Scientist", 
                "Marketing Manager", "UX Designer", "Sales Director",
                "CEO", "CTO", "Project Manager", "Business Analyst"
            ]),
            "company": f"TestCompany{i % 10 + 1}",
            "industry": random.choice(INDUSTRIES),
            "location": random.choice(LOCATIONS),
            "bio": f"This is a test user {i} with interests in technology and networking.",
            "skills": random.sample(SKILLS, min(random.randint(3, 8), len(SKILLS))),
            "interests": random.sample(INTERESTS, min(random.randint(2, 6), len(INTERESTS))),
            "experience_level": random.choice(["Entry", "Mid", "Senior", "Executive"]),
            "created_at": datetime.now().isoformat()
        }
        users.append(test_user)
    return users

def generate_test_connections(users: List[Dict[str, Any]], connection_density: float = 0.1) -> List[Dict[str, Any]]:
    """Generate test connections between users with specified density"""
    connections = []
    user_ids = [user["id"] for user in users]
    
    # For each user, create connections with probability = connection_density
    for i, user_id in enumerate(user_ids):
        for j, other_id in enumerate(user_ids):
            if i != j and random.random() < connection_density:
                connection = {
                    "id": str(uuid4()),
                    "requester_id": user_id,
                    "receiver_id": other_id,
                    "status": random.choice(["pending", "accepted"]),
                    "created_at": datetime.now().isoformat()
                }
                connections.append(connection)
    
    return connections

def generate_test_interactions(users: List[Dict[str, Any]], interaction_density: float = 0.2) -> List[Dict[str, Any]]:
    """Generate test interaction history between users"""
    interactions = []
    user_ids = [user["id"] for user in users]
    
    interaction_types = ["PROFILE_VIEW", "RECOMMENDATION_CLICK", "SWIPE_RIGHT", "SWIPE_LEFT"]
    
    for i, user_id in enumerate(user_ids):
        # Each user has some interactions with other users
        for j, other_id in enumerate(user_ids):
            if i != j and random.random() < interaction_density:
                interaction_type = random.choice(interaction_types)
                
                interaction = {
                    "id": str(uuid4()),
                    "user_id": user_id,
                    "interaction_type": interaction_type,
                    "target_entity_type": "PROFILE",
                    "target_entity_id": other_id,
                    "created_at": datetime.now().isoformat(),
                    "metadata": {
                        "source": random.choice(["search", "recommendation", "browse"]),
                        "session_id": f"test-session-{random.randint(1000, 9999)}"
                    }
                }
                interactions.append(interaction)
    
    return interactions

async def insert_test_data(users, connections, interactions):
    """Insert test data into Supabase (simulation only for this test)"""
    # In a real implementation, we would insert the data using the Supabase client
    print(f"[Simulation] Inserted {len(users)} test users")
    print(f"[Simulation] Inserted {len(connections)} test connections")
    print(f"[Simulation] Inserted {len(interactions)} test interactions")
    
    # Just for this test, we'll simulate successful insertion
    return True

def simulate_simple_recommendations(
    user: Dict[str, Any], 
    all_users: List[Dict[str, Any]], 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Simulate the simple attribute-matching recommendation algorithm"""
    user_id = user.get("id")
    recommendations = []
    
    # Get user's attributes
    user_industry = user.get("industry", "")
    user_location = user.get("location", "")
    user_interests = set(user.get("interests", []))
    user_skills = set(user.get("skills", []))
    
    # Calculate similarity scores for each candidate
    for candidate in all_users:
        candidate_id = candidate.get("id")
        if candidate_id == user_id:
            continue
        
        # Initial score
        score = 0.0
        matching_attributes = {}
        
        # Industry match (highest weight)
        if candidate.get("industry") == user_industry:
            score += 0.35
            matching_attributes["industry"] = candidate.get("industry")
        
        # Location match
        if candidate.get("location") == user_location:
            score += 0.2
            matching_attributes["location"] = candidate.get("location")
        
        # Interests overlap
        candidate_interests = set(candidate.get("interests", []))
        shared_interests = user_interests.intersection(candidate_interests)
        if shared_interests:
            score += len(shared_interests) * 0.15
            matching_attributes["interests"] = list(shared_interests)
        
        # Skills overlap
        candidate_skills = set(candidate.get("skills", []))
        shared_skills = user_skills.intersection(candidate_skills)
        if shared_skills:
            score += len(shared_skills) * 0.1
            matching_attributes["skills"] = list(shared_skills)
        
        # Experience level
        if candidate.get("experience_level") == user.get("experience_level"):
            score += 0.1
            matching_attributes["experience_level"] = candidate.get("experience_level")
        
        # Add recommendation if score is high enough
        if score > 0.2:
            # Create reason based on matching attributes
            reason = generate_recommendation_reason(matching_attributes)
            
            # Add to recommendations
            recommendations.append({
                **candidate,
                "similarity_score": score,
                "reason": reason
            })
    
    # Sort by similarity score and return top recommendations
    recommendations.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    return recommendations[:limit]

def simulate_ml_recommendations(
    user: Dict[str, Any], 
    all_users: List[Dict[str, Any]], 
    interactions: List[Dict[str, Any]], 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Simulate the ML-based recommendation algorithm"""
    user_id = user.get("id")
    recommendations = []
    
    # This is a more sophisticated simulation that includes:
    # 1. Network-based features (interactions)
    # 2. Content-based features (similar to simple)
    # 3. Some randomness to simulate ML model prediction variance
    
    # Create an interaction graph
    interaction_counts = {}
    for interaction in interactions:
        if interaction.get("user_id") == user_id:
            target_id = interaction.get("target_entity_id")
            
            if target_id not in interaction_counts:
                interaction_counts[target_id] = 0
            
            # Positive interactions increase score
            if interaction.get("interaction_type") in ["PROFILE_VIEW", "RECOMMENDATION_CLICK", "SWIPE_RIGHT"]:
                interaction_counts[target_id] += 1
            # Negative interactions decrease score
            elif interaction.get("interaction_type") == "SWIPE_LEFT":
                interaction_counts[target_id] -= 1
    
    # Get user's attributes
    user_industry = user.get("industry", "")
    user_location = user.get("location", "")
    user_interests = set(user.get("interests", []))
    user_skills = set(user.get("skills", []))
    
    # Find users who interacted with similar profiles
    second_degree_connections = set()
    for interaction in interactions:
        # If someone interacted with the same profiles as our user
        if (interaction.get("user_id") != user_id and 
            interaction.get("target_entity_id") in interaction_counts and
            interaction_counts[interaction.get("target_entity_id")] > 0):
            
            second_degree_connections.add(interaction.get("user_id"))
    
    # Calculate similarity scores for each candidate
    for candidate in all_users:
        candidate_id = candidate.get("id")
        if candidate_id == user_id:
            continue
        
        # Base score from content similarity (similar to simple algorithm but with different weights)
        content_score = 0.0
        
        # Industry match
        if candidate.get("industry") == user_industry:
            content_score += 0.25
        
        # Location match
        if candidate.get("location") == user_location:
            content_score += 0.15
        
        # Interests overlap
        candidate_interests = set(candidate.get("interests", []))
        shared_interests = user_interests.intersection(candidate_interests)
        content_score += len(shared_interests) * 0.1
        
        # Skills overlap
        candidate_skills = set(candidate.get("skills", []))
        shared_skills = user_skills.intersection(candidate_skills)
        content_score += len(shared_skills) * 0.08
        
        # Network score
        network_score = 0.0
        
        # Direct interactions
        if candidate_id in interaction_counts:
            network_score += min(interaction_counts[candidate_id] * 0.2, 0.4)
        
        # Second-degree connection
        if candidate_id in second_degree_connections:
            network_score += 0.15
        
        # Final score with ML random variance
        ml_variance = random.uniform(-0.1, 0.1)  # Add some randomness to simulate ML variance
        final_score = (content_score * 0.5) + (network_score * 0.5) + ml_variance
        
        # Threshold and reason
        if final_score > 0.15:
            # Create reason - ML model has more generic reasons sometimes
            if random.random() < 0.7:  # 70% chance of specific reason
                matching_attributes = {}
                if candidate.get("industry") == user_industry:
                    matching_attributes["industry"] = candidate.get("industry")
                if len(shared_interests) > 0:
                    matching_attributes["interests"] = list(shared_interests)[:2]  # Limit to 2 for reason
                reason = generate_recommendation_reason(matching_attributes)
            else:
                reason = "This professional may be valuable based on your network and profile"
            
            # Add to recommendations
            recommendations.append({
                **candidate,
                "similarity_score": final_score,
                "reason": reason
            })
    
    # Sort by similarity score and return top recommendations
    recommendations.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    return recommendations[:limit]

def generate_recommendation_reason(matching_attributes: Dict[str, Any]) -> str:
    """Generate a human-readable recommendation reason based on matching attributes"""
    if not matching_attributes:
        return "This professional may be a valuable connection for your career growth."
    
    reasons = []
    
    # Industry match
    if "industry" in matching_attributes:
        reasons.append(f"They also work in the {matching_attributes['industry']} industry")
    
    # Location match
    if "location" in matching_attributes:
        reasons.append(f"They're located in {matching_attributes['location']} like you")
    
    # Interest matches
    if "interests" in matching_attributes and matching_attributes["interests"]:
        if len(matching_attributes["interests"]) == 1:
            reasons.append(f"You both share an interest in {matching_attributes['interests'][0]}")
        else:
            interests_str = ", ".join(matching_attributes["interests"][:2])
            reasons.append(f"You share interests in {interests_str}")
    
    # Skills matches
    if "skills" in matching_attributes and matching_attributes["skills"]:
        if len(matching_attributes["skills"]) == 1:
            reasons.append(f"You both have experience with {matching_attributes['skills'][0]}")
        else:
            skills_str = ", ".join(matching_attributes["skills"][:2])
            reasons.append(f"You have skills in common: {skills_str}")
    
    # Experience level
    if "experience_level" in matching_attributes:
        reasons.append(f"You're both at the {matching_attributes['experience_level']} career level")
    
    # Build the final reason string
    if reasons:
        return reasons[0] + (f". {reasons[1]}" if len(reasons) > 1 else "")
    else:
        return "This professional may be a valuable connection for your career growth."

async def test_recommendation_algorithm(
    users: List[Dict[str, Any]], 
    connections: List[Dict[str, Any]],
    interactions: List[Dict[str, Any]],
    test_size: int = 5, 
    algorithm: str = "auto",
    limit: int = 10
) -> Dict[str, Any]:
    """Test the recommendation algorithm for a subset of users and collect metrics"""
    if not users or test_size <= 0:
        return {"error": "Invalid test parameters"}
    
    # Select a subset of users to test
    test_users = random.sample(users, min(test_size, len(users)))
    
    # Dictionary to store cumulative metrics
    metrics = {
        "algorithm": algorithm,
        "total_users_tested": 0,
        "total_recommendations": 0,
        "avg_recommendation_count": 0,
        "avg_precision_at_5": 0,
        "avg_recall_at_5": 0,
        "avg_diversity": 0,
        "avg_explanation_quality": 0,
        "response_times": []
    }
    
    # Creating mock "ground truth" for each test user - in a real system this would
    # come from actual interaction data, connections, or similar user interests
    ground_truth = {}
    for user in test_users:
        # Simulate ground truth - users with same industry or 2+ shared interests
        user_industry = user.get("industry", "")
        user_interests = set(user.get("interests", []))
        
        truth_set = set()
        for other in users:
            other_id = other.get("id")
            if other_id == user.get("id"):
                continue
            
            if other.get("industry") == user_industry:
                truth_set.add(other_id)
            elif len(user_interests.intersection(set(other.get("interests", [])))) >= 2:
                truth_set.add(other_id)
        
        # Add users with positive interactions
        for interaction in interactions:
            if (interaction.get("user_id") == user.get("id") and 
                interaction.get("interaction_type") in ["PROFILE_VIEW", "RECOMMENDATION_CLICK", "SWIPE_RIGHT"]):
                truth_set.add(interaction.get("target_entity_id"))
        
        ground_truth[user.get("id")] = truth_set
    
    # Test each user
    for user in test_users:
        user_id = user.get("id")
        
        # Call the API endpoint (simulation in this test)
        start_time = time.time()
        try:
            # In a real implementation, this would be an actual API call:
            # response = requests.get(
            #     f"{API_BASE_URL}/recommendations/{user_id}",
            #     params={"algorithm": algorithm, "limit": limit},
            #     headers={"Authorization": f"Bearer {your_jwt_token}"}
            # )
            # recommendations = response.json().get("recommendations", [])
            
            # For this test, we'll simulate API response based on algorithm
            if algorithm == "simple":
                recommendations = simulate_simple_recommendations(user, users, limit)
                # Add artificial delay
                time.sleep(random.uniform(0.01, 0.03))
            elif algorithm == "ml":
                recommendations = simulate_ml_recommendations(user, users, interactions, limit)
                # ML algorithm is typically slower
                time.sleep(random.uniform(0.05, 0.15))
            elif algorithm == "auto":
                # Auto chooses between simple and ML based on data availability
                # For this simulation, give 70% chance of using ML
                if random.random() < 0.7:
                    recommendations = simulate_ml_recommendations(user, users, interactions, limit)
                    time.sleep(random.uniform(0.03, 0.1))
                else:
                    recommendations = simulate_simple_recommendations(user, users, limit)
                    time.sleep(random.uniform(0.01, 0.03))
            else:
                # Default to simple
                recommendations = simulate_simple_recommendations(user, users, limit)
            
            end_time = time.time()
            response_time = end_time - start_time
            metrics["response_times"].append(response_time)
            
            # Calculate metrics
            recommended_ids = [r.get("id") for r in recommendations]
            p_at_5 = precision_at_k(ground_truth[user_id], recommended_ids, 5)
            r_at_5 = recall_at_k(ground_truth[user_id], recommended_ids, 5)
            div_score = diversity_score(recommendations)
            exp_quality = explanation_quality(recommendations)
            
            # Update cumulative metrics
            metrics["total_users_tested"] += 1
            metrics["total_recommendations"] += len(recommendations)
            metrics["avg_precision_at_5"] += p_at_5
            metrics["avg_recall_at_5"] += r_at_5
            metrics["avg_diversity"] += div_score
            metrics["avg_explanation_quality"] += exp_quality
            
            print(f"User {user.get('full_name')}: P@5={p_at_5:.2f}, R@5={r_at_5:.2f}, Diversity={div_score:.2f}")
            
        except Exception as e:
            print(f"Error testing recommendations for user {user_id}: {e}")
    
    # Calculate averages
    if metrics["total_users_tested"] > 0:
        metrics["avg_recommendation_count"] = metrics["total_recommendations"] / metrics["total_users_tested"]
        metrics["avg_precision_at_5"] /= metrics["total_users_tested"]
        metrics["avg_recall_at_5"] /= metrics["total_users_tested"]
        metrics["avg_diversity"] /= metrics["total_users_tested"]
        metrics["avg_explanation_quality"] /= metrics["total_users_tested"]
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    return metrics

async def compare_algorithms(users: List[Dict[str, Any]], connections: List[Dict[str, Any]], interactions: List[Dict[str, Any]], test_size: int = 5):
    """Compare different recommendation algorithms"""
    algorithms = ["simple", "ml", "auto"]
    results = []
    
    for algorithm in algorithms:
        print(f"\nTesting '{algorithm}' algorithm...")
        metrics = await test_recommendation_algorithm(users, connections, interactions, test_size, algorithm)
        results.append(metrics)
    
    # Print comparison table
    print("\n=== Algorithm Comparison ===")
    print(f"{'Algorithm':<10} {'P@5':<8} {'R@5':<8} {'Diversity':<10} {'Expl Quality':<13} {'Avg Recs':<10} {'Resp Time':<10}")
    print("-" * 65)
    
    for result in results:
        algo = result.get("algorithm", "unknown")
        p_at_5 = result.get("avg_precision_at_5", 0)
        r_at_5 = result.get("avg_recall_at_5", 0)
        diversity = result.get("avg_diversity", 0)
        expl_quality = result.get("avg_explanation_quality", 0)
        avg_recs = result.get("avg_recommendation_count", 0)
        resp_time = result.get("avg_response_time", 0)
        
        print(f"{algo:<10} {p_at_5:<8.3f} {r_at_5:<8.3f} {diversity:<10.3f} {expl_quality:<13.3f} {avg_recs:<10.1f} {resp_time:<10.3f}s")
    
    # Determine the best algorithm based on combined metrics
    best_algo = max(results, key=lambda x: 
        (x.get("avg_precision_at_5", 0) * 0.4 + 
         x.get("avg_recall_at_5", 0) * 0.3 +
         x.get("avg_diversity", 0) * 0.15 +
         x.get("avg_explanation_quality", 0) * 0.15))
    
    print(f"\nRecommended algorithm: {best_algo.get('algorithm', 'unknown')}")
    return results

def generate_test_groups(num_groups: int, users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate test groups with varying attributes"""
    groups = []
    organizer_ids = [user["id"] for user in users]
    
    for i in range(num_groups):
        group_id = str(uuid4())
        
        # Randomly select an organizer
        organizer_id = random.choice(organizer_ids)
        
        # Select random category and topics
        category = random.choice(GROUP_CATEGORIES)
        num_topics = random.randint(1, 4)
        topics = random.sample(GROUP_TOPICS, num_topics)
        
        # Select random location (50% chance to match an existing user location)
        if random.random() < 0.5:
            location = random.choice([user.get("location", "") for user in users])
        else:
            location = random.choice(LOCATIONS)
        
        # Random industry focus (50% chance to match an existing user industry)
        if random.random() < 0.5:
            industry = random.choice([user.get("industry", "") for user in users])
        else:
            industry = random.choice(INDUSTRIES)
        
        group = {
            "id": group_id,
            "name": f"Test Group {i}",
            "description": f"This is a test group {i} focused on {category}.",
            "organizer_id": organizer_id,
            "category": category,
            "topics": topics,
            "location": location,
            "industry": industry,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "image_url": f"https://picsum.photos/seed/group{i}/400/300",
            "member_count": random.randint(3, 50)
        }
        groups.append(group)
    
    return groups

def generate_test_group_members(groups: List[Dict[str, Any]], users: List[Dict[str, Any]], membership_density: float = 0.1) -> List[Dict[str, Any]]:
    """Generate group memberships between users and groups"""
    group_members = []
    
    for user in users:
        user_id = user["id"]
        for group in groups:
            group_id = group["id"]
            
            # Skip if user is the organizer (they're already a member)
            if user_id == group.get("organizer_id"):
                continue
            
            # Randomly assign membership based on density
            if random.random() < membership_density:
                group_member = {
                    "id": str(uuid4()),
                    "group_id": group_id,
                    "member_id": user_id,
                    "role": random.choice(["member", "admin"]),
                    "joined_at": datetime.now().isoformat()
                }
                group_members.append(group_member)
    
    return group_members

def generate_test_events(num_events: int, groups: List[Dict[str, Any]], users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate test events with varying attributes"""
    events = []
    
    for i in range(num_events):
        event_id = str(uuid4())
        
        # 70% chance the event is organized by a group, 30% by an individual
        if random.random() < 0.7 and groups:
            group = random.choice(groups)
            organizer_type = "GROUP"
            organizer_id = group["id"]
            # Event inherits some attributes from the group
            category = group["category"]
            topics = group["topics"]
            location = group["location"]
        else:
            user = random.choice(users)
            organizer_type = "USER"
            organizer_id = user["id"]
            # User-organized event gets random attributes
            category = random.choice(EVENT_CATEGORIES)
            num_topics = random.randint(1, 3)
            topics = random.sample(GROUP_TOPICS, num_topics)
            location = user.get("location", random.choice(LOCATIONS))
        
        # Generate random start time (between now and 30 days in future)
        days_in_future = random.randint(0, 30)
        hours = random.randint(0, 23)
        minutes = random.choice([0, 15, 30, 45])
        
        start_time = datetime.now() + timedelta(days=days_in_future, hours=hours, minutes=minutes)
        # Event lasts between 1 and 4 hours
        end_time = start_time + timedelta(hours=random.randint(1, 4))
        
        event_format = random.choice(EVENT_FORMATS)
        
        event = {
            "id": event_id,
            "title": f"Test Event {i}",
            "description": f"This is a test event {i} focused on {category}.",
            "organizer_type": organizer_type,
            "organizer_id": organizer_id,
            "category": category,
            "topics": topics,
            "location": location,
            "format": event_format,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "image_url": f"https://picsum.photos/seed/event{i}/400/300",
            "max_attendees": random.randint(10, 100) if event_format != "online" else None,
            "is_public": random.choice([True, False])
        }
        events.append(event)
    
    return events

def generate_test_event_attendance(events: List[Dict[str, Any]], users: List[Dict[str, Any]], attendance_density: float = 0.1) -> List[Dict[str, Any]]:
    """Generate event attendance records between users and events"""
    event_attendance = []
    
    for user in users:
        user_id = user["id"]
        for event in events:
            event_id = event["id"]
            
            # Randomly assign attendance based on density
            if random.random() < attendance_density:
                attendance = {
                    "id": str(uuid4()),
                    "event_id": event_id,
                    "user_id": user_id,
                    "status": random.choice(["registered", "attended", "cancelled"]),
                    "registered_at": datetime.now().isoformat()
                }
                event_attendance.append(attendance)
    
    return event_attendance

# Add functions to test group and event recommendations
async def test_group_recommendation_algorithm(
    users: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    group_members: List[Dict[str, Any]],
    test_size: int = 5,
    limit: int = 10
) -> Dict[str, Any]:
    """Test the group recommendation algorithm for a subset of users"""
    if not users or not groups or test_size <= 0:
        return {"error": "Invalid test parameters"}
    
    # Select a subset of users to test
    test_users = random.sample(users, min(test_size, len(users)))
    
    # Dictionary to store cumulative metrics
    metrics = {
        "total_users_tested": 0,
        "total_recommendations": 0,
        "avg_recommendation_count": 0,
        "avg_precision_at_5": 0,
        "avg_recall_at_5": 0,
        "avg_diversity": 0,
        "avg_explanation_quality": 0,
        "response_times": []
    }
    
    # Store all recommendations for analysis
    all_recommendations = []
    
    # Creating mock "ground truth" for each test user
    ground_truth = {}
    for user in test_users:
        user_id = user.get("id")
        user_industry = user.get("industry", "")
        user_interests = set(user.get("interests", []))
        user_location = user.get("location", "")
        
        # Find groups that match user interests/industry/location
        truth_set = set()
        for group in groups:
            group_id = group.get("id")
            if group.get("industry") == user_industry:
                truth_set.add(group_id)
            elif group.get("location") == user_location:
                truth_set.add(group_id)
            elif group.get("category") in user_interests:
                truth_set.add(group_id)
            elif any(topic in user_interests for topic in group.get("topics", [])):
                truth_set.add(group_id)
        
        ground_truth[user_id] = truth_set
    
    # Get mapping of group memberships
    user_groups = {}
    for membership in group_members:
        user_id = membership.get("member_id")
        group_id = membership.get("group_id")
        if user_id not in user_groups:
            user_groups[user_id] = set()
        user_groups[user_id].add(group_id)
    
    # Test each user
    for user in test_users:
        user_id = user.get("id")
        
        # Simulate API call
        start_time = time.time()
        try:
            # Simulate group recommendations
            recommendations = simulate_group_recommendations(user, users, groups, group_members, limit)
            all_recommendations.extend(recommendations)
            
            end_time = time.time()
            response_time = end_time - start_time
            metrics["response_times"].append(response_time)
            
            # Calculate metrics
            recommended_ids = [r.get("id") for r in recommendations]
            p_at_5 = precision_at_k(ground_truth[user_id], recommended_ids, 5)
            r_at_5 = recall_at_k(ground_truth[user_id], recommended_ids, 5)
            div_score = calculate_group_diversity(recommendations)
            exp_quality = explanation_quality(recommendations)
            
            # Update cumulative metrics
            metrics["total_users_tested"] += 1
            metrics["total_recommendations"] += len(recommendations)
            metrics["avg_precision_at_5"] += p_at_5
            metrics["avg_recall_at_5"] += r_at_5
            metrics["avg_diversity"] += div_score
            metrics["avg_explanation_quality"] += exp_quality
            
            print(f"User {user.get('full_name')} (Group Recommendations): P@5={p_at_5:.2f}, R@5={r_at_5:.2f}, Diversity={div_score:.2f}")
            
        except Exception as e:
            print(f"Error testing group recommendations for user {user_id}: {e}")
    
    # Calculate averages
    if metrics["total_users_tested"] > 0:
        metrics["avg_recommendation_count"] = metrics["total_recommendations"] / metrics["total_users_tested"]
        metrics["avg_precision_at_5"] /= metrics["total_users_tested"]
        metrics["avg_recall_at_5"] /= metrics["total_users_tested"]
        metrics["avg_diversity"] /= metrics["total_users_tested"]
        metrics["avg_explanation_quality"] /= metrics["total_users_tested"]
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    # Analyze relevance scores
    if all_recommendations:
        relevance_analysis = analyze_relevance_distribution(all_recommendations, "group")
        metrics["relevance_analysis"] = relevance_analysis
    
    return metrics

async def test_event_recommendation_algorithm(
    users: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    event_attendance: List[Dict[str, Any]],
    test_size: int = 5,
    limit: int = 10
) -> Dict[str, Any]:
    """Test the event recommendation algorithm for a subset of users"""
    if not users or not events or test_size <= 0:
        return {"error": "Invalid test parameters"}
    
    # Select a subset of users to test
    test_users = random.sample(users, min(test_size, len(users)))
    
    # Dictionary to store cumulative metrics
    metrics = {
        "total_users_tested": 0,
        "total_recommendations": 0,
        "avg_recommendation_count": 0,
        "avg_precision_at_5": 0,
        "avg_recall_at_5": 0,
        "avg_diversity": 0,
        "avg_explanation_quality": 0,
        "response_times": []
    }
    
    # Store all recommendations for analysis
    all_recommendations = []
    
    # Creating mock "ground truth" for each test user
    ground_truth = {}
    for user in test_users:
        user_id = user.get("id")
        user_industry = user.get("industry", "")
        user_interests = set(user.get("interests", []))
        user_location = user.get("location", "")
        
        # Find events that match user interests/location/timing
        truth_set = set()
        now = datetime.now()
        for event in events:
            event_id = event.get("id")
            # Check if event is in future
            event_start = datetime.fromisoformat(event.get("start_time").replace("Z", "+00:00")) if isinstance(event.get("start_time"), str) else event.get("start_time")
            
            if event_start and event_start >= now:
                if event.get("location") == user_location:
                    truth_set.add(event_id)
                elif event.get("category") in user_interests:
                    truth_set.add(event_id)
                elif any(topic in user_interests for topic in event.get("topics", [])):
                    truth_set.add(event_id)
        
        ground_truth[user_id] = truth_set
    
    # Get mapping of event attendance
    user_events = {}
    for attendance in event_attendance:
        user_id = attendance.get("user_id")
        event_id = attendance.get("event_id")
        if user_id not in user_events:
            user_events[user_id] = set()
        user_events[user_id].add(event_id)
    
    # Test each user
    for user in test_users:
        user_id = user.get("id")
        
        # Simulate API call
        start_time = time.time()
        try:
            # Simulate event recommendations
            recommendations = simulate_event_recommendations(user, users, events, event_attendance, limit)
            all_recommendations.extend(recommendations)
            
            end_time = time.time()
            response_time = end_time - start_time
            metrics["response_times"].append(response_time)
            
            # Calculate metrics
            recommended_ids = [r.get("id") for r in recommendations]
            p_at_5 = precision_at_k(ground_truth[user_id], recommended_ids, 5)
            r_at_5 = recall_at_k(ground_truth[user_id], recommended_ids, 5)
            div_score = calculate_event_diversity(recommendations)
            exp_quality = explanation_quality(recommendations)
            
            # Update cumulative metrics
            metrics["total_users_tested"] += 1
            metrics["total_recommendations"] += len(recommendations)
            metrics["avg_precision_at_5"] += p_at_5
            metrics["avg_recall_at_5"] += r_at_5
            metrics["avg_diversity"] += div_score
            metrics["avg_explanation_quality"] += exp_quality
            
            print(f"User {user.get('full_name')} (Event Recommendations): P@5={p_at_5:.2f}, R@5={r_at_5:.2f}, Diversity={div_score:.2f}")
            
        except Exception as e:
            print(f"Error testing event recommendations for user {user_id}: {e}")
    
    # Calculate averages
    if metrics["total_users_tested"] > 0:
        metrics["avg_recommendation_count"] = metrics["total_recommendations"] / metrics["total_users_tested"]
        metrics["avg_precision_at_5"] /= metrics["total_users_tested"]
        metrics["avg_recall_at_5"] /= metrics["total_users_tested"]
        metrics["avg_diversity"] /= metrics["total_users_tested"]
        metrics["avg_explanation_quality"] /= metrics["total_users_tested"]
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    # Analyze relevance scores
    if all_recommendations:
        relevance_analysis = analyze_relevance_distribution(all_recommendations, "event")
        metrics["relevance_analysis"] = relevance_analysis
    
    return metrics

def simulate_group_recommendations(
    user: Dict[str, Any],
    all_users: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    group_members: List[Dict[str, Any]],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Simulate the group recommendation algorithm"""
    user_id = user.get("id")
    recommendations = []
    
    # Get user's attributes
    user_industry = user.get("industry", "").lower() if user.get("industry") else ""
    user_location = user.get("location", "").lower() if user.get("location") else ""
    user_interests = set(user.get("interests", []))
    
    # Get groups user has already joined
    user_joined_groups = set()
    for member in group_members:
        if member.get("member_id") == user_id:
            user_joined_groups.add(member.get("group_id"))
    
    # Get user's connections
    user_connections = set()
    for other_user in all_users:
        if other_user.get("id") != user_id and random.random() < 0.2:  # Simulate connections (20% chance)
            user_connections.add(other_user.get("id"))
    
    # Find groups that user's connections have joined
    connection_groups = {}
    for member in group_members:
        if member.get("member_id") in user_connections:
            group_id = member.get("group_id")
            if group_id not in connection_groups:
                connection_groups[group_id] = 0
            connection_groups[group_id] += 1
    
    # Calculate relevance scores for each group
    for group in groups:
        group_id = group.get("id")
        
        # Skip if user has already joined
        if group_id in user_joined_groups:
            continue
        
        # Initial score and matching attributes
        score = 0.0
        matching_attributes = {}
        
        # 1. Category match
        group_category = group.get("category", "").lower() if group.get("category") else ""
        if group_category and any(interest.lower() == group_category for interest in user_interests):
            score += 0.3
            matching_attributes["category"] = group_category
        
        # 2. Interest match (topics)
        group_topics = group.get("topics", [])
        matching_interests = [interest for interest in user_interests if interest.lower() in [topic.lower() for topic in group_topics]]
        
        if matching_interests:
            score += min(len(matching_interests) * 0.1, 0.25)
            matching_attributes["interests"] = matching_interests
        
        # 3. Connection match
        if group_id in connection_groups:
            connection_count = connection_groups[group_id]
            score += min(connection_count * 0.05, 0.25)
            matching_attributes["connection_count"] = connection_count
        
        # 4. Location match
        group_location = group.get("location", "").lower() if group.get("location") else ""
        if user_location and group_location and (
            user_location == group_location or
            user_location in group_location or
            group_location in user_location
        ):
            score += 0.15
            matching_attributes["location"] = group_location
        
        # 5. Industry match
        group_industry = group.get("industry", "").lower() if group.get("industry") else ""
        if user_industry and group_industry and user_industry == group_industry:
            score += 0.05
            matching_attributes["industry"] = group_industry
        
        # Add recommendation if score is high enough
        if score > 0.15:
            # Generate reason
            reason = generate_group_recommendation_reason(matching_attributes)
            
            # Add to recommendations
            recommendations.append({
                **group,
                "relevance_score": score,
                "reason": reason
            })
    
    # Sort by relevance score and return top recommendations
    recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return recommendations[:limit]

def simulate_event_recommendations(
    user: Dict[str, Any],
    all_users: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    event_attendance: List[Dict[str, Any]],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Simulate the event recommendation algorithm"""
    user_id = user.get("id")
    recommendations = []
    
    # Get user's attributes
    user_location = user.get("location", "").lower() if user.get("location") else ""
    user_interests = set(user.get("interests", []))
    
    # Get events user has already registered for
    user_registered_events = set()
    for attendance in event_attendance:
        if attendance.get("user_id") == user_id:
            user_registered_events.add(attendance.get("event_id"))
    
    # Get user's connections
    user_connections = set()
    for other_user in all_users:
        if other_user.get("id") != user_id and random.random() < 0.2:  # Simulate connections (20% chance)
            user_connections.add(other_user.get("id"))
    
    # Find events that user's connections are attending
    connection_events = {}
    for attendance in event_attendance:
        if attendance.get("user_id") in user_connections:
            event_id = attendance.get("event_id")
            if event_id not in connection_events:
                connection_events[event_id] = 0
            connection_events[event_id] += 1
    
    # Simulate past event attendance categories/formats
    user_event_preferences = {
        "event_categories": {},
        "event_formats": {}
    }
    
    # Populate with random preferences
    for category in random.sample(EVENT_CATEGORIES, random.randint(1, 3)):
        user_event_preferences["event_categories"][category] = random.randint(1, 3)
    
    for event_format in random.sample(EVENT_FORMATS, random.randint(1, 2)):
        user_event_preferences["event_formats"][event_format] = random.randint(1, 3)
    
    # Calculate relevance scores for each event
    now = datetime.now()
    for event in events:
        event_id = event.get("id")
        
        # Skip if user has already registered
        if event_id in user_registered_events:
            continue
        
        # Skip past events
        event_start = event.get("start_time")
        if isinstance(event_start, str):
            event_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
        
        if event_start < now:
            continue
        
        # Initial score and matching attributes
        score = 0.0
        matching_attributes = {}
        
        # 1. Category match
        event_category = event.get("category", "").lower() if event.get("category") else ""
        if event_category and event_category in user_event_preferences["event_categories"]:
            category_frequency = user_event_preferences["event_categories"][event_category]
            score += min(category_frequency * 0.08, 0.25)
            matching_attributes["category"] = event_category
        
        # 2. Interest match (topics)
        event_topics = event.get("topics", [])
        matching_interests = [interest for interest in user_interests if interest.lower() in [topic.lower() for topic in event_topics]]
        
        if matching_interests:
            score += min(len(matching_interests) * 0.1, 0.2)
            matching_attributes["interests"] = matching_interests
        
        # 3. Connection match
        if event_id in connection_events:
            connection_count = connection_events[event_id]
            score += min(connection_count * 0.05, 0.2)
            matching_attributes["connection_count"] = connection_count
        
        # 4. Location match
        event_location = event.get("location", "").lower() if event.get("location") else ""
        if user_location and event_location and (
            user_location == event_location or
            user_location in event_location or
            event_location in user_location
        ):
            score += 0.15
            matching_attributes["location"] = event_location
        
        # 5. Format match
        event_format = event.get("format", "").lower() if event.get("format") else ""
        if event_format and event_format in user_event_preferences["event_formats"]:
            format_frequency = user_event_preferences["event_formats"][event_format]
            score += min(format_frequency * 0.03, 0.1)
            matching_attributes["format"] = event_format
        
        # 6. Proximity match
        days_until = (event_start - now).days
        if 0 <= days_until <= 7:
            proximity_score = 0.1 * (1 - (days_until / 7))
            score += proximity_score
            matching_attributes["days_until"] = days_until
        
        # Add recommendation if score is high enough
        if score > 0.15:
            # Generate reason
            reason = generate_event_recommendation_reason(matching_attributes)
            
            # Add to recommendations
            recommendations.append({
                **event,
                "relevance_score": score,
                "reason": reason
            })
    
    # Sort by relevance score and return top recommendations
    recommendations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return recommendations[:limit]

def generate_group_recommendation_reason(matching_attributes: Dict[str, Any]) -> str:
    """Generate a reason for a group recommendation"""
    if not matching_attributes:
        return "This group might be interesting for your professional development."
    
    reasons = []
    
    if "category" in matching_attributes:
        reasons.append(f"This group is about {matching_attributes['category']}, which aligns with your interests")
    
    if "interests" in matching_attributes and matching_attributes["interests"]:
        if len(matching_attributes["interests"]) == 1:
            reasons.append(f"The group focuses on {matching_attributes['interests'][0]}, which you've expressed interest in")
        else:
            interests_text = ", ".join(matching_attributes["interests"][:2])
            if len(matching_attributes["interests"]) > 2:
                interests_text += f", and {len(matching_attributes['interests']) - 2} more topics"
            reasons.append(f"The group covers topics like {interests_text} that match your interests")
    
    if "connection_count" in matching_attributes:
        count = matching_attributes["connection_count"]
        if count == 1:
            reasons.append("One of your connections is a member of this group")
        elif count == 2:
            reasons.append("Two of your connections are members of this group")
        else:
            reasons.append(f"{count} of your connections are members of this group")
    
    if "location" in matching_attributes:
        reasons.append(f"This group is located in {matching_attributes['location']}, matching your location")
    
    if "industry" in matching_attributes:
        reasons.append(f"This group is focused on the {matching_attributes['industry']} industry, which matches your industry")
    
    # Combine reasons into a compelling statement
    if len(reasons) >= 3:
        return f"{reasons[0]}. {reasons[1]}. Additionally, {reasons[2].lower()}"
    elif len(reasons) == 2:
        return f"{reasons[0]}. {reasons[1]}"
    elif reasons:
        return reasons[0]
    else:
        return "This group might be interesting for your professional development."

def generate_event_recommendation_reason(matching_attributes: Dict[str, Any]) -> str:
    """Generate a reason for an event recommendation"""
    if not matching_attributes:
        return "This event might be interesting for your professional development."
    
    reasons = []
    
    if "category" in matching_attributes:
        reasons.append(f"This {matching_attributes['category'].lower()} event aligns with your interests")
    
    if "interests" in matching_attributes and matching_attributes["interests"]:
        if len(matching_attributes["interests"]) == 1:
            reasons.append(f"The event covers {matching_attributes['interests'][0]}, which you've expressed interest in")
        else:
            interests_text = ", ".join(matching_attributes["interests"][:2])
            if len(matching_attributes["interests"]) > 2:
                interests_text += f", and {len(matching_attributes['interests']) - 2} more topics"
            reasons.append(f"The event includes topics like {interests_text} that align with your interests")
    
    if "connection_count" in matching_attributes:
        count = matching_attributes["connection_count"]
        if count == 1:
            reasons.append("One of your connections is attending this event")
        elif count == 2:
            reasons.append("Two of your connections are attending this event")
        else:
            reasons.append(f"{count} of your connections are attending this event, making it a great networking opportunity")
    
    if "location" in matching_attributes:
        reasons.append(f"This event is located in {matching_attributes['location']}, convenient for you to attend")
    
    if "format" in matching_attributes:
        format_name = matching_attributes["format"]
        if format_name == "online":
            reasons.append("This is an online event, which you've shown preference for in the past")
        elif format_name == "in-person":
            reasons.append("This is an in-person event, which you've shown preference for in the past")
        elif format_name == "hybrid":
            reasons.append("This is a hybrid event, offering both in-person and online attendance options")
    
    if "days_until" in matching_attributes:
        days = matching_attributes["days_until"]
        if days == 0:
            reasons.append("This event is happening today!")
        elif days == 1:
            reasons.append("This event is happening tomorrow")
        elif days <= 3:
            reasons.append(f"This event is coming up soon, in just {days} days")
        elif days <= 7:
            reasons.append(f"This event is happening this week, in {days} days")
        else:
            reasons.append(f"This event is happening in {days} days, giving you time to prepare")
    
    # Combine reasons into a compelling statement
    if len(reasons) >= 3:
        return f"{reasons[0]}. {reasons[1]}. Additionally, {reasons[2].lower()}"
    elif len(reasons) == 2:
        return f"{reasons[0]}. {reasons[1]}"
    elif reasons:
        return reasons[0]
    else:
        return "This event might be interesting for your professional development."

def calculate_group_diversity(recommendations: List[Dict[str, Any]]) -> float:
    """Calculate diversity of group recommendations"""
    if not recommendations:
        return 0.0
    
    categories = set(r.get('category', '') for r in recommendations if r.get('category'))
    locations = set(r.get('location', '') for r in recommendations if r.get('location'))
    industries = set(r.get('industry', '') for r in recommendations if r.get('industry'))
    
    # Calculate diversity as the average of unique attributes divided by total recommendations
    category_diversity = len(categories) / len(recommendations) if categories else 0
    location_diversity = len(locations) / len(recommendations) if locations else 0
    industry_diversity = len(industries) / len(recommendations) if industries else 0
    
    return (category_diversity * 0.4 + location_diversity * 0.4 + industry_diversity * 0.2)

def calculate_event_diversity(recommendations: List[Dict[str, Any]]) -> float:
    """Calculate diversity of event recommendations"""
    if not recommendations:
        return 0.0
    
    categories = set(r.get('category', '') for r in recommendations if r.get('category'))
    locations = set(r.get('location', '') for r in recommendations if r.get('location'))
    formats = set(r.get('format', '') for r in recommendations if r.get('format'))
    
    # Get time diversity (spread of event times)
    event_dates = []
    for r in recommendations:
        start_time = r.get('start_time')
        if start_time:
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    event_dates.append(start_time)
                except:
                    pass
            else:
                event_dates.append(start_time)
    
    # Calculate time spread in days
    time_diversity = 0
    if len(event_dates) >= 2:
        min_date = min(event_dates)
        max_date = max(event_dates)
        days_span = (max_date - min_date).days
        # Normalize to 0-1 scale (0 = all same day, 1 = spread over 30+ days)
        time_diversity = min(days_span / 30, 1.0)
    
    # Calculate diversity as weighted average of unique attributes
    category_diversity = len(categories) / len(recommendations) if categories else 0
    location_diversity = len(locations) / len(recommendations) if locations else 0
    format_diversity = len(formats) / len(recommendations) if formats else 0
    
    return (category_diversity * 0.3 + location_diversity * 0.3 + format_diversity * 0.2 + time_diversity * 0.2)

# API functions for testing against real endpoints
async def api_get_user_recommendations(user_id: str, algorithm: str, limit: int = 10) -> Dict[str, Any]:
    """Get user recommendations from the API endpoint"""
    try:
        url = f"{API_BASE_URL}/recommendations/{user_id}?algorithm={algorithm}&limit={limit}"
        response = requests.get(url, headers={"apikey": SUPABASE_KEY})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling user recommendation API: {e}")
        return {"recommendations": [], "count": 0}

async def api_get_group_recommendations(user_id: str, limit: int = 10, exclude_joined: bool = True) -> Dict[str, Any]:
    """Get group recommendations from the API endpoint"""
    try:
        url = f"{API_BASE_URL}/group-recommendations/{user_id}?limit={limit}&exclude_joined={str(exclude_joined).lower()}"
        response = requests.get(url, headers={"apikey": SUPABASE_KEY})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling group recommendation API: {e}")
        return {"group_recommendations": [], "count": 0}

async def api_get_event_recommendations(user_id: str, limit: int = 10, exclude_registered: bool = True) -> Dict[str, Any]:
    """Get event recommendations from the API endpoint"""
    try:
        url = f"{API_BASE_URL}/event-recommendations/{user_id}?limit={limit}&exclude_registered={str(exclude_registered).lower()}"
        response = requests.get(url, headers={"apikey": SUPABASE_KEY})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling event recommendation API: {e}")
        return {"event_recommendations": [], "count": 0}

# Add function to test API-based group recommendations
async def test_api_group_recommendations(users: List[Dict[str, Any]], test_size: int = 5) -> Dict[str, Any]:
    """Test group recommendations using the API endpoint"""
    if not users or test_size <= 0 or not API_BASE_URL:
        return {"error": "Invalid test parameters or missing API_BASE_URL"}
    
    # Select a subset of users to test
    test_users = random.sample(users, min(test_size, len(users)))
    
    # Dictionary to store cumulative metrics
    metrics = {
        "total_users_tested": 0,
        "total_recommendations": 0,
        "avg_recommendation_count": 0,
        "avg_response_time": 0,
        "response_times": []
    }
    
    # Test each user
    for user in test_users:
        user_id = user.get("id")
        
        # Make API call
        start_time = time.time()
        try:
            result = await api_get_group_recommendations(user_id, limit=10)
            recommendations = result.get("group_recommendations", [])
            
            end_time = time.time()
            response_time = end_time - start_time
            metrics["response_times"].append(response_time)
            
            # Update metrics
            metrics["total_users_tested"] += 1
            metrics["total_recommendations"] += len(recommendations)
            
            print(f"User {user.get('full_name')} (API Group Recommendations): Got {len(recommendations)} recommendations in {response_time:.3f}s")
        except Exception as e:
            print(f"Error testing API group recommendations for user {user_id}: {e}")
    
    # Calculate averages
    if metrics["total_users_tested"] > 0:
        metrics["avg_recommendation_count"] = metrics["total_recommendations"] / metrics["total_users_tested"]
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    return metrics

# Add function to test API-based event recommendations
async def test_api_event_recommendations(users: List[Dict[str, Any]], test_size: int = 5) -> Dict[str, Any]:
    """Test event recommendations using the API endpoint"""
    if not users or test_size <= 0 or not API_BASE_URL:
        return {"error": "Invalid test parameters or missing API_BASE_URL"}
    
    # Select a subset of users to test
    test_users = random.sample(users, min(test_size, len(users)))
    
    # Dictionary to store cumulative metrics
    metrics = {
        "total_users_tested": 0,
        "total_recommendations": 0,
        "avg_recommendation_count": 0,
        "avg_response_time": 0,
        "response_times": []
    }
    
    # Test each user
    for user in test_users:
        user_id = user.get("id")
        
        # Make API call
        start_time = time.time()
        try:
            result = await api_get_event_recommendations(user_id, limit=10)
            recommendations = result.get("event_recommendations", [])
            
            end_time = time.time()
            response_time = end_time - start_time
            metrics["response_times"].append(response_time)
            
            # Update metrics
            metrics["total_users_tested"] += 1
            metrics["total_recommendations"] += len(recommendations)
            
            print(f"User {user.get('full_name')} (API Event Recommendations): Got {len(recommendations)} recommendations in {response_time:.3f}s")
        except Exception as e:
            print(f"Error testing API event recommendations for user {user_id}: {e}")
    
    # Calculate averages
    if metrics["total_users_tested"] > 0:
        metrics["avg_recommendation_count"] = metrics["total_recommendations"] / metrics["total_users_tested"]
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    return metrics

async def test_api_user_recommendations(users: List[Dict[str, Any]], algorithm: str, test_size: int = 5) -> Dict[str, Any]:
    """Test user recommendations using the API endpoint"""
    if not users or test_size <= 0 or not API_BASE_URL:
        return {"error": "Invalid test parameters or missing API_BASE_URL"}
    
    # Select a subset of users to test
    test_users = random.sample(users, min(test_size, len(users)))
    
    # Dictionary to store cumulative metrics
    metrics = {
        "total_users_tested": 0,
        "total_recommendations": 0,
        "avg_recommendation_count": 0,
        "avg_response_time": 0,
        "response_times": []
    }
    
    # Test each user
    for user in test_users:
        user_id = user.get("id")
        
        # Make API call
        start_time = time.time()
        try:
            result = await api_get_user_recommendations(user_id, algorithm, limit=10)
            recommendations = result.get("recommendations", [])
            
            end_time = time.time()
            response_time = end_time - start_time
            metrics["response_times"].append(response_time)
            
            # Update metrics
            metrics["total_users_tested"] += 1
            metrics["total_recommendations"] += len(recommendations)
            
            print(f"User {user.get('full_name')} (API {algorithm.upper()}): Got {len(recommendations)} recommendations in {response_time:.3f}s")
        except Exception as e:
            print(f"Error testing API recommendations for user {user_id}: {e}")
    
    # Calculate averages
    if metrics["total_users_tested"] > 0:
        metrics["avg_recommendation_count"] = metrics["total_recommendations"] / metrics["total_users_tested"]
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
    
    return metrics

def print_comparison_table(algorithms_results, metric_names):
    """Print a formatted comparison table for algorithm results"""
    # Define column widths
    algo_width = max(len(algo) for algo in algorithms_results.keys()) + 2
    metric_width = max(len(metric) for metric in metric_names) + 2
    value_width = 10
    
    # Print header
    header = f"{'Metric'.ljust(metric_width)}|"
    for algo in algorithms_results.keys():
        header += f"{algo.upper().ljust(algo_width)}|"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # Print metrics
    for metric in metric_names:
        row = f"{metric.ljust(metric_width)}|"
        for algo, results in algorithms_results.items():
            value = results.get(metric, 0)
            if isinstance(value, float):
                value_str = f"{value:.3f}"
            else:
                value_str = str(value)
            row += f"{value_str.ljust(value_width)}|"
        print(row)
    
    print("-" * len(header))

async def compare_all_recommendation_types(
    users: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    interactions: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    group_members: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    event_attendance: List[Dict[str, Any]],
    test_size: int = 5
) -> Dict[str, Dict[str, Any]]:
    """Compare all recommendation types (user, group, event)"""
    results = {}
    
    # 1. Test user recommendations with auto algorithm
    user_metrics = await test_recommendation_algorithm(
        users, connections, interactions, test_size, "auto"
    )
    results["user"] = user_metrics
    
    # 2. Test group recommendations
    group_metrics = await test_group_recommendation_algorithm(
        users, groups, group_members, test_size
    )
    results["group"] = group_metrics
    
    # 3. Test event recommendations
    event_metrics = await test_event_recommendation_algorithm(
        users, events, event_attendance, test_size
    )
    results["event"] = event_metrics
    
    # Print comparison
    metric_names = [
        "avg_precision_at_5",
        "avg_recall_at_5", 
        "avg_diversity",
        "avg_explanation_quality",
        "avg_recommendation_count",
        "avg_response_time"
    ]
    
    print("\n=== Recommendation Types Comparison ===")
    print_comparison_table(results, metric_names)
    
    return results

def save_test_results(results, filename="recommendation_test_results.json"):
    """Save test results to a file for future comparison"""
    try:
        with open(filename, "w") as f:
            results_with_timestamp = {
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            json.dump(results_with_timestamp, f, indent=2)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def analyze_relevance_distribution(recommendations: List[Dict[str, Any]], entity_type: str = "user") -> Dict[str, Any]:
    """Analyze the distribution of relevance scores in recommendations"""
    if not recommendations:
        return {"error": "No recommendations to analyze"}
    
    # Extract relevance scores
    relevance_scores = [r.get("relevance_score", 0) for r in recommendations]
    
    # Calculate basic statistics
    analysis = {
        "count": len(relevance_scores),
        "min_score": min(relevance_scores) if relevance_scores else 0,
        "max_score": max(relevance_scores) if relevance_scores else 0,
        "avg_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
        "median_score": sorted(relevance_scores)[len(relevance_scores) // 2] if relevance_scores else 0
    }
    
    # Score distribution by range
    ranges = {
        "very_high": [0.8, 1.0],
        "high": [0.6, 0.8],
        "medium": [0.4, 0.6],
        "low": [0.2, 0.4],
        "very_low": [0.0, 0.2]
    }
    
    distribution = {}
    for range_name, (min_val, max_val) in ranges.items():
        count = sum(1 for score in relevance_scores if min_val <= score < max_val)
        distribution[range_name] = {
            "count": count,
            "percentage": (count / len(relevance_scores)) * 100 if relevance_scores else 0
        }
    
    analysis["distribution"] = distribution
    
    # Analyze reason categories
    reason_categories = {}
    for rec in recommendations:
        reason = rec.get("reason", "")
        if reason:
            # Simple categorization based on starting phrases
            category = "other"
            if "interest" in reason.lower():
                category = "interest-based"
            elif "connection" in reason.lower():
                category = "connection-based"
            elif "location" in reason.lower():
                category = "location-based"
            elif "industry" in reason.lower():
                category = "industry-based"
            
            if category not in reason_categories:
                reason_categories[category] = 0
            reason_categories[category] += 1
    
    analysis["reason_categories"] = reason_categories
    
    # Print analysis
    print(f"\n=== {entity_type.title()} Recommendation Relevance Analysis ===")
    print(f"Total recommendations: {analysis['count']}")
    print(f"Score range: {analysis['min_score']:.3f} - {analysis['max_score']:.3f}")
    print(f"Average score: {analysis['avg_score']:.3f}")
    print(f"Median score: {analysis['median_score']:.3f}")
    
    print("\nScore Distribution:")
    for range_name, data in distribution.items():
        print(f"  {range_name.replace('_', ' ').title()}: {data['count']} ({data['percentage']:.1f}%)")
    
    print("\nReason Categories:")
    total_reasons = sum(reason_categories.values())
    for category, count in reason_categories.items():
        percentage = (count / total_reasons) * 100 if total_reasons > 0 else 0
        print(f"  {category.replace('-', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return analysis

# Update main function to test all recommendation types
async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test recommendation algorithms")
    parser.add_argument("--num-users", type=int, default=20, help="Number of test users to generate")
    parser.add_argument("--test-size", type=int, default=5, help="Number of users to test recommendations for")
    parser.add_argument("--algorithms", type=str, default="simple,ml,auto", help="Comma-separated list of algorithms to test")
    parser.add_argument("--rec-types", type=str, default="user,group,event", help="Comma-separated list of recommendation types to test")
    parser.add_argument("--use-api", action="store_true", help="Use actual API endpoints for testing")
    parser.add_argument("--compare-all", action="store_true", help="Run comparison of all recommendation types")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    args = parser.parse_args()
    
    print(f"Generating {args.num_users} test users...")
    users = generate_test_users(args.num_users)
    
    # Generate connections (10% connectivity density)
    connections = generate_test_connections(users, connection_density=0.1)
    
    # Generate interaction history (20% interaction density)
    interactions = generate_test_interactions(users, interaction_density=0.2)
    
    # Generate groups and group memberships
    num_groups = int(args.num_users * 0.7)  # 70% as many groups as users
    print(f"Generating {num_groups} test groups...")
    groups = generate_test_groups(num_groups, users)
    
    # Generate group memberships (10% membership density)
    group_members = generate_test_group_members(groups, users, membership_density=0.1)
    print(f"Generated {len(group_members)} group memberships")
    
    # Generate events
    num_events = int(args.num_users * 0.8)  # 80% as many events as users
    print(f"Generating {num_events} test events...")
    events = generate_test_events(num_events, groups, users)
    
    # Generate event attendance (10% attendance density)
    event_attendance = generate_test_event_attendance(events, users, attendance_density=0.1)
    print(f"Generated {len(event_attendance)} event attendance records")
    
    # Insert test data (simulated)
    await insert_test_data(users, connections, interactions)
    
    # Store all results for potential saving
    all_results = {}
    
    # Run comprehensive comparison if requested
    if args.compare_all:
        print("\n=== Running Comprehensive Comparison of All Recommendation Types ===")
        comparison_results = await compare_all_recommendation_types(
            users, connections, interactions, groups, group_members, 
            events, event_attendance, args.test_size
        )
        all_results["comparison"] = comparison_results
    else:
        # Determine which recommendation types to test
        rec_types = args.rec_types.split(",")
        
        # Test user recommendations if requested
        if "user" in rec_types:
            print("\n=== Testing User Recommendations ===")
            algorithms = args.algorithms.split(",")
            
            if args.use_api:
                print("Using API endpoints for user recommendations testing")
                api_results = {}
                for algorithm in algorithms:
                    # Test each algorithm with the API
                    api_metrics = await test_api_user_recommendations(users, algorithm, args.test_size)
                    api_results[algorithm] = api_metrics
                    print(f"\n=== API User Recommendation Metrics ({algorithm}) ===")
                    print(f"Average recommendations per user: {api_metrics['avg_recommendation_count']:.1f}")
                    print(f"Average response time: {api_metrics['avg_response_time']:.3f}s")
                all_results["api_user"] = api_results
            else:
                if len(algorithms) == 1:
                    # Test a single algorithm
                    alg_metrics = await test_recommendation_algorithm(users, connections, interactions, args.test_size, algorithms[0])
                    all_results["user_" + algorithms[0]] = alg_metrics
                else:
                    # Compare algorithms
                    alg_comparison = await compare_algorithms(users, connections, interactions, args.test_size)
                    all_results["user_comparison"] = alg_comparison
        
        # Test group recommendations if requested
        if "group" in rec_types:
            print("\n=== Testing Group Recommendations ===")
            
            if args.use_api and API_BASE_URL:
                print("Using API endpoints for group recommendations testing")
                api_group_metrics = await test_api_group_recommendations(users, args.test_size)
                
                print("\n=== API Group Recommendation Metrics ===")
                print(f"Average recommendations per user: {api_group_metrics['avg_recommendation_count']:.1f}")
                print(f"Average response time: {api_group_metrics['avg_response_time']:.3f}s")
                all_results["api_group"] = api_group_metrics
            else:
                group_metrics = await test_group_recommendation_algorithm(users, groups, group_members, args.test_size)
                
                print("\n=== Group Recommendation Metrics ===")
                print(f"Average precision@5: {group_metrics['avg_precision_at_5']:.3f}")
                print(f"Average recall@5: {group_metrics['avg_recall_at_5']:.3f}")
                print(f"Average diversity: {group_metrics['avg_diversity']:.3f}")
                print(f"Average explanation quality: {group_metrics['avg_explanation_quality']:.3f}")
                print(f"Average recommendations per user: {group_metrics['avg_recommendation_count']:.1f}")
                print(f"Average response time: {group_metrics['avg_response_time']:.3f}s")
                all_results["group"] = group_metrics
        
        # Test event recommendations if requested
        if "event" in rec_types:
            print("\n=== Testing Event Recommendations ===")
            
            if args.use_api and API_BASE_URL:
                print("Using API endpoints for event recommendations testing")
                api_event_metrics = await test_api_event_recommendations(users, args.test_size)
                
                print("\n=== API Event Recommendation Metrics ===")
                print(f"Average recommendations per user: {api_event_metrics['avg_recommendation_count']:.1f}")
                print(f"Average response time: {api_event_metrics['avg_response_time']:.3f}s")
                all_results["api_event"] = api_event_metrics
            else:
                event_metrics = await test_event_recommendation_algorithm(users, events, event_attendance, args.test_size)
                
                print("\n=== Event Recommendation Metrics ===")
                print(f"Average precision@5: {event_metrics['avg_precision_at_5']:.3f}")
                print(f"Average recall@5: {event_metrics['avg_recall_at_5']:.3f}")
                print(f"Average diversity: {event_metrics['avg_diversity']:.3f}")
                print(f"Average explanation quality: {event_metrics['avg_explanation_quality']:.3f}")
                print(f"Average recommendations per user: {event_metrics['avg_recommendation_count']:.1f}")
                print(f"Average response time: {event_metrics['avg_response_time']:.3f}s")
                all_results["event"] = event_metrics
    
    # Save results if requested
    if args.save_results and all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_test_results(all_results, f"recommendation_test_results_{timestamp}.json")

if __name__ == "__main__":
    asyncio.run(main()) 