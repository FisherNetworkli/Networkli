#!/usr/bin/env python3
"""
Test Data Generator for Networkli Recommendation System

This script generates synthetic test data for:
1. User profiles with various attributes
2. User connections 
3. Interaction history

Usage:
    python generate_test_data.py
"""

import os
import uuid
import random
import json
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any
from supabase import create_client, Client

# Configuration
NUM_USERS = 50  # Number of test users to create
INDUSTRIES = [
    "Software Development", "Data Science", "Product Management", 
    "Marketing", "Design", "Sales", "Finance", "Healthcare",
    "Education", "Consulting", "Human Resources"
]
SKILLS = [
    "Python", "JavaScript", "React", "Node.js", "PostgreSQL", "Machine Learning",
    "Data Analysis", "Product Strategy", "UX Design", "UI Design", 
    "Digital Marketing", "SEO", "Content Marketing", "Sales", "Business Development",
    "Financial Analysis", "Leadership", "Project Management", "Agile", "Scrum",
    "Communication", "Public Speaking", "Writing", "Problem Solving"
]
INTERESTS = [
    "Artificial Intelligence", "Blockchain", "Startups", "Venture Capital",
    "Web Development", "Mobile Development", "Cloud Computing", "Cybersecurity",
    "Sustainability", "Remote Work", "Future of Work", "EdTech", "FinTech",
    "HealthTech", "E-commerce", "Social Media", "Virtual Reality", "Augmented Reality",
    "IoT", "Robotics", "Space Tech", "Green Energy", "Mentorship", "Career Development"
]
COMPANIES = [
    "TechCorp", "DataViz", "InnovateSoft", "CloudScale", "DevGenius",
    "MarketPulse", "DesignMind", "SalesForce", "FinTech Solutions", "HealthCare Plus",
    "LearnTech", "ConsultPro", "HRMasters", "StartupIncubator", "CodeNation"
]
LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", "Boston, MA",
    "Chicago, IL", "Los Angeles, CA", "Denver, CO", "Miami, FL", "Atlanta, GA",
    "Toronto, Canada", "London, UK", "Berlin, Germany", "Paris, France", "Amsterdam, Netherlands",
    "Sydney, Australia", "Singapore", "Tokyo, Japan", "Tel Aviv, Israel", "Bangalore, India"
]
EXPERIENCE_LEVELS = ["Entry", "Mid-Level", "Senior", "Executive"]
INTERACTION_TYPES = [
    "PROFILE_VIEW", "RECOMMENDATION_CLICK", "RECOMMENDATION_LIKE", "RECOMMENDATION_PASS",
    "SEARCH", "MESSAGE_SENT", "CONNECTION_REQUEST"
]
TARGET_ENTITY_TYPES = ["profile", "recommendation", "search", "message", "connection"]

# Initialize Supabase client
def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    
    return create_client(url, key)

def generate_random_profile() -> Dict[str, Any]:
    """Generate a random user profile"""
    first_name = random.choice([
        "Alex", "Jamie", "Taylor", "Jordan", "Casey", "Riley", "Dakota", "Avery",
        "Morgan", "Quinn", "Blake", "Cameron", "Reese", "Emerson", "Parker", "Hayden"
    ])
    last_name = random.choice([
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas"
    ])
    
    # Generate random data
    user_id = str(uuid.uuid4())
    industry = random.choice(INDUSTRIES)
    title = f"{random.choice(['Senior', 'Lead', 'Principal', 'Junior', 'Staff', 'Chief', 'VP of'])} {random.choice(['Engineer', 'Developer', 'Designer', 'Analyst', 'Manager', 'Consultant', 'Specialist', 'Director'])}"
    company = random.choice(COMPANIES)
    location = random.choice(LOCATIONS)
    experience_level = random.choice(EXPERIENCE_LEVELS)
    
    # Generate a random subset of skills and interests
    user_skills = random.sample(SKILLS, random.randint(3, 8))
    user_interests = random.sample(INTERESTS, random.randint(3, 8))
    
    # Create profile
    return {
        "id": user_id,
        "email": f"{first_name.lower()}.{last_name.lower()}@example.com",
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}",
        "title": title,
        "company": company,
        "industry": industry,
        "bio": f"I'm a {experience_level} {industry} professional at {company}, focused on {', '.join(user_skills[:2])}.",
        "location": location,
        "avatar_url": f"https://i.pravatar.cc/150?u={user_id}",
        "experience_level": experience_level,
        "profile_visibility": "public",
        "user_preferences": {
            "skills": user_skills,
            "interests": user_interests
        }
    }

async def generate_and_insert_profiles(supabase: Client, num_users: int) -> List[Dict[str, Any]]:
    """Generate and insert random profiles"""
    profiles = [generate_random_profile() for _ in range(num_users)]
    
    print(f"Generating {num_users} test profiles...")
    
    # Insert profiles
    for profile in profiles:
        user_prefs = profile.pop("user_preferences")
        
        # Insert profile
        try:
            result = await supabase.table("profiles").insert({
                "id": profile["id"],
                "email": profile["email"],
                "first_name": profile["first_name"],
                "last_name": profile["last_name"],
                "full_name": profile["full_name"],
                "title": profile["title"],
                "company": profile["company"],
                "industry": profile["industry"],
                "bio": profile["bio"],
                "location": profile["location"],
                "avatar_url": profile["avatar_url"],
                "experience_level": profile["experience_level"],
                "profile_visibility": profile["profile_visibility"]
            }).execute()
            
            # Insert user preferences
            await supabase.table("user_preferences").insert({
                "user_id": profile["id"],
                "skills": user_prefs["skills"],
                "interests": user_prefs["interests"]
            }).execute()
            
        except Exception as e:
            print(f"Error inserting profile: {e}")
    
    print("Profiles created successfully!")
    return profiles

async def generate_connections(supabase: Client, profiles: List[Dict[str, Any]]):
    """Generate connections between users"""
    print("Generating connections between users...")
    connections = []
    
    # Each user will connect with 5-15 other users
    for profile in profiles:
        num_connections = random.randint(5, min(15, len(profiles) - 1))
        potential_connections = [p for p in profiles if p["id"] != profile["id"]]
        connection_targets = random.sample(potential_connections, num_connections)
        
        for target in connection_targets:
            status = random.choice(["pending", "accepted", "rejected"])
            # Use the smaller ID as requester to avoid duplicates
            if profile["id"] < target["id"]:
                requester_id = profile["id"]
                receiver_id = target["id"]
            else:
                requester_id = target["id"]
                receiver_id = profile["id"]
                
            connections.append({
                "requester_id": requester_id,
                "receiver_id": receiver_id,
                "status": status
            })
    
    # Remove duplicates
    unique_connections = []
    connection_pairs = set()
    
    for conn in connections:
        pair = (conn["requester_id"], conn["receiver_id"])
        if pair not in connection_pairs:
            connection_pairs.add(pair)
            unique_connections.append(conn)
    
    # Insert connections
    for connection in unique_connections:
        try:
            await supabase.table("connections").insert(connection).execute()
        except Exception as e:
            print(f"Error inserting connection: {e}")
    
    print(f"Created {len(unique_connections)} connections successfully!")

async def generate_interaction_history(supabase: Client, profiles: List[Dict[str, Any]]):
    """Generate interaction history for users"""
    print("Generating interaction history...")
    
    # Generate between 10-30 interactions per user
    for profile in profiles:
        num_interactions = random.randint(10, 30)
        
        for _ in range(num_interactions):
            # Pick a random target user
            target_profile = random.choice([p for p in profiles if p["id"] != profile["id"]])
            interaction_type = random.choice(INTERACTION_TYPES)
            target_entity_type = random.choice(TARGET_ENTITY_TYPES)
            
            # Generate a timestamp in the last 30 days
            days_ago = random.randint(0, 30)
            timestamp = datetime.now() - timedelta(days=days_ago, 
                                                 hours=random.randint(0, 23),
                                                 minutes=random.randint(0, 59))
            
            # Generate metadata based on interaction type
            metadata = {}
            if interaction_type == "PROFILE_VIEW":
                metadata = {"duration": random.randint(5, 120), "source": random.choice(["search", "recommendations", "connections"])}
            elif interaction_type == "RECOMMENDATION_CLICK":
                metadata = {"algorithm": random.choice(["simple", "ml", "auto"]), "position": random.randint(1, 10)}
            elif interaction_type == "SEARCH":
                metadata = {"query": random.choice(SKILLS + INTERESTS), "results": random.randint(0, 20)}
            
            # Create interaction
            interaction = {
                "user_id": profile["id"],
                "interaction_type": interaction_type,
                "target_entity_type": target_entity_type,
                "target_entity_id": target_profile["id"],
                "timestamp": timestamp.isoformat(),
                "metadata": metadata
            }
            
            try:
                await supabase.table("interaction_history").insert(interaction).execute()
            except Exception as e:
                print(f"Error inserting interaction: {e}")
    
    print("Interaction history created successfully!")

async def main():
    """Main function to generate all test data"""
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Generate profiles
        profiles = await generate_and_insert_profiles(supabase, NUM_USERS)
        
        # Generate connections
        await generate_connections(supabase, profiles)
        
        # Generate interaction history
        await generate_interaction_history(supabase, profiles)
        
        print("All test data generated successfully!")
        
    except Exception as e:
        print(f"Error generating test data: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 