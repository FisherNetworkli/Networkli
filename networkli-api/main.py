from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Networkli API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ML models
try:
    model_path = "models/"
    graph_sage_model = torch.load(f"{model_path}graphsage.pt")
    maml_model = torch.load(f"{model_path}maml.pt")
    graph_sage_model.eval()
    maml_model.eval()
except Exception as e:
    logger.warning(f"Could not load models: {e}")
    graph_sage_model = None
    maml_model = None

class ProfileCreate(BaseModel):
    username: str
    full_name: str
    role: str
    headline: str
    bio: Optional[str] = None
    location: Optional[str] = None
    industry: str
    company: Optional[str] = None
    years_of_experience: int
    skills: List[str]
    interests: List[str]

class ProfileUpdate(BaseModel):
    headline: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    skills: Optional[List[str]] = None
    interests: Optional[List[str]] = None

@app.post("/profiles/")
async def create_profile(profile: ProfileCreate):
    try:
        # Convert profile data to feature vector
        features = process_profile_to_features(profile)
        
        # Get embedding and cluster using ML models
        with torch.no_grad():
            embedding = graph_sage_model(torch.tensor(features).float().unsqueeze(0))
            cluster_id = maml_model(embedding).argmax().item()
        
        # Store profile in Supabase
        profile_data = profile.dict()
        profile_data["embedding"] = embedding.squeeze().tolist()
        profile_data["cluster_id"] = cluster_id
        profile_data["created_at"] = datetime.utcnow().isoformat()
        
        result = supabase.table("profiles").insert(profile_data).execute()
        
        return {"profile_id": result.data[0]["id"], "cluster_id": cluster_id}
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/{profile_id}")
async def get_recommendations(profile_id: str, limit: int = 10):
    try:
        # Get profile's embedding
        profile = supabase.table("profiles").select("embedding, cluster_id").eq("id", profile_id).single().execute()
        
        if not profile.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        embedding = torch.tensor(profile.data["embedding"])
        cluster_id = profile.data["cluster_id"]
        
        # Find similar profiles in the same cluster
        similar_profiles = supabase.table("profiles")\
            .select("id, username, full_name, role, headline, company, embedding")\
            .eq("cluster_id", cluster_id)\
            .neq("id", profile_id)\
            .execute()
        
        # Calculate similarity scores
        recommendations = []
        for p in similar_profiles.data:
            other_embedding = torch.tensor(p["embedding"])
            similarity = torch.cosine_similarity(embedding.unsqueeze(0), other_embedding.unsqueeze(0)).item()
            recommendations.append({**p, "similarity_score": similarity})
        
        # Sort by similarity and return top matches
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return recommendations[:limit]
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/connections/")
async def create_connection(requester_id: str, receiver_id: str, message: Optional[str] = None):
    try:
        connection_data = {
            "requester_id": requester_id,
            "receiver_id": receiver_id,
            "status": "pending",
            "message": message,
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("connections").insert(connection_data).execute()
        return result.data[0]
    
    except Exception as e:
        logger.error(f"Error creating connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/connections/{connection_id}")
async def update_connection_status(connection_id: str, status: str):
    try:
        if status not in ["accepted", "rejected"]:
            raise HTTPException(status_code=400, detail="Invalid status")
        
        result = supabase.table("connections")\
            .update({"status": status, "updated_at": datetime.utcnow().isoformat()})\
            .eq("id", connection_id)\
            .execute()
            
        return result.data[0]
    
    except Exception as e:
        logger.error(f"Error updating connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def process_profile_to_features(profile: ProfileCreate) -> np.ndarray:
    """Convert profile data to feature vector for ML models."""
    # This is a simplified version - you should implement proper feature engineering
    features = []
    
    # Add numerical features
    features.append(profile.years_of_experience)
    
    # Add categorical features (one-hot encoded)
    role_categories = ["professional", "entrepreneur", "student", "mentor"]
    role_vector = [1 if profile.role == role else 0 for role in role_categories]
    features.extend(role_vector)
    
    # Add skill features (you should use a proper skill taxonomy)
    common_skills = ["python", "javascript", "react", "machine_learning", "design"]
    skill_vector = [1 if skill.lower() in [s.lower() for s in profile.skills] else 0 for skill in common_skills]
    features.extend(skill_vector)
    
    return np.array(features)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 