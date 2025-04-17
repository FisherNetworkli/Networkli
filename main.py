#!/usr/bin/env python
"""
Networkli API Main Application

FastAPI application entry point for the Networkli backend.
Handles user profiles, connections, recommendations, interactions, and more.
"""

import os
import json
import random
import time
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple, Callable, Union, TypeVar
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from supabase import Client, create_client
from sentence_transformers import SentenceTransformer
import asyncio

# --- Core Application Imports ---
from app.ml.recommendation import create_recommendation_engine # Use the consolidated engine
from settings import settings # Import settings from root settings.py

# --- New Services Imports ---
from app.ml.services.group_recommendation import create_group_recommendation_service
from app.ml.services.event_recommendation import create_event_recommendation_service

# --- Import the new alignment module
from app.ml.alignment import create_member_alignment_service

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add file handler if needed
# file_handler = logging.FileHandler('networkli-api.log')
# ... configure handler ...
# logger.addHandler(file_handler)

logger.info("--- Networkli API Starting Up ---")

# --- Supabase Client Initialization ---
supabase: Optional[Client] = None
if settings:
    try:
        logger.info(f"Initializing Supabase client with URL: {settings.SUPABASE_URL}")
        # Prefer service role key if available, fall back to regular key
        api_key = settings.SUPABASE_SERVICE_ROLE_KEY if settings.SUPABASE_SERVICE_ROLE_KEY else settings.SUPABASE_KEY
        key_type = 'service role' if settings.SUPABASE_SERVICE_ROLE_KEY else 'regular'
        logger.info(f"Using {key_type} API key (Length: {len(api_key)})")
        
        # Log the beginning of the key for debugging (careful with sensitive info)
        if len(api_key) > 8:
            logger.debug(f"API key starts with: {api_key[:4]}...{api_key[-4:]}")
            
        supabase = create_client(settings.SUPABASE_URL, api_key)

        # Test connection with more detailed error reporting
        try:
            test_result = supabase.table("profiles").select("count", count="exact").limit(1).execute()
            if test_result.error:
                logger.error(f"Supabase connection test failed: {test_result.error}")
            else:
                logger.info("Supabase client initialized and connection test passed.")
        except Exception as conn_err:
            logger.error(f"Could not test Supabase connectivity during startup: {conn_err}", exc_info=True)
            # Try to get more specific error information
            if "invalid api key" in str(conn_err).lower():
                logger.critical("API key appears to be invalid. Check your SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY.")
            elif "network" in str(conn_err).lower():
                logger.critical(f"Network error connecting to Supabase at {settings.SUPABASE_URL}. Check connectivity.")

    except Exception as e:
        logger.critical(f"FATAL: Failed to create Supabase client: {e}", exc_info=True)
elif not settings:
    logger.critical("FATAL: Cannot initialize Supabase client because settings failed to load.")

# --- Embedding Model Initialization ---
embedding_model = None
try:
    # Specify cache folder if needed, e.g., cache_folder='/path/to/cache'
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}", exc_info=True)

# --- FastAPI App Setup ---
app = FastAPI(
    title="Networkli API",
    description="Backend API for the Networkli professional networking platform.",
    version="1.0.0"
)

# Configure CORS (adjust origins as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], # Add other origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Domain ID Mapping --- (Move to config or DB later?)
INDUSTRY_TO_DOMAIN_ID = {
    "technology": 1, "software development": 1, "information technology": 1,
    "finance": 2, "banking": 2, "investment": 2,
    "healthcare": 3, "medical": 3, "pharmaceuticals": 3,
    "education": 4,
    "marketing": 5, "advertising": 5,
    "sales": 6,
    "design": 7, "ux/ui": 7,
    "human resources": 8, "recruiting": 8,
}
DEFAULT_DOMAIN_ID = 0

def get_domain_id(industry: Optional[str]) -> Optional[int]:
    if not industry: return DEFAULT_DOMAIN_ID
    return INDUSTRY_TO_DOMAIN_ID.get(industry.lower().strip(), DEFAULT_DOMAIN_ID)

# --- Pydantic Models ---
# (Keep core models like ProfileCreate, ProfileUpdate, etc. here)
# (Models specific to features like events, groups can be in separate files later)

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
    expertise: Optional[str] = None
    needs: Optional[str] = None
    meaningful_goals: Optional[str] = None

class ProfileUpdate(BaseModel):
    headline: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    skills: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    expertise: Optional[str] = None
    needs: Optional[str] = None
    meaningful_goals: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    twitter_url: Optional[str] = None
    avatar_url: Optional[str] = None
    is_premium: Optional[bool] = None # Allow updating premium status
    # Add other fields as needed

class ProfileLimitedView(BaseModel):
    id: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    limited_view: bool = True

class ProfileViewEntry(BaseModel):
    viewer_id: str
    viewed_at: datetime
    viewer_full_name: Optional[str] = None
    viewer_avatar_url: Optional[str] = None
    source: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    type: str = "USER"
    filters: Optional[Dict[str, Any]] = None

class RecommendationClickInfo(BaseModel):
    recommendation_id: str
    source_page: str
    algorithm: Optional[str] = None
    position: Optional[int] = None

class ProfileRecommendationClick(BaseModel):
    profile_id: str
    source_page: str
    rank: Optional[int] = None
    section: Optional[str] = None
    algorithm_version: Optional[str] = None

class RecommendationViewData(BaseModel):
    profile_id: str
    recommendation_id: Optional[str] = None
    source_page: str
    section: Optional[str] = None
    time_to_view: Optional[int] = None

class EventInteraction(BaseModel):
    event_id: str
    status: Optional[str] = None

class GroupInteraction(BaseModel):
    group_id: str

class DashboardResponse(BaseModel):
    # Define dashboard structure here
    profile_views: List[ProfileViewEntry]
    profile_view_count: int
    unique_viewers_count: int
    recent_views_count: int
    view_sources: Dict[str, int]
    last_updated: datetime
    # Add other dashboard elements

class SwipeInteraction(BaseModel):
    profile_id: str
    direction: str = Field(..., description="Direction of swipe: 'like' or 'pass'")
    source: Optional[str] = Field(None, description="Source section: 'recommendations', 'search', etc.")
    recommendation_id: Optional[str] = None
    algorithm_version: Optional[str] = None

# --- Helper Functions ---

def generate_embedding(text: Optional[str], field_name: str, identifier: str) -> Optional[List[float]]:
    """Safely generate embedding for a given text field."""
    if not text or not text.strip():
        logger.warning(f"Empty or null text for field '{field_name}' for {identifier}. Skipping embedding generation.")
        return None
    if embedding_model is None:
        logger.warning(f"Embedding model not loaded. Cannot generate embedding for '{field_name}' for {identifier}.")
        return None
    try:
        embedding = embedding_model.encode(text)
        logger.info(f"Successfully generated embedding for '{field_name}' for {identifier}")
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    except Exception as e:
        logger.error(f"Error generating embedding for '{field_name}' for {identifier}: {e}", exc_info=True)
        return None

async def get_current_user(request: Request) -> Optional[str]:
    """Placeholder for getting user ID from request (e.g., JWT token)."""
    # In a real app, you would decode a JWT token from the Authorization header
    # For now, let's assume a header `X-User-ID` is passed for testing
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        # Try fetching from auth token if available (example)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                if supabase and settings.SUPABASE_JWT_SECRET:
                    # This assumes supabase client can validate JWTs
                    # user_data = supabase.auth.get_user(token)
                    # user_id = user_data.user.id if user_data.user else None
                    # Replace with actual Supabase JWT validation logic if needed
                    pass # Placeholder
                else:
                    logger.warning("Supabase client or JWT secret not configured for token validation.")
            except Exception as e:
                logger.warning(f"Token validation failed: {e}")
                return None
    return user_id

async def log_interaction(
    supabase_client: Client, # Ensure client is passed correctly
    user_id: str,
    interaction_type: str,
    target_entity_type: Optional[str] = None,
    target_entity_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Helper function to log user interactions asynchronously."""
    if not supabase_client:
        logger.error("Supabase client not available, cannot log interaction.")
        return
    try:
        interaction_data = {
            "user_id": user_id,
            "interaction_type": interaction_type,
            "target_entity_type": target_entity_type,
            "target_entity_id": target_entity_id,
            "metadata": metadata,
            # timestamp/created_at have defaults in DB
        }
        # Ensure keys with None values are handled or omitted if the DB expects it
        interaction_data = {k: v for k, v in interaction_data.items() if v is not None}

        # Check if interaction_history table exists before logging
        # (Consider doing this check once at startup)
        # if not await table_exists(supabase_client, "interaction_history"):
        #     logger.warning("Interaction history table not found. Skipping log.")
        #     return

        result = await supabase_client.table("interaction_history").insert(interaction_data).execute()
        if result.error:
            logger.error(f"Failed to log interaction {interaction_type} for user {user_id}: {result.error}")
        else:
            logger.info(f"Logged interaction: {interaction_type} for user {user_id}")
    except Exception as e:
        logger.error(f"Exception logging interaction {interaction_type} for user {user_id}: {e}", exc_info=True)

# --- Database Check Helpers ---
async def table_exists(client: Client, table_name: str, schema: str = "public") -> bool:
    """Check if a table exists in the database."""
    if not client:
        return False
    try:
        # Query information_schema is generally reliable
        res = await client.rpc(
            'table_exists', # Assuming a DB function `table_exists` exists
            {'table_name': table_name, 'table_schema': schema}
        ).execute()
        if res.error:
            logger.warning(f"Error checking if table {schema}.{table_name} exists (RPC failed): {res.error}")
            # Fallback: Try a select query
            try:
                select_res = await client.from_(f"{schema}.{table_name}").select("count", count="exact").limit(1).execute()
                return select_res.error is None
            except Exception as select_err:
                logger.warning(f"Error checking if table {schema}.{table_name} exists (SELECT failed): {select_err}")
                return False
        return res.data == True
    except Exception as e:
        logger.error(f"Exception checking if table {schema}.{table_name} exists: {e}")
        return False

# --- Startup Event --- (Example: Check required tables)
@app.on_event("startup")
async def startup_db_check():
    logger.info("Running database checks on startup...")
    required_tables = ["profiles", "connections", "interaction_history"]
    if supabase:
        for table in required_tables:
            exists = await table_exists(supabase, table)
            if not exists:
                logger.error(f"CRITICAL: Required table '{table}' does not exist in the database!")
            else:
                logger.info(f"Table '{table}' confirmed.")
    else:
        logger.error("Supabase client not initialized, skipping startup DB checks.")

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    db_status = "ok" if supabase else "unavailable"
    db_connected = False
    if supabase:
        try:
            res = await supabase.from_("profiles").select("id").limit(1).execute()
            db_connected = res.error is None
        except Exception:
            db_connected = False
    return {
        "status": "ok",
        "database_client": db_status,
        "database_connected": db_connected,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/env-check")
async def env_check():
    """Endpoint to check environment variables and basic connectivity (for debugging)."""
    # Limit exposure in a real environment
    env_vars = {
        "SUPABASE_URL_Set": "Yes" if settings and settings.SUPABASE_URL else "No",
        "SUPABASE_KEY_Set": "Yes" if settings and settings.SUPABASE_KEY else "No",
        "SUPABASE_JWT_SECRET_Set": "Yes" if settings and settings.SUPABASE_JWT_SECRET else "No",
        "SUPABASE_SERVICE_ROLE_KEY_Set": "Yes" if settings and settings.SUPABASE_SERVICE_ROLE_KEY else "No",
    }
    db_status = {"client_initialized": supabase is not None, "connected": False, "checked_tables": {}}
    if supabase:
        try:
            # Test basic connectivity
            res = await supabase.from_("profiles").select("id").limit(1).execute()
            db_status["connected"] = res.error is None
            # Check required tables
            required = ["profiles", "connections", "interaction_history"]
            for t in required:
                db_status["checked_tables"][t] = await table_exists(supabase, t)
        except Exception as e:
            db_status["connection_error"] = str(e)

    return {
        "environment_variables_status": env_vars,
        "database_status": db_status,
        "timestamp": datetime.now().isoformat()
    }

# --- Profile Endpoints ---
@app.post("/profiles/", status_code=201)
async def create_profile(profile: ProfileCreate):
    """Create a new user profile."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        profile_data = profile.dict()
        profile_data["created_at"] = datetime.now().isoformat()
        profile_data["updated_at"] = profile_data["created_at"]
        profile_data["domain_id"] = get_domain_id(profile.industry)

        # Generate embeddings
        profile_data["bio_embedding"] = generate_embedding(profile.bio, "bio", profile.username)
        profile_data["expertise_embedding"] = generate_embedding(profile.expertise, "expertise", profile.username)
        profile_data["needs_embedding"] = generate_embedding(profile.needs, "needs", profile.username)
        profile_data["goals_embedding"] = generate_embedding(profile.meaningful_goals, "goals", profile.username)

        # Remove embedding fields if they are None (if DB requires non-null or specific type)
        embedding_fields = ["bio_embedding", "expertise_embedding", "needs_embedding", "goals_embedding"]
        for field in embedding_fields:
            if profile_data[field] is None:
                del profile_data[field] # Or set to default DB value if applicable

        result = await supabase.table("profiles").insert(profile_data).execute()

        if result.error:
            logger.error(f"Failed to insert profile for {profile.username}: {result.error}")
            # Check for specific errors like unique constraint violation
            if "duplicate key value violates unique constraint" in str(result.error):
                 raise HTTPException(status_code=409, detail=f"Username '{profile.username}' already exists.")
            raise HTTPException(status_code=500, detail=f"Database error: {result.error.message}")

        if not result.data:
            logger.error(f"Profile insert for {profile.username} succeeded but returned no data.")
            raise HTTPException(status_code=500, detail="Profile created but failed to retrieve details.")

        created_profile_id = result.data[0].get("id")
        if not created_profile_id:
             logger.error(f"Profile inserted for {profile.username} but no ID returned. Response: {result.data}")
             raise HTTPException(status_code=500, detail="Profile created but failed to retrieve ID.")

        return {"profile_id": created_profile_id, "username": profile.username}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error creating profile for {profile.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.patch("/profiles/{profile_id}")
async def update_profile(profile_id: str, update_data: ProfileUpdate, request: Request):
    """Update an existing profile."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    current_user_id = await get_current_user(request)
    # Add authorization: Only allow user to update their own profile, or admins
    # user_role = await get_user_role(current_user_id) # You'd need a get_user_role helper
    # if profile_id != current_user_id and user_role != 'admin':
    #     raise HTTPException(status_code=403, detail="Not authorized to update this profile")

    try:
        profile_check = await supabase.table("profiles").select("id, bio, expertise, needs, meaningful_goals").eq("id", profile_id).maybe_single().execute()
        if not profile_check.data:
            raise HTTPException(status_code=404, detail="Profile not found")

        existing_profile = profile_check.data
        update_dict = update_data.dict(exclude_unset=True)

        # Generate embeddings only if relevant text fields changed
        embeddings_to_update = {}
        text_fields = {"bio": "bio", "expertise": "expertise", "needs": "needs", "meaningful_goals": "goals"}
        for field_key, embedding_name_part in text_fields.items():
            if field_key in update_dict:
                # Check if the field has been updated or is empty
                new_value = update_dict[field_key]
                old_value = existing_profile.get(field_key)
                
                if new_value != old_value:
                    logger.info(f"Field '{field_key}' updated for profile {profile_id}. Regenerating embedding.")
                    
                    if new_value and new_value.strip():
                        # Generate embedding for non-empty field
                        embedding = generate_embedding(new_value, embedding_name_part, profile_id)
                        if embedding is not None:
                            embeddings_to_update[f"{embedding_name_part}_embedding"] = embedding
                            logger.info(f"Successfully updated {embedding_name_part}_embedding for profile {profile_id}")
                    else:
                        # Field is empty, set embedding to NULL
                        embeddings_to_update[f"{embedding_name_part}_embedding"] = None
                        logger.info(f"Setting {embedding_name_part}_embedding to NULL for profile {profile_id} (empty text)")

        if embeddings_to_update:
            update_dict.update(embeddings_to_update)

        # Add updated_at timestamp
        update_dict["updated_at"] = datetime.now().isoformat()

        # Update the profile
        result = await supabase.table("profiles").update(update_dict).eq("id", profile_id).execute()

        if result.error:
            logger.error(f"Failed to update profile {profile_id}: {result.error}")
            raise HTTPException(status_code=500, detail=f"Database error: {result.error.message}")

        if not result.data:
             logger.warning(f"Profile update for {profile_id} executed but returned no data.")
             # Fetch the updated profile to return it
             updated_profile_res = await supabase.table("profiles").select("*").eq("id", profile_id).maybe_single().execute()
             if updated_profile_res.data:
                 return updated_profile_res.data
             else:
                 raise HTTPException(status_code=500, detail="Failed to update profile or retrieve updated data.")

        return result.data[0]

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error updating profile {profile_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/profiles/{profile_id}", response_model=Union[ProfileLimitedView, Dict[str, Any]])
async def get_profile(
    profile_id: str,
    request: Request,
    source: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Fetches a single user profile. Access control determines full/limited view."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # --- Get Target Profile --- 
        target_profile_res = await supabase.table("profiles").select(
            "id, email, first_name, last_name, full_name, avatar_url, title, company, industry, bio, location, website, role, linkedin_url, github_url, portfolio_url, twitter_url, expertise, needs, meaningful_goals, is_premium, created_at"
        ).eq("id", profile_id).maybe_single().execute()

        if not target_profile_res.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        profile_data = target_profile_res.data

        # --- Access Control --- 
        current_user_id: Optional[str] = await get_current_user(request)
        can_view_full_profile = False

        if current_user_id == profile_id:
            can_view_full_profile = True # View own profile
        elif current_user_id:
            # Authenticated user viewing someone else - check roles/premium
            viewer_profile_res = await supabase.table("profiles").select("role, is_premium").eq("id", current_user_id).maybe_single().execute()
            if viewer_profile_res.data:
                viewer_role = viewer_profile_res.data.get("role")
                viewer_is_premium = viewer_profile_res.data.get("is_premium", False)
                allowed_roles = {"admin", "organizer"}
                if viewer_role in allowed_roles or viewer_is_premium:
                    can_view_full_profile = True
            else:
                logger.warning(f"Viewer profile {current_user_id} not found for access check.")
        # else: Anonymous user - defaults to can_view_full_profile = False

        # --- Log Interaction (Full View Only) --- 
        if can_view_full_profile and current_user_id and current_user_id != profile_id:
            metadata = {"source": source} if source else None
            background_tasks.add_task(
                log_interaction,
                supabase_client=supabase, # Pass the client
                user_id=current_user_id,
                interaction_type='PROFILE_VIEW',
                target_entity_type='USER',
                target_entity_id=profile_id,
                metadata=metadata
            )
            # Consider also calling the DB function `record_profile_view` if it exists

        # --- Return Profile Data --- 
        if can_view_full_profile:
            return profile_data
        else:
            # Return limited view
            return ProfileLimitedView(
                id=profile_data["id"],
                full_name=profile_data.get("full_name"),
                avatar_url=profile_data.get("avatar_url"),
                title=profile_data.get("title"),
                company=profile_data.get("company"),
                industry=profile_data.get("industry"),
                location=profile_data.get("location"),
            )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error fetching profile {profile_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Recommendation Endpoint ---
@app.get("/recommendations/{profile_id}")
async def get_recommendations(
    profile_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    limit: int = Query(10, ge=1, le=50),
    exclude_connected: bool = True,
    include_reason: bool = True,
    algorithm: str = Query("auto", enum=["simple", "ml", "auto"])
):
    """Get personalized user recommendations."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    current_user_id = await get_current_user(request)
    # Optional: Check if the requestor matches the profile_id or is admin
    # if current_user_id != profile_id and not is_admin(current_user_id):
    #    raise HTTPException(status_code=403, detail="Forbidden")

    if current_user_id:
        background_tasks.add_task(
            log_interaction,
            supabase_client=supabase,
            user_id=current_user_id,
            interaction_type="VIEW_RECOMMENDATIONS",
            target_entity_type="RECOMMENDATIONS",
            target_entity_id=profile_id,
            metadata={"limit": limit, "exclude_connected": exclude_connected, "algorithm": algorithm}
        )

    try:
        recommendation_engine = create_recommendation_engine(supabase)
        recommendations_result = await recommendation_engine.get_recommendations(
            user_id=profile_id,
            algorithm=algorithm,
            limit=limit,
            exclude_connected=exclude_connected,
            include_reason=include_reason
        )
        return recommendations_result
    except Exception as e:
        logger.error(f"Error getting recommendations for {profile_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations")

# --- Interaction Logging Endpoints ---
@app.post("/interactions/swipe")
async def log_swipe_interaction(
    swipe_data: SwipeInteraction,
    request: Request,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Log a swipe interaction (like/pass) on a profile."""
    current_user_id = await get_current_user(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    interaction_type = f"SWIPE_{swipe_data.direction.upper()}"
    
    metadata = {
        "source": swipe_data.source,
        "recommendation_id": swipe_data.recommendation_id,
        "algorithm_version": swipe_data.algorithm_version
    }
    # Filter out None values
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    background_tasks.add_task(
        log_interaction,
        supabase_client=supabase,
        user_id=current_user_id,
        interaction_type=interaction_type,
        target_entity_type="USER",
        target_entity_id=swipe_data.profile_id,
        metadata=metadata
    )
    
    # If swipe was a "like", see if we should create a connection
    if swipe_data.direction.lower() == "like":
        # Check if the swiped user has already liked the current user
        try:
            # Look for a previous "like" interaction from the other user to this user
            like_query = supabase.table("interaction_history").select("*") \
                .eq("user_id", swipe_data.profile_id) \
                .eq("target_entity_id", current_user_id) \
                .eq("interaction_type", "SWIPE_LIKE") \
                .execute()
                
            if not like_query.error and like_query.data and len(like_query.data) > 0:
                # Match found! Create a connection
                logger.info(f"Match found between {current_user_id} and {swipe_data.profile_id}. Creating connection.")
                
                # Log the match
                background_tasks.add_task(
                    log_interaction,
                    supabase_client=supabase,
                    user_id=current_user_id,
                    interaction_type="MATCH_CREATED",
                    target_entity_type="USER",
                    target_entity_id=swipe_data.profile_id,
                    metadata={"matched_at": datetime.now().isoformat()}
                )
                
                # TODO: Create the actual connection record
                # This would insert into the connections table
        except Exception as e:
            logger.error(f"Error checking for mutual like: {e}", exc_info=True)
    
    return {"success": True, "message": f"Swipe {swipe_data.direction} recorded successfully"}

@app.post("/interactions/log")
async def log_generic_interaction(
    interaction_type: str,
    request: Request,
    target_entity_type: Optional[str] = None,
    target_entity_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generic endpoint to log various user interactions."""
    current_user_id = await get_current_user(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    background_tasks.add_task(
        log_interaction,
        supabase_client=supabase,
        user_id=current_user_id,
        interaction_type=interaction_type.upper(), # Standardize type
        target_entity_type=target_entity_type,
        target_entity_id=target_entity_id,
        metadata=metadata
    )
    return {"message": "Interaction logged successfully"}

@app.post("/recommendations/click")
async def record_recommendation_click(
    click_info: RecommendationClickInfo,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Records when a recommendation is clicked."""
    current_user_id = await get_current_user(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    metadata = {
        "recommendation_id": click_info.recommendation_id,
        "source_page": click_info.source_page,
        "algorithm": click_info.algorithm,
        "position": click_info.position
    }
    background_tasks.add_task(
        log_interaction,
        supabase_client=supabase,
        user_id=current_user_id,
        interaction_type="RECOMMENDATION_CLICK",
        target_entity_type="RECOMMENDATION", # Or maybe the target user ID?
        target_entity_id=click_info.recommendation_id,
        metadata=metadata
    )
    return {"success": True, "message": "Recommendation click recorded"}

# Add other endpoints (connections, dashboard, etc.) as needed, 
# potentially refactoring them into separate routers/files for organization.

# --- Example: Connection Endpoints ---
@app.post("/connections/")
async def create_connection(request: Request, receiver_id: str = Body(...), message: Optional[str] = Body(None)):
    # Implementation...
    logger.info(f"Placeholder: Create connection request to {receiver_id}")
    return {"message": "Connection creation endpoint not fully implemented"}

@app.put("/connections/{connection_id}")
async def update_connection_status(connection_id: str, request: Request, status: str = Body(..., enum=["accepted", "rejected"])): 
    # Implementation...
    logger.info(f"Placeholder: Update connection {connection_id} to status {status}")
    return {"message": "Connection status update endpoint not fully implemented"}

# --- Group Recommendation Endpoint ---
@app.get("/group-recommendations/{profile_id}")
async def get_group_recommendations(
    profile_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    limit: int = Query(10, ge=1, le=50),
    exclude_joined: bool = True,
    include_reason: bool = True
):
    """Get personalized group recommendations."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    current_user_id = await get_current_user(request)
    # Optional: Check if the requestor matches the profile_id or is admin
    # if current_user_id != profile_id and not is_admin(current_user_id):
    #    raise HTTPException(status_code=403, detail="Forbidden")

    if current_user_id:
        background_tasks.add_task(
            log_interaction,
            supabase_client=supabase,
            user_id=current_user_id,
            interaction_type="VIEW_GROUP_RECOMMENDATIONS",
            target_entity_type="GROUP_RECOMMENDATIONS",
            target_entity_id=profile_id,
            metadata={"limit": limit, "exclude_joined": exclude_joined}
        )

    try:
        group_recommendation_service = create_group_recommendation_service(supabase)
        recommendations_result = await group_recommendation_service.get_group_recommendations(
            user_id=profile_id,
            limit=limit,
            exclude_joined=exclude_joined,
            include_reason=include_reason
        )
        return {
            "group_recommendations": recommendations_result,
            "count": len(recommendations_result)
        }
    except Exception as e:
        logger.error(f"Error getting group recommendations for {profile_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve group recommendations")

# --- Event Recommendation Endpoint ---
@app.get("/event-recommendations/{profile_id}")
async def get_event_recommendations(
    profile_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    limit: int = Query(10, ge=1, le=50),
    exclude_registered: bool = True,
    include_reason: bool = True,
    include_past_events: bool = False
):
    """Get personalized event recommendations."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    current_user_id = await get_current_user(request)
    # Optional: Check if the requestor matches the profile_id or is admin
    # if current_user_id != profile_id and not is_admin(current_user_id):
    #    raise HTTPException(status_code=403, detail="Forbidden")

    if current_user_id:
        background_tasks.add_task(
            log_interaction,
            supabase_client=supabase,
            user_id=current_user_id,
            interaction_type="VIEW_EVENT_RECOMMENDATIONS",
            target_entity_type="EVENT_RECOMMENDATIONS",
            target_entity_id=profile_id,
            metadata={
                "limit": limit, 
                "exclude_registered": exclude_registered,
                "include_past_events": include_past_events
            }
        )

    try:
        event_recommendation_service = create_event_recommendation_service(supabase)
        recommendations_result = await event_recommendation_service.get_event_recommendations(
            user_id=profile_id,
            limit=limit,
            exclude_registered=exclude_registered,
            include_reason=include_reason,
            include_past_events=include_past_events
        )
        return {
            "event_recommendations": recommendations_result,
            "count": len(recommendations_result)
        }
    except Exception as e:
        logger.error(f"Error getting event recommendations for {profile_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve event recommendations")

# --- Group Recommendation Click Endpoint ---
@app.post("/group-recommendations/click")
async def record_group_recommendation_click(
    click_info: RecommendationClickInfo,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Records when a group recommendation is clicked."""
    current_user_id = await get_current_user(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    metadata = {
        "group_id": click_info.recommendation_id,
        "source_page": click_info.source_page,
        "position": click_info.position
    }
    background_tasks.add_task(
        log_interaction,
        supabase_client=supabase,
        user_id=current_user_id,
        interaction_type="GROUP_RECOMMENDATION_CLICK",
        target_entity_type="GROUP",
        target_entity_id=click_info.recommendation_id,
        metadata=metadata
    )
    return {"success": True, "message": "Group recommendation click recorded"}

# --- Event Recommendation Click Endpoint ---
@app.post("/event-recommendations/click")
async def record_event_recommendation_click(
    click_info: RecommendationClickInfo,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Records when an event recommendation is clicked."""
    current_user_id = await get_current_user(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    metadata = {
        "event_id": click_info.recommendation_id,
        "source_page": click_info.source_page,
        "position": click_info.position
    }
    background_tasks.add_task(
        log_interaction,
        supabase_client=supabase,
        user_id=current_user_id,
        interaction_type="EVENT_RECOMMENDATION_CLICK",
        target_entity_type="EVENT",
        target_entity_id=click_info.recommendation_id,
        metadata=metadata
    )
    return {"success": True, "message": "Event recommendation click recorded"}

# --- Member Alignment Endpoint ---
@app.get("/member-alignment/{user_id}")
async def get_member_alignment(
    user_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    entity_type: str = Query(..., regex="^(group|event)$"),
    entity_id: str = Query(...),
    limit: int = Query(5, ge=1, le=20),
    min_similarity: float = Query(0.1, ge=0.0, le=1.0)
):
    """
    Get alignment/similarity between a user and other members of a group or event.
    
    This helps users find potential connections when they join a group or event.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    current_user_id = await get_current_user(request)
    
    # Optional access control
    # if current_user_id != user_id and not is_admin(current_user_id):
    #    raise HTTPException(status_code=403, detail="Forbidden")

    if current_user_id:
        background_tasks.add_task(
            log_interaction,
            supabase_client=supabase,
            user_id=current_user_id,
            interaction_type="VIEW_MEMBER_ALIGNMENT",
            target_entity_type=entity_type.upper(),
            target_entity_id=entity_id,
            metadata={
                "user_id": user_id,
                "limit": limit,
                "min_similarity": min_similarity
            }
        )

    try:
        # Check if user is a member of the entity
        is_member = await check_entity_membership(user_id, entity_type, entity_id)
        if not is_member:
            raise HTTPException(
                status_code=400, 
                detail=f"User {user_id} is not a member of this {entity_type}"
            )
        
        # Get aligned members
        alignment_service = create_member_alignment_service(supabase)
        aligned_members = await alignment_service.get_aligned_members(
            user_id=user_id,
            entity_type=entity_type,
            entity_id=entity_id,
            limit=limit,
            min_similarity=min_similarity
        )
        
        return {
            "aligned_members": aligned_members,
            "count": len(aligned_members)
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error getting member alignment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to calculate member alignment")

async def check_entity_membership(user_id: str, entity_type: str, entity_id: str) -> bool:
    """Check if a user is a member of the specified group or event."""
    try:
        if entity_type == "group":
            # Check group membership
            result = await supabase.table("group_members").select("*").eq("group_id", entity_id).eq("member_id", user_id).execute()
            return len(result.data) > 0
        elif entity_type == "event":
            # Check event attendance
            result = await supabase.table("event_attendance").select("*").eq("event_id", entity_id).eq("user_id", user_id).execute()
            return len(result.data) > 0
        else:
            return False
    except Exception as e:
        logger.error(f"Error checking membership: {e}")
        return False

# --- Member Alignment Click Endpoint ---
@app.post("/member-alignment/click")
async def record_member_alignment_click(
    click_info: RecommendationClickInfo,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Records when a member alignment recommendation is clicked."""
    current_user_id = await get_current_user(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    metadata = {
        "member_id": click_info.recommendation_id,
        "source_page": click_info.source_page,
        "position": click_info.position
    }
    background_tasks.add_task(
        log_interaction,
        supabase_client=supabase,
        user_id=current_user_id,
        interaction_type="MEMBER_ALIGNMENT_CLICK",
        target_entity_type="PROFILE",
        target_entity_id=click_info.recommendation_id,
        metadata=metadata
    )
    return {"success": True, "message": "Member alignment click recorded"}

# --- Main Execution (for running directly with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly...")
    # Use environment variables for host/port if available
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Default reload to True for development if RELOAD env var is not explicitly set to 'false'
    reload_flag = os.getenv("RELOAD", "true").lower() != "false"

    uvicorn.run("main:app", host=host, port=port, reload=reload_flag) 