"""
Machine Learning-based recommendation service for Networkli.

This module uses GraphSAGE for generating user recommendations based on embeddings.
"""

import logging
import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Set
from supabase import Client
from settings import settings # Import settings from root

# Import the specific model we intend to use
try:
    from ..models import GraphSAGEModel
    models_available = True
except ImportError as e:
    models_available = False
    logging.warning(f"Could not import GraphSAGEModel: {e}")

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for generating GraphSAGE-based recommendations."""
    
    def __init__(self, supabase_client: Client):
        """Initialize the prediction service.
        
        Args:
            supabase_client: Supabase client for database access
        """
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[GraphSAGEModel] = None
        self.feature_dim: int = 768 # Default dimension for sentence-transformers/all-MiniLM-L6-v2
        self.hidden_dim: int = 256 # Based on original script config
        self.num_classes: int = 10 # Placeholder, adjust if needed based on actual labels/clusters
        
        if models_available:
            try:
                self._load_models()
            except Exception as e:
                self.logger.error(f"Failed to load GraphSAGE model during init: {e}", exc_info=True)
        else:
            self.logger.warning("GraphSAGEModel could not be imported. ML predictions disabled.")
    
    def _load_models(self):
        """Load the trained GraphSAGE model from storage."""
        if not settings or not settings.ML_MODEL_PATH or not settings.NETWORKLI_GNN_MODEL:
            self.logger.error("ML Model path or name not configured in settings. Cannot load model.")
            return
        
        model_file = os.path.join(settings.ML_MODEL_PATH, settings.NETWORKLI_GNN_MODEL)
        self.logger.info(f"Attempting to load GraphSAGE model from: {model_file}")
        
        if not os.path.exists(model_file):
            self.logger.error(f"Model file not found: {model_file}")
            return
        
        try:
            # Instantiate the model structure
            # TODO: Confirm these dimensions match the trained model!
            self.model = GraphSAGEModel(
                in_channels=self.feature_dim,
                hidden_channels=self.hidden_dim,
                num_classes=self.num_classes
            )
            
            # Load the saved state dictionary
            if torch.cuda.is_available():
                state_dict = torch.load(model_file)
            else:
                state_dict = torch.load(model_file, map_location=self.device)
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            self.logger.info(f"Successfully loaded GraphSAGE model from {model_file} to {self.device}")
            
        except FileNotFoundError:
            self.logger.error(f"Model file not found at {model_file}")
            self.model = None
        except Exception as e:
            self.logger.error(f"Error loading GraphSAGE model state_dict from {model_file}: {e}", exc_info=True)
            self.model = None
    
    async def get_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        exclude_connected: bool = True
    ) -> List[Dict[str, Any]]:
        """Get recommendations using the GraphSAGE model.
        
        Args:
            user_id: ID of the user to get recommendations for
            limit: Maximum number of recommendations to return
            exclude_connected: Whether to exclude already connected users
            
        Returns:
            List of recommended profiles with similarity scores
        """
        if self.model is None:
            self.logger.warning("GraphSAGE model not loaded. Cannot generate ML recommendations.")
            return []
        if not self.supabase:
            self.logger.error("Supabase client not available.")
            return []
        
        try:
            # --- Data Fetching --- 
            self.logger.info(f"Fetching data for ML recommendation for user: {user_id}")
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                self.logger.error(f"User profile not found for user_id: {user_id}")
                return []
            
            excluded_user_ids = {user_id}
            if exclude_connected:
                connections = await self._get_user_connections(user_id)
                excluded_user_ids.update(connections)
                self.logger.info(f"Excluding {len(excluded_user_ids)} users (self + connections).")
            
            candidates = await self._get_candidate_profiles(excluded_user_ids)
            if not candidates:
                self.logger.warning(f"No candidate profiles found for user {user_id}.")
                return []
            self.logger.info(f"Found {len(candidates)} candidate profiles.")
            
            # --- Feature Extraction & Embedding Generation --- 
            # Combine user and candidates for batch processing if possible
            all_profiles = [user_profile] + candidates
            profile_ids = [p['id'] for p in all_profiles]
            
            # TODO: Build edge_index for the subgraph of user + candidates
            # This requires fetching connections *between* these profiles
            # For now, using a placeholder or assuming model can run without edge_index for inference
            # This is a simplification and likely needs refinement.
            edge_index_placeholder = torch.empty((2, 0), dtype=torch.long).to(self.device)
            
            # Extract features (e.g., sentence embeddings)
            # Ensure features have the dimension self.feature_dim
            features = [self._extract_features(p) for p in all_profiles]
            # Filter out profiles where feature extraction failed
            valid_indices = [i for i, f in enumerate(features) if f is not None]
            if not valid_indices:
                self.logger.error("Feature extraction failed for all profiles.")
                return []
            
            valid_features = torch.FloatTensor(np.array([features[i] for i in valid_indices])).to(self.device)
            valid_profile_ids = [profile_ids[i] for i in valid_indices]
            user_idx_in_valid = valid_profile_ids.index(user_id) if user_id in valid_profile_ids else -1
            
            if user_idx_in_valid == -1:
                self.logger.error(f"Feature extraction failed for the target user {user_id}.")
                return []
            
            self.logger.info(f"Generating embeddings for {valid_features.shape[0]} profiles.")
            with torch.no_grad():
                # Get embeddings from the GraphSAGE model
                # The model returns (embeddings, out), we only need embeddings
                all_embeddings, _ = self.model(valid_features, edge_index_placeholder)
            
            user_embedding = all_embeddings[user_idx_in_valid]
            candidate_embeddings = torch.cat([all_embeddings[:user_idx_in_valid], all_embeddings[user_idx_in_valid+1:]])
            candidate_ids = [pid for i, pid in enumerate(valid_profile_ids) if i != user_idx_in_valid]
            valid_candidates = [p for p in all_profiles if p['id'] in candidate_ids]
            
            if candidate_embeddings.shape[0] == 0:
                self.logger.warning(f"No valid candidate embeddings generated for user {user_id}.")
                return []
            
            # --- Similarity Calculation & Ranking --- 
            similarity_scores = torch.nn.functional.cosine_similarity(
                user_embedding.unsqueeze(0), # Shape: [1, hidden_dim]
                candidate_embeddings         # Shape: [num_candidates, hidden_dim]
            ).cpu().numpy() 
            
            recommendations = []
            for i, candidate_profile in enumerate(valid_candidates):
                # Use the profile data fetched earlier
                recommendation = { 
                    **candidate_profile, # Include all profile fields
                    "similarity_score": float(similarity_scores[i])
                }
                recommendations.append(recommendation)
            
            recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
            self.logger.info(f"Generated {len(recommendations)} ML recommendations for user {user_id}.")
            return recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Error generating ML recommendations for user {user_id}: {e}", exc_info=True)
            return [] # Fallback to empty list on error
    
    def _extract_features(self, profile: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract feature vector from a user profile.
        Focus on using pre-computed sentence embeddings if available.
        Args:
            profile: User profile dictionary
        Returns:
            Feature vector as numpy array, or None if extraction fails.
        """
        # Prioritize using a combined or primary embedding field if available
        # Example: Using 'bio_embedding' as the primary feature source
        embedding = profile.get("bio_embedding") 
        # Add logic to potentially combine bio, expertise, needs, goals embeddings if needed
        # combined_embedding = combine_embeddings(...) 
        
        if embedding and isinstance(embedding, list) and len(embedding) == self.feature_dim:
            return np.array(embedding)
        else:
            # Fallback or error if no suitable embedding found
            # logger.warning(f"No suitable embedding found for profile {profile.get('id')}. Cannot generate ML features.")
            # Alternative: Generate features from other fields (less ideal)
            # features = np.zeros(self.feature_dim)
            # ... populate features based on experience, domain_id, etc. ...
            # return features 
            return None # Return None if features cannot be generated reliably
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's profile including embedding fields."""
        if not self.supabase: return None
        try:
            # Select all relevant fields, including embeddings
            result = await self.supabase.table("profiles") \
                .select("*, bio_embedding, expertise_embedding, needs_embedding, goals_embedding") \
                .eq("id", user_id) \
                .maybe_single() \
                .execute()
            if result.error:
                self.logger.error(f"DB error retrieving profile for {user_id}: {result.error}")
                return None
            return result.data
        except Exception as e:
            self.logger.error(f"Exception retrieving profile for {user_id}: {e}", exc_info=True)
            return None
    
    async def _get_user_connections(self, user_id: str) -> Set[str]:
        """Get IDs of users connected to the given user."""
        if not self.supabase: return set()
        connected_ids = set()
        try:
            # Consider pagination for users with many connections
            # Fetch accepted connections only? Depends on definition
            # status_filter = "accepted"

            # Connections where user is requester
            req_res = await self.supabase.table("connections") \
                .select("receiver_id") \
                .eq("requester_id", user_id) \
                .execute()
                # .eq("status", status_filter).execute()
            if req_res.data:
                connected_ids.update([item['receiver_id'] for item in req_res.data])

            # Connections where user is receiver
            rec_res = await self.supabase.table("connections") \
                .select("requester_id") \
                .eq("receiver_id", user_id) \
                .execute()
                # .eq("status", status_filter).execute()
            if rec_res.data:
                connected_ids.update([item['requester_id'] for item in rec_res.data])

            return connected_ids
        except Exception as e:
            self.logger.error(f"Exception retrieving connections for {user_id}: {e}", exc_info=True)
            return set()
    
    async def _get_candidate_profiles(self, exclude_ids: Set[str]) -> List[Dict[str, Any]]:
        """Get candidate profiles for recommendations, excluding specified IDs."""
        if not self.supabase: return []
        try:
            # TODO: Implement more efficient candidate selection for large scale
            # - Consider filtering by location, industry, recent activity?
            # - Implement pagination
            result = await self.supabase.table("profiles") \
                .select("*, bio_embedding, expertise_embedding, needs_embedding, goals_embedding") \
                .not_.in_("id", list(exclude_ids)) \
                .limit(1000) \
                .execute()  # Add a reasonable limit to avoid fetching entire DB

            if result.error:
                self.logger.error(f"DB error retrieving candidate profiles: {result.error}")
                return []
            return result.data
        except Exception as e:
            self.logger.error(f"Exception retrieving candidate profiles: {e}", exc_info=True)
            return []

# Note: This service now focuses solely on GraphSAGE recommendations.
# The MAML model logic, PCA, KMeans from the original script would need
# separate integration, likely for training or offline analysis/clustering tasks,
# rather than direct recommendation ranking in this service. 