"""
Machine Learning-based recommendation service for Networkli.

This module uses GraphSAGE for generating user recommendations based on embeddings.
"""

import logging
import os
import numpy as np
import torch
import joblib # Added for loading sklearn models
from typing import List, Dict, Any, Optional, Set
from supabase import Client
from settings import settings # Import settings from root
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import the specific model we intend to use
try:
    # Assuming MAMLModel is defined in the same place as GraphSAGEModel
    from ..models import GraphSAGEModel, MAMLModel 
    models_available = True
except ImportError as e:
    models_available = False
    logging.warning(f"Could not import GraphSAGEModel or MAMLModel: {e}")

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for generating GraphSAGE-based recommendations."""
    
    def __init__(self, supabase_client: Client):
        """Initialize the prediction service."""
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models and related objects
        self.graphsage_model: Optional[GraphSAGEModel] = None
        self.maml_model: Optional[MAMLModel] = None
        self.pca: Optional[PCA] = None
        self.kmeans: Optional[KMeans] = None
        
        # Configuration / Dimensions (should match training)
        self.feature_dim: int = 768 # Input feature dim
        self.hidden_dim: int = 256 # GNN hidden dim
        self.pca_dim: int = 32 # PCA output dim (input for MAML)
        self.num_classes: int = 10 # Output classes/clusters
        
        if models_available:
            try:
                self._load_models()
            except Exception as e:
                self.logger.error(f"Failed to load models during init: {e}", exc_info=True)
        else:
            self.logger.warning("GraphSAGE/MAML models could not be imported. ML predictions disabled.")
    
    def _load_models(self):
        """Load the trained models and transformers from storage."""
        if not settings or not settings.ML_MODEL_PATH:
            self.logger.error("ML Model path not configured in settings. Cannot load models.")
            return

        model_dir = settings.ML_MODEL_PATH
        self.logger.info(f"Attempting to load models from directory: {model_dir}")

        # Define expected filenames (adjust if needed)
        graphsage_file = os.path.join(model_dir, settings.NETWORKLI_GNN_MODEL or "graphsage_best.pth")
        maml_file = os.path.join(model_dir, "maml_best.pth")
        pca_file = os.path.join(model_dir, "pca.pkl")
        kmeans_file = os.path.join(model_dir, "kmeans.pkl")

        # Load GraphSAGE
        if os.path.exists(graphsage_file):
            try:
                self.graphsage_model = GraphSAGEModel(
                    in_channels=self.feature_dim,
                    hidden_channels=self.hidden_dim,
                    num_classes=self.num_classes # GraphSAGE output layer might be for classification during training
                )
                state_dict = torch.load(graphsage_file, map_location=self.device)
                self.graphsage_model.load_state_dict(state_dict)
                self.graphsage_model.to(self.device)
                self.graphsage_model.eval()
                self.logger.info(f"Successfully loaded GraphSAGE model from {graphsage_file}")
            except Exception as e:
                self.logger.error(f"Error loading GraphSAGE model from {graphsage_file}: {e}", exc_info=True)
                self.graphsage_model = None
        else:
            self.logger.error(f"GraphSAGE model file not found: {graphsage_file}")

        # Load MAML
        if os.path.exists(maml_file) and models_available: # Check models_available again
             try:
                # MAML input dim should be PCA output dim
                self.maml_model = MAMLModel(
                    input_dim=self.pca_dim, 
                    hidden_dim=self.hidden_dim, # Assuming same hidden dim, adjust if needed
                    num_classes=self.num_classes 
                )
                state_dict = torch.load(maml_file, map_location=self.device)
                self.maml_model.load_state_dict(state_dict)
                self.maml_model.to(self.device)
                self.maml_model.eval()
                self.logger.info(f"Successfully loaded MAML model from {maml_file}")
             except Exception as e:
                 self.logger.error(f"Error loading MAML model from {maml_file}: {e}", exc_info=True)
                 self.maml_model = None
        else:
            self.logger.warning(f"MAML model file not found or MAMLModel class unavailable: {maml_file}")


        # Load PCA
        if os.path.exists(pca_file):
            try:
                self.pca = joblib.load(pca_file)
                # Verify the expected output dimension
                if hasattr(self.pca, 'n_components_') and self.pca.n_components_ != self.pca_dim:
                     self.logger.warning(f"Loaded PCA n_components ({self.pca.n_components_}) differs from configured pca_dim ({self.pca_dim})")
                elif hasattr(self.pca, 'n_components') and self.pca.n_components != self.pca_dim:
                     self.logger.warning(f"Loaded PCA n_components ({self.pca.n_components}) differs from configured pca_dim ({self.pca_dim})")
                self.logger.info(f"Successfully loaded PCA transformer from {pca_file}")
            except Exception as e:
                self.logger.error(f"Error loading PCA transformer from {pca_file}: {e}", exc_info=True)
                self.pca = None
        else:
            self.logger.error(f"PCA file not found: {pca_file}")

        # Load KMeans
        if os.path.exists(kmeans_file):
            try:
                self.kmeans = joblib.load(kmeans_file)
                # Verify the expected number of clusters
                if hasattr(self.kmeans, 'n_clusters') and self.kmeans.n_clusters != self.num_classes:
                     self.logger.warning(f"Loaded KMeans n_clusters ({self.kmeans.n_clusters}) differs from configured num_classes ({self.num_classes})")
                self.logger.info(f"Successfully loaded KMeans model from {kmeans_file}")
            except Exception as e:
                self.logger.error(f"Error loading KMeans model from {kmeans_file}: {e}", exc_info=True)
                self.kmeans = None
        else:
            self.logger.error(f"KMeans file not found: {kmeans_file}")

    async def get_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        exclude_connected: bool = True
    ) -> List[Dict[str, Any]]:
        """Get recommendations using the GraphSAGE -> PCA -> MAML -> KMeans pipeline.
           Recommends users from the same cluster as the target user.
        
        Args:
            user_id: ID of the user to get recommendations for
            limit: Maximum number of recommendations to return
            exclude_connected: Whether to exclude already connected users
            
        Returns:
            List of recommended profiles from the same cluster.
        """
        # Check if all necessary models/transformers are loaded
        if not all([self.graphsage_model, self.maml_model, self.pca, self.kmeans]):
            missing = [
                name for name, model in [
                    ('GraphSAGE', self.graphsage_model),
                    ('MAML', self.maml_model),
                    ('PCA', self.pca),
                    ('KMeans', self.kmeans)
                ] if model is None
            ]
            self.logger.warning(f"ML models/transformers not fully loaded ({', '.join(missing)} missing). Cannot generate ML recommendations.")
            return []
            
        if not self.supabase:
            self.logger.error("Supabase client not available.")
            return []
        
        try:
            # --- Data Fetching (Same as before) --- 
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
            
            # --- Feature Extraction (Same as before) --- 
            all_profiles = [user_profile] + candidates
            profile_ids = [p['id'] for p in all_profiles]
            features = [self._extract_features(p) for p in all_profiles]
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

            # --- Build Edge Index for the Subgraph --- 
            self.logger.info(f"Building subgraph edge index for {len(valid_profile_ids)} profiles.")
            profile_id_to_index = {pid: i for i, pid in enumerate(valid_profile_ids)}
            valid_profile_ids_set = set(valid_profile_ids)
            
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device) # Default to empty
            try:
                # Query connections where both users are in our valid set
                self.logger.info(f"Querying connections for {len(valid_profile_ids)} profiles...")
                connection_result = await self.supabase.table("connections") \
                    .select("user_id1, user_id2") \
                    .in_("user_id1", valid_profile_ids) \
                    .in_("user_id2", valid_profile_ids) \
                    .execute()

                if connection_result.error:
                    self.logger.error(f"DB error fetching connections: {connection_result.error}")
                elif connection_result.data:
                    edges = []
                    for conn in connection_result.data:
                        u1, u2 = conn['user_id1'], conn['user_id2']
                        if u1 in profile_id_to_index and u2 in profile_id_to_index:
                            idx1 = profile_id_to_index[u1]
                            idx2 = profile_id_to_index[u2]
                            edges.append([idx1, idx2])
                            edges.append([idx2, idx1])
                    
                    if edges:
                        edge_index = torch.LongTensor(edges).t().contiguous().to(self.device)
                        self.logger.info(f"Built edge index with {edge_index.shape[1]} edges.")
                    else:
                        self.logger.info("No connections found between valid profiles.")
                else:
                    self.logger.info("Connection query returned no data.")
            except Exception as e:
                self.logger.error(f"Exception fetching/building connections: {e}", exc_info=True)
                # Fallback to empty edge_index already set
            
            # --- Embedding Generation & Clustering --- 
            self.logger.info(f"Generating embeddings and clusters for {valid_features.shape[0]} profiles using {edge_index.shape[1]} edges.")
            with torch.no_grad():
                # 1. Get GraphSAGE embeddings using the real edge_index
                all_embeddings_raw, _ = self.graphsage_model(valid_features, edge_index) 
                
                # 2. Apply PCA
                all_embeddings_np = all_embeddings_raw.cpu().numpy()
                all_embeddings_pca = self.pca.transform(all_embeddings_np)
                all_embeddings_pca_torch = torch.tensor(all_embeddings_pca, dtype=torch.float).to(self.device)
                
                # 3. Get MAML logits
                maml_logits = self.maml_model(all_embeddings_pca_torch)
                
                # 4. Refine cluster assignments using KMeans centers (cosine similarity)
                kmeans_centers = self.kmeans.cluster_centers_ # Shape: [num_clusters, pca_dim]
                from sklearn.metrics.pairwise import cosine_similarity
                cluster_assignments = []
                for i in range(all_embeddings_pca.shape[0]):
                    similarities = cosine_similarity(all_embeddings_pca[i].reshape(1, -1), kmeans_centers)
                    cluster_assignments.append(np.argmax(similarities)) # Assign to cluster with most similar center
                cluster_assignments = np.array(cluster_assignments)

            # --- Calculate Similarity & Apply Cluster Boost --- 
            user_cluster = cluster_assignments[user_idx_in_valid]
            user_embedding_raw = all_embeddings_raw[user_idx_in_valid]
            self.logger.info(f"User {user_id} assigned to cluster {user_cluster}. Calculating recommendations...")
            
            recommendations = []
            candidate_indices = [i for i, pid in enumerate(valid_profile_ids) if i != user_idx_in_valid]
            
            # Define the boost factor for same-cluster recommendations
            cluster_boost = 0.1 # Configurable: how much to boost score for same cluster

            for i in candidate_indices:
                candidate_profile_id = valid_profile_ids[i]
                candidate_cluster = cluster_assignments[i]
                candidate_embedding_raw = all_embeddings_raw[i]

                # Calculate base similarity score using raw GraphSAGE embeddings
                base_similarity = torch.nn.functional.cosine_similarity(
                    user_embedding_raw.unsqueeze(0),
                    candidate_embedding_raw.unsqueeze(0)
                ).item()

                # Apply boost if candidate is in the same cluster
                final_similarity = base_similarity
                if candidate_cluster == user_cluster:
                    final_similarity += cluster_boost
                    # Optional: Clamp score to a max (e.g., 1.0 if using cosine similarity)
                    # final_similarity = min(final_similarity, 1.0)

                # Find the original profile data for this candidate
                original_candidate_profile = next((p for p in all_profiles if p['id'] == candidate_profile_id), None)
                if original_candidate_profile:
                    recommendation = { 
                        **original_candidate_profile,
                        "cluster": int(candidate_cluster),
                        "similarity_score": float(final_similarity) # Use the potentially boosted score for ranking
                    }
                    recommendations.append(recommendation)
            
            # Sort recommendations by the final similarity score (higher is better)
            recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            self.logger.info(f"Generated {len(recommendations)} ML recommendations for user {user_id} (ranked by boosted score).")
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