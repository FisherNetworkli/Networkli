import os
import logging
from datetime import datetime # Import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import httpx # Import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Domain ID Mapping (Should match main.py or be imported) ---
# TODO: Move this mapping to a shared config/module
INDUSTRY_TO_DOMAIN_ID = {
    "technology": 1, "software development": 1, "information technology": 1,
    "finance": 2, "banking": 2, "investment": 2,
    "healthcare": 3, "medical": 3, "pharmaceuticals": 3,
    "education": 4,
    "marketing": 5, "advertising": 5,
    "sales": 6,
    "design": 7, "ux/ui": 7,
    "human resources": 8, "recruiting": 8,
    # Add more industries
}
DEFAULT_DOMAIN_ID = 0 # ID for unknown/other industries

def get_domain_id(industry: Optional[str]) -> Optional[int]:
    if not industry:
        return DEFAULT_DOMAIN_ID
    return INDUSTRY_TO_DOMAIN_ID.get(industry.lower().strip(), DEFAULT_DOMAIN_ID)
# --- End Domain ID Mapping ---

# --- Embedding Generation Helper (Should match main.py or be imported) ---
embedding_model = None

def generate_embedding(text: Optional[str], field_name: str, profile_id: str) -> Optional[List[float]]:
    global embedding_model
    if not text:
        return None
    if embedding_model is None:
        logger.warning(f"Skipping {field_name} embedding for {profile_id}: Model not loaded.")
        return None
    try:
        vector = embedding_model.encode(text)
        return vector.tolist()
    except Exception as e:
        logger.error(f"Error generating {field_name} embedding for {profile_id}: {e}")
        return None
# --- End Embedding Generation Helper ---

def backfill_profiles(supabase: Client, batch_size: int = 50):
    """Fetches profiles with missing embeddings or domain_id and updates them."""
    global embedding_model
    logger.info("Starting profile backfill process...")

    # Load Sentence Transformer Model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence Transformer model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load Sentence Transformer model: {e}")
        return # Cannot proceed without the model

    processed_count = 0
    updated_count = 0
    current_page = 0
    
    while True:
        logger.info(f"Fetching batch {current_page + 1} (size: {batch_size})...")
        offset = current_page * batch_size
        try:
            # Fetch profiles needing update - REMOVED 'username', ADDED 'email' for logging
            response = supabase.table("profiles") \
                .select("id, email, industry, bio, expertise, needs, meaningful_goals, bio_embedding, expertise_embedding, needs_embedding, goals_embedding, domain_id") \
                .or_("bio_embedding.is.null,expertise_embedding.is.null,needs_embedding.is.null,goals_embedding.is.null,domain_id.is.null") \
                .range(offset, offset + batch_size - 1) \
                .execute()

            if not response.data:
                logger.info("No more profiles found requiring backfill.")
                break

            profiles_to_process = response.data
            logger.info(f"Processing {len(profiles_to_process)} profiles in this batch...")

            for profile in profiles_to_process:
                profile_id = profile.get("id")
                # Use email for logging, fallback to ID
                log_identifier = profile.get("email", f"user_{profile_id}") 
                update_payload: Dict[str, Any] = {}

                # 1. Check and generate embeddings
                if profile.get("bio_embedding") is None and profile.get("bio"):
                    embedding = generate_embedding(profile.get("bio"), "bio", profile_id)
                    if embedding: update_payload["bio_embedding"] = embedding

                if profile.get("expertise_embedding") is None and profile.get("expertise"):
                    embedding = generate_embedding(profile.get("expertise"), "expertise", profile_id)
                    if embedding: update_payload["expertise_embedding"] = embedding

                if profile.get("needs_embedding") is None and profile.get("needs"):
                    embedding = generate_embedding(profile.get("needs"), "needs", profile_id)
                    if embedding: update_payload["needs_embedding"] = embedding

                if profile.get("goals_embedding") is None and profile.get("meaningful_goals"):
                    embedding = generate_embedding(profile.get("meaningful_goals"), "goals", profile_id)
                    if embedding: update_payload["goals_embedding"] = embedding
                
                # 2. Check and set domain_id
                if profile.get("domain_id") is None:
                     domain_id = get_domain_id(profile.get("industry"))
                     if domain_id is not None: # Check if get_domain_id returns a value
                         update_payload["domain_id"] = domain_id


                # 3. Update profile if changes were made
                if update_payload:
                    try:
                        # Use log_identifier in log message
                        logger.info(f"Updating profile {profile_id} ({log_identifier}) with fields: {list(update_payload.keys())}")
                        update_payload["updated_at"] = datetime.utcnow().isoformat() # Update timestamp
                        
                        update_response = supabase.table("profiles") \
                            .update(update_payload) \
                            .eq("id", profile_id) \
                            .execute()

                        # Check response data (v2+ style)
                        if update_response.data:
                            updated_count += 1
                            logger.info(f"Successfully updated profile {profile_id}.")
                        else:
                            # This case might happen if RLS prevents update or no data actually changed,
                            # but no error was raised. Treat as warning or investigate RLS if unexpected.
                            logger.warning(f"Update API call succeeded for profile {profile_id} but returned no data (maybe no change or RLS issue?).")
                    except Exception as update_e:
                        # Catch potential exceptions from .execute() directly
                        logger.error(f"Exception during update for profile {profile_id}: {update_e}")
                else:
                    logger.info(f"No updates needed for profile {profile_id} ({log_identifier}) in this pass.")

                processed_count += 1

            # Check if we fetched less than the batch size, indicating the last page
            if len(profiles_to_process) < batch_size:
                logger.info("Reached the end of profiles requiring backfill.")
                break

            current_page += 1

        except Exception as fetch_e:
            logger.error(f"Error fetching profile batch {current_page + 1}: {fetch_e}")
            break # Stop processing if fetching fails

    logger.info(f"Backfill process finished. Processed: {processed_count}, Updated: {updated_count}")


if __name__ == "__main__":
    logger.info("--- Starting Backfill Script ---")
    
    # Load .env file from the ROOT project directory
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    dotenv_path = os.path.join(project_root, '.env')
    logger.info(f"Looking for .env file at: {dotenv_path}")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(".env file loaded from project root.")
    else:
        logger.warning(".env file not found in project root directory. Ensure relevant Supabase env vars are set.")

    # Get specific environment variables
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("FATAL: NEXT_PUBLIC_SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY environment variables not set.")
    else:
        try:
            # --- Temporarily Unset Proxy Environment Variables ---
            # This prevents httpx (used by supabase-py) from auto-detecting them
            logger.info("Attempting to unset proxy environment variables for script context...")
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            os.environ.pop('ALL_PROXY', None)
            os.environ.pop('http_proxy', None) # Check lowercase too
            os.environ.pop('https_proxy', None)
            os.environ.pop('all_proxy', None)
            logger.info("Proxy environment variables unset.")
            # --- End Unset Proxy ---
            
            logger.info(f"Connecting to Supabase at {supabase_url[:20]}...") 
            # Initialize client without explicit options
            supabase_client: Client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized.")
            
            # --- RUN THE BACKFILL ---
            backfill_profiles(supabase_client)
            # --- END RUN ---

        except Exception as client_e:
            logger.error(f"FATAL: Failed to initialize Supabase client: {client_e}")

    logger.info("--- Backfill Script Finished ---") 