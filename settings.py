from pathlib import Path
import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings  # Correct import

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_JWT_SECRET: str # For JWT verification
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None  # Added for service role key
    # Model paths (Update these if ML models move)
    ML_MODEL_PATH: Optional[str] = "app/ml/models/saved_models" # Point to potential new location
    NETWORKLI_GNN_MODEL: Optional[str] = "gnn_model.pt" # Just the filename
    
    class Config:
        # Load .env.local first (for local development) and fallback to .env
        env_file = [
            Path(__file__).parent / '.env.local',
            Path(__file__).parent / '.env'
        ]
        env_file_encoding = 'utf-8'
        extra = 'ignore' 
        
# Initialize settings
try:
    settings = Settings()
    logging.info(f"Settings loaded from {settings.Config.env_file}: SUPABASE_URL={settings.SUPABASE_URL}")
    logging.info(f"Using key type: {'Service Role Key' if settings.SUPABASE_SERVICE_ROLE_KEY else 'Regular Key'}")
except Exception as e:
    logging.error(f"Error loading settings: {e}")
    settings = None 