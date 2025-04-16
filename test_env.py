from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Define all the environment variables we need
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_JWT_SECRET: str
    
    class Config:
        env_file = str(Path(__file__).parent / '.env')
        env_file_encoding = 'utf-8'
        # Allow extra fields in the .env file
        extra = 'ignore'
        
settings = Settings()
print(f'SUPABASE_URL: {settings.SUPABASE_URL}')
print(f'SUPABASE_KEY: {settings.SUPABASE_KEY[:10]}...')
print(f'SUPABASE_JWT_SECRET: {settings.SUPABASE_JWT_SECRET[:10]}...') 