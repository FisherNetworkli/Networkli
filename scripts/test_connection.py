import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path

# Load the .env file from parent directory
env_file_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_file_path)

# Print environment variables for debugging (redact sensitive parts)
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
print(f"SUPABASE_KEY (first 10 chars): {os.getenv('SUPABASE_KEY')[:10]}...")
print(f"SUPABASE_SERVICE_ROLE_KEY (first 10 chars): {os.getenv('SUPABASE_SERVICE_ROLE_KEY')[:10]}...")

# Try connecting with service role key
try:
    print("\nAttempting connection with SERVICE_ROLE_KEY:")
    service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    supabase = create_client(os.getenv('SUPABASE_URL'), service_role_key)
    
    # Test connection by querying the groups table
    result = supabase.from_("groups").select("*").limit(1).execute()
    print(f"Success with SERVICE_ROLE_KEY: {result.data}")
    
    # Try another query to auth.users table
    try:
        user_result = supabase.from_("auth.users").select("count", count="exact").limit(1).execute()
        print(f"Auth query result: {user_result}")
    except Exception as e:
        print(f"Auth query failed: {e}")
except Exception as e:
    print(f"Failed with SERVICE_ROLE_KEY: {e}")

# Try connecting with regular key
try:
    print("\nAttempting connection with SUPABASE_KEY:")
    regular_key = os.getenv('SUPABASE_KEY')
    supabase = create_client(os.getenv('SUPABASE_URL'), regular_key)
    
    # Test connection by querying the groups table
    result = supabase.from_("groups").select("*").limit(1).execute()
    print(f"Success with SUPABASE_KEY: {result.data}")
except Exception as e:
    print(f"Failed with SUPABASE_KEY: {e}")

# Verify the value we'd use in main.py logic
print("\nVerifying key selection logic:")
api_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') if os.getenv('SUPABASE_SERVICE_ROLE_KEY') else os.getenv('SUPABASE_KEY')
print(f"Selected key (first 10 chars): {api_key[:10]}...") 