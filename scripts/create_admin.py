import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

def create_admin():
    email = os.getenv("ADMIN_EMAIL", "admin@example.com")
    password = os.getenv("ADMIN_PASSWORD", "admin123")

    try:
        # Check if admin user already exists
        response = supabase.table("profiles").select("*").eq("email", email).execute()
        
        if response.data:
            print(f"Admin user already exists: {email}")
            return

        # Create admin user
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
        })

        if response.user:
            # Update profile with admin role
            supabase.table("profiles").update({
                "role": "ADMIN",
                "full_name": "Admin",
            }).eq("id", response.user.id).execute()
            
            print(f"Admin user created: {email}")
        else:
            print("Failed to create admin user")

    except Exception as e:
        print(f"Error creating admin user: {str(e)}")

if __name__ == "__main__":
    create_admin() 