import jwt
import time
from datetime import datetime, timedelta
import os

# Set the actual JWT secret
JWT_SECRET = "XDvkcek5MvyEpHtgwd6GWjSNCWeTyDTP2yyOoWLDYmpkkG08fvWkBBKkiVRWx+sB7/mu/qTSPSyAl3YrPxTlsg=="

# Premium user information
premium_user_id = "a946b4db-5138-4227-840b-1da8fdcebf05"
premium_user_email = "test.premium@networkli.com"

# Current timestamp and expiration (24 hours from now)
now = int(time.time())
expiration = now + (24 * 60 * 60)  # 24 hours

# Create payload similar to Supabase JWT
payload = {
    "aud": "authenticated",
    "exp": expiration,
    "sub": premium_user_id,
    "email": premium_user_email,
    "phone": "",
    "app_metadata": {
        "provider": "email",
        "providers": ["email"]
    },
    "user_metadata": {
        "is_premium": True
    },
    "role": "authenticated",
    "aal": "aal1",
    "amr": [
        {
            "method": "password",
            "timestamp": now - 3600  # 1 hour ago
        }
    ],
    "session_id": f"premium-test-session-{now}"
}

# Generate the token
token = jwt.encode(
    payload,
    JWT_SECRET,
    algorithm="HS256"
)

print(f"Generated JWT token for premium user:")
print(token)
print("\nTo use this token with curl:")
print(f"curl -H \"Authorization: Bearer {token}\" http://localhost:8000/dashboard")
print(f"\nAnd to test premium features:")
print(f"curl -H \"Authorization: Bearer {token}\" http://localhost:8000/premium-features") 