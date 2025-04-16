import jwt
import time
import requests
from pprint import pprint

# JWT secret for token generation
JWT_SECRET = "XDvkcek5MvyEpHtgwd6GWjSNCWeTyDTP2yyOoWLDYmpkkG08fvWkBBKkiVRWx+sB7/mu/qTSPSyAl3YrPxTlsg=="

# Premium user information
premium_user_id = "a946b4db-5138-4227-840b-1da8fdcebf05"
premium_user_email = "test.premium@networkli.com"

# Current timestamp and expiration (24 hours from now)
now = int(time.time())
expiration = now + (24 * 60 * 60)  # 24 hours

# Create payload
payload = {
    "aud": "authenticated",
    "exp": expiration,
    "sub": premium_user_id,
    "email": premium_user_email,
    "user_metadata": {
        "is_premium": True
    },
    "role": "authenticated"
}

# Generate token
token = jwt.encode(
    payload,
    JWT_SECRET,
    algorithm="HS256"
)

print(f"Generated token: {token}")

# Test API endpoints
api_headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Test endpoints
endpoints = [
    "/dashboard",
    "/premium-features",
    "/organizer/dashboard"
]

for endpoint in endpoints:
    url = f"http://localhost:8000{endpoint}"
    print(f"\nTesting endpoint: {url}")
    try:
        response = requests.get(url, headers=api_headers, timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Success! Response preview:")
            data = response.json()
            pprint(data)
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}") 