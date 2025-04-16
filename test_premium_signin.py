import requests
import json
from pprint import pprint

# URL for the Supabase authentication endpoint
auth_url = "https://ctglknfjoryifmpoynjb.supabase.co/auth/v1/token?grant_type=password"

# Premium user credentials
user_data = {
    "email": "test.premium@networkli.com",
    "password": "testpremium123"
}

# Headers with the Supabase anon key
headers = {
    "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN0Z2xrbmZqb3J5aWZtcG95bmpiIiwicm9sZSI6ImFub24iLCJpYXQiOjE2MzI0OTI4MDAsImV4cCI6MTk0ODA2ODgwMH0.IkbJDJFD8MhnGXOKsm-ggbizEJECw-o0yB_fpVcPm4w",
    "Content-Type": "application/json"
}

print("Attempting to sign in with premium user...")
try:
    # Try to login with Supabase
    response = requests.post(auth_url, json=user_data, headers=headers)
    response.raise_for_status()
    auth_data = response.json()
    
    # Extract the access token
    access_token = auth_data.get("access_token")
    print(f"Successfully logged in with premium user!")
    print(f"Access token: {access_token[:20]}...")
    
    # Now use the token to access the API
    api_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Get dashboard data
    dashboard_url = "http://localhost:8000/dashboard"
    print("\nFetching dashboard data...")
    dashboard_response = requests.get(dashboard_url, headers=api_headers)
    if dashboard_response.status_code == 200:
        dashboard_data = dashboard_response.json()
        print("Successfully retrieved dashboard data:")
        pprint(dashboard_data)
    else:
        print(f"Failed to get dashboard: {dashboard_response.status_code}")
        print(dashboard_response.text)
    
    # Get premium features
    premium_url = "http://localhost:8000/premium-features"
    print("\nFetching premium features...")
    premium_response = requests.get(premium_url, headers=api_headers)
    if premium_response.status_code == 200:
        premium_data = premium_response.json()
        print("Successfully retrieved premium features:")
        pprint(premium_data)
    else:
        print(f"Failed to get premium features: {premium_response.status_code}")
        print(premium_response.text)
        
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
except json.JSONDecodeError:
    print(f"Error parsing JSON response: {response.text}") 