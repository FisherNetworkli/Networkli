#!/bin/bash

# Start the backend API server in the background
echo "Starting the backend API server..."
cd networkli-api
source ../venv/bin/activate
pkill -f "uvicorn main:app" || true  # Kill any existing instances
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a bit for the API to start
sleep 2

# Start the frontend Next.js server
echo "Starting the frontend Next.js server..."
cd ..
npm run dev -- -p 3002

# Clean up when the script is terminated
trap 'echo "Shutting down servers..."; kill $API_PID; exit' INT TERM 