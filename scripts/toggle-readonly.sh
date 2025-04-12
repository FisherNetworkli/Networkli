#!/bin/bash

# Toggle read-only mode
if [ -f .env.local ]; then
  if grep -q "READONLY_MODE=true" .env.local; then
    # Disable read-only mode
    sed -i '' 's/READONLY_MODE=true/READONLY_MODE=false/' .env.local
    echo "Read-only mode disabled"
  else
    # Enable read-only mode
    if grep -q "READONLY_MODE=false" .env.local; then
      sed -i '' 's/READONLY_MODE=false/READONLY_MODE=true/' .env.local
    else
      echo "READONLY_MODE=true" >> .env.local
    fi
    echo "Read-only mode enabled"
  fi
else
  # Create .env.local file with read-only mode enabled
  echo "READONLY_MODE=true" > .env.local
  echo "Read-only mode enabled"
fi

# Restart the development server
echo "Restarting the development server..."
pkill -f "next dev"
npm run dev 