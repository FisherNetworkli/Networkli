#!/bin/bash

# Exit on error
set -e

# Load environment variables
source .env

# Set database connection details
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="postgres"
DB_USER="postgres"
DB_PASSWORD="postgres"

# Apply migrations in order
echo "Applying initial schema migration..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f migrations/01_initial_schema.sql

echo "Applying related tables migration..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f migrations/02_related_tables.sql

echo "Migrations completed successfully!" 