# Networkli

This project uses a local Supabase instance for development.

## Local Development Setup

### Prerequisites
- Docker
- Docker Compose

### Starting the Local Supabase Instance

1. Create required directories:
```bash
mkdir -p volumes/storage volumes/db
```

2. Start the Supabase services:
```bash
docker-compose up -d
```

### Accessing Services

Once the services are running, you can access:

- Supabase Studio: http://localhost:3000
- REST API: http://localhost:8000/rest/v1
- Authentication API: http://localhost:8000/auth/v1
- Storage API: http://localhost:8000/storage/v1

### Database Connection Details

- Host: localhost
- Port: 5432
- Database: postgres
- User: postgres
- Password: your-super-secret-password

### Important Notes

- The database data is persisted in the `volumes/db` directory
- Uploaded files are stored in the `volumes/storage` directory
- Make sure to replace the default passwords in production 