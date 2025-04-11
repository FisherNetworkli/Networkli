# Networkli

Networkli is a professional networking platform designed specifically for introverts. Our AI-powered platform helps professionals make meaningful connections in a comfortable, authentic way.

## Features

- AI-powered matching algorithm
- Custom conversation starters
- Privacy controls
- Event integration
- Analytics dashboard
- Enterprise API access

## Tech Stack

- Next.js 14
- TypeScript
- Tailwind CSS
- Framer Motion
- Supabase

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/networkli.git
cd networkli
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Deployment

The project is configured for deployment on Vercel. Simply connect your GitHub repository to Vercel and it will automatically deploy your changes.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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