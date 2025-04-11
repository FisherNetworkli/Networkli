/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost', 'your-supabase-project.supabase.co'],
  },
}

export default nextConfig 