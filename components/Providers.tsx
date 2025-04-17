'use client'

// import { SessionProvider } from 'next-auth/react' // Removed next-auth provider

// Note: AuthProvider from '@/app/providers/AuthProvider.tsx' 
// should likely wrap the application layout instead, 
// as it manages the Supabase session via useAuth hook.

export default function Providers({ 
  children 
}: { 
  children: React.ReactNode
}) {
  // Return children directly, remove SessionProvider wrapper
  return <>{children}</>;
} 