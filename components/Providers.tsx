'use client'

import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'

export default function Providers({ 
  children 
}: { 
  children: React.ReactNode
}) {
  return (
    <>{children}</>
  )
} 