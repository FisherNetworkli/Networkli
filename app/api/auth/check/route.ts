import { NextResponse } from 'next/server';
import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';

export async function GET() {
  const supabase = createRouteHandlerClient({ cookies });
  
  // Check if user is authenticated
  const { data: { session } } = await supabase.auth.getSession();
  
  if (session) {
    // Check if user is admin by querying the profiles table
    const { data: profile } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', session.user.id)
      .single();
    
    if (profile && profile.role === 'admin') {
      return NextResponse.json({ redirect: '/admin' });
    } else {
      return NextResponse.json({ redirect: '/dashboard' });
    }
  }

  return NextResponse.json({ redirect: null });
} 