import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs'
import { cookies } from 'next/headers'
import { NextResponse } from 'next/server'
import { createAdminClient } from '@/utils/supabase/server'

export async function GET() {
  try {
    const cookieStore = cookies()
    const supabase = createRouteHandlerClient({ cookies: () => cookieStore })
    const supabaseAdmin = createAdminClient()
    
    // Test getting session
    const { data: { session }, error: sessionError } = await supabase.auth.getSession()
    if (sessionError) throw new Error(`Session Error: ${sessionError.message}`)
    console.log('Test Route - Session:', session ? `User ID: ${session.user.id}` : 'No session')

    // Test admin operation (e.g., list users - careful with large user bases)
    const { data: usersData, error: usersError } = await supabaseAdmin.auth.admin.listUsers({ perPage: 1 })
    if (usersError) throw new Error(`Admin Error: ${usersError.message}`)
    console.log('Test Route - Admin User List Count:', usersData.users.length)

    return NextResponse.json({ 
      message: 'Auth test completed successfully', 
      sessionStatus: session ? 'Active' : 'Inactive',
      adminTest: 'Passed'
    })
  } catch (error: any) {
    console.error('Auth Test Error:', error.message)
    return NextResponse.json({ error: 'Auth test failed', details: error.message }, { status: 500 })
  }
} 