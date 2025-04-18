import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server'; // Assuming you have this utility

export async function POST(request: Request) {
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Authentication/Authorization Check ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    console.error("[API Create Group] Authentication failed: No session found.");
    return NextResponse.json({ error: 'Unauthorized: Not logged in' }, { status: 401 });
  }

  // Verify admin role using the user client first
  const { data: profile, error: profileError } = await supabaseUserClient
    .from('profiles')
    .select('role')
    .eq('id', session.user.id)
    .single();

  if (profileError || !profile || profile.role !== 'admin') {
     console.error('[API Create Group] Admin check failed:', profileError || 'Profile not found or not admin');
     return NextResponse.json({ error: 'Forbidden: Requires admin role' }, { status: 403 });
  }
  // --- End Authentication/Authorization Check ---

  let groupData;
  try {
    groupData = await request.json();
    if (!groupData || !groupData.celebrityId || !groupData.name || !groupData.category) {
      return NextResponse.json({ error: 'Missing required group data (celebrityId, name, category)' }, { status: 400 });
    }
  } catch (e) {
    console.error('[API Create Group] Error parsing request body:', e);
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  // Check if required environment variables are present
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!supabaseUrl) {
    console.error('[API Create Group] Missing NEXT_PUBLIC_SUPABASE_URL environment variable');
    return NextResponse.json({ error: 'Server configuration error: Missing Supabase URL' }, { status: 500 });
  }
  
  if (!supabaseServiceRoleKey) {
    console.error('[API Create Group] Missing SUPABASE_SERVICE_ROLE_KEY environment variable');
    return NextResponse.json({ error: 'Server configuration error: Missing Supabase service role key' }, { status: 500 });
  }

  try {
    // Use direct creation instead of the utility function to provide more detailed errors
    const supabaseAdmin = createAdminClient();

    // Log successful client creation
    console.log('[API Create Group] Admin client created successfully');

    const { data, error: insertError } = await supabaseAdmin
      .from('groups')
      .insert([{
        name: groupData.name,
        description: `Demo group led by a celebrity in the ${groupData.category} industry`,
        category: groupData.category,
        created_by: groupData.celebrityId, 
        is_demo: true,
      }])
      .select('id')
      .single();

    if (insertError) {
      console.error('[API Create Group] Error inserting group:', insertError);
      return NextResponse.json({ error: `Database error: ${insertError.message}` }, { status: 500 });
    }

    console.log(`[API Create Group] Group '${groupData.name}' created successfully with ID: ${data?.id}`);
    return NextResponse.json({ 
      message: 'Demo group created successfully',
      groupId: data?.id
    }, { status: 201 }); // 201 Created

  } catch (error: any) {
    console.error('[API Create Group] CRITICAL ERROR:', error);
    // Enhanced error reporting with environment check (but sanitize sensitive info)
    const envCheck = {
      hasUrl: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasAnonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      hasServiceKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
      urlPrefix: process.env.NEXT_PUBLIC_SUPABASE_URL?.substring(0, 15) + '...' // Only log prefix for security
    };
    console.error('[API Create Group] Environment check:', envCheck);
    
    return NextResponse.json({ 
      error: 'Internal server error creating group', 
      details: error.message,
      code: error.code || 'unknown'
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; // Ensure fresh execution 