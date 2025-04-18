import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

export async function POST(request: Request) {
  console.log('[API Seed Connections] Route handler started');
  
  // Check environment variables early
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Connections] Missing Supabase environment variables');
    return NextResponse.json(
      { 
        error: 'Server configuration error', 
        details: {
          hasUrl: !!supabaseUrl,
          hasServiceKey: !!supabaseServiceRoleKey 
        }
      }, 
      { status: 500 }
    );
  }

  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  try {
    // --- Auth Check ---
    const { data: { session } } = await supabaseUserClient.auth.getSession();
    if (!session) {
      console.error('[API Seed Connections] No authenticated session found');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // --- Admin Role Check ---
    const { data: profile, error: profileError } = await supabaseUserClient
      .from('profiles')
      .select('role')
      .eq('id', session.user.id)
      .single();

    if (profileError) {
      console.error('[API Seed Connections] Error checking admin role:', profileError);
      return NextResponse.json({ error: 'Error verifying permissions' }, { status: 500 });
    }

    if (!profile || profile.role !== 'admin') {
      console.error('[API Seed Connections] User is not an admin:', session.user.id);
      return NextResponse.json({ error: 'Forbidden: Admin role required' }, { status: 403 });
    }

    // Parse request data
    let requestData;
    try {
      requestData = await request.json();
    } catch (e) {
      console.error('[API Seed Connections] Error parsing request body:', e);
      return NextResponse.json({ error: 'Invalid request format' }, { status: 400 });
    }

    const targetCount = requestData.targetCount || 50; // Default to 50 if not specified
    console.log(`[API Seed Connections] Requested target count: ${targetCount}`);

    // --- Create Admin Client ---
    const supabaseAdmin = createAdminClient();
    console.log('[API Seed Connections] Admin client created successfully');

    // --- Get Demo Profile IDs ---
    const { data: demoProfiles, error: profilesError } = await supabaseAdmin
      .from('profiles')
      .select('id')
      .eq('is_demo', true)
      .limit(100);

    if (profilesError) {
      console.error('[API Seed Connections] Error fetching demo profiles:', profilesError);
      return NextResponse.json({ error: 'Error fetching demo profiles' }, { status: 500 });
    }

    if (!demoProfiles || demoProfiles.length < 2) {
      console.error('[API Seed Connections] Not enough demo profiles to create connections');
      return NextResponse.json({ 
        error: 'Not enough demo profiles', 
        details: 'Need at least 2 profiles to create connections',
        count: demoProfiles?.length || 0
      }, { status: 400 });
    }

    console.log(`[API Seed Connections] Found ${demoProfiles.length} demo profiles`);
    
    // --- Create Random Connections ---
    const profileIds = demoProfiles.map(p => p.id);
    const connections = [];
    let seededCount = 0;
    let errorCount = 0;

    // Create up to targetCount random connections between demo profiles
    const existingConnectionPairs = new Set();
    
    for (let i = 0; i < Math.min(targetCount, 200); i++) { // Cap at 200 to prevent timeouts
      // Pick two random profiles
      const randomIndex1 = Math.floor(Math.random() * profileIds.length);
      let randomIndex2 = Math.floor(Math.random() * profileIds.length);
      
      // Ensure we don't pick the same profile twice
      while (randomIndex1 === randomIndex2) {
        randomIndex2 = Math.floor(Math.random() * profileIds.length);
      }
      
      const profile1Id = profileIds[randomIndex1];
      const profile2Id = profileIds[randomIndex2];
      
      // Create a unique pair identifier (alphabetically ordered to avoid duplicates)
      const pairKey = [profile1Id, profile2Id].sort().join('_');
      
      // Skip if this pair already exists
      if (existingConnectionPairs.has(pairKey)) {
        continue;
      }
      
      existingConnectionPairs.add(pairKey);
      
      // Insert the connection
      try {
        const { error: connectionError } = await supabaseAdmin
          .from('connections')
          .insert({
            user_id_1: profile1Id,
            user_id_2: profile2Id,
            status: 'accepted',
            is_demo: true
          });
          
        if (connectionError) {
          console.error(`[API Seed Connections] Error creating connection ${i}:`, connectionError);
          errorCount++;
        } else {
          seededCount++;
        }
      } catch (err) {
        console.error(`[API Seed Connections] Exception creating connection ${i}:`, err);
        errorCount++;
      }
    }

    console.log(`[API Seed Connections] Created ${seededCount} connections with ${errorCount} errors`);
    
    return NextResponse.json({
      message: `Created ${seededCount} demo connections`,
      seededCount,
      errorCount,
      success: true
    });

  } catch (error: any) {
    console.error('[API Seed Connections] Unhandled error:', error);
    
    // Log environment variable status (safely)
    const envCheck = {
      hasUrl: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasAnonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      hasServiceKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY
    };
    console.error('[API Seed Connections] Environment check:', envCheck);
    
    return NextResponse.json({ 
      error: 'Internal server error', 
      details: error.message,
      code: error.code || 'unknown'
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';

// Helper RPC function definition (if not already present in your DB)
/* 
CREATE OR REPLACE FUNCTION execute_sql(sql TEXT)
RETURNS void
LANGUAGE plpgsql 
SECURITY DEFINER -- Or the role that has permissions to ALTER TABLE
AS $$
BEGIN
  EXECUTE sql;
END;
$$;
*/ 