import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

export async function POST(request: Request) {
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Auth Check ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  // Optional: Add Admin Role check here
  // --- End Auth Check ---

  // --- Debug: Log if the service key is being loaded ---
  console.log('[API Seed Connections] Service Key Loaded (start):', supabaseServiceRoleKey?.substring(0, 5) ?? 'MISSING/UNDEFINED');
  // --- End Debug ---

  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Connections] Missing Supabase URL or Service Role Key.');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false }
  });

  // --- Get target count from request body ---
  let targetCount = 50; // Default count
  try {
    const body = await request.json();
    if (body.targetCount && typeof body.targetCount === 'number' && body.targetCount > 0) {
      targetCount = Math.min(body.targetCount, 1000); // Cap count
    }
  } catch (e) { /* Ignore parsing errors, use default */ }

  console.log(`[API Seed Connections] Starting process (Target: ${targetCount})...`);

  let seededCount = 0;
  const errors: string[] = [];
  let rlsDisabled = false; // Flag to track RLS state

  try {
    // --- Disable RLS --- 
    console.log("[API Seed Connections] Attempting to DISABLE RLS for connections...");
    const { error: disableRlsError } = await supabaseAdmin.rpc('execute_sql', { 
        sql: 'ALTER TABLE public.connections DISABLE ROW LEVEL SECURITY;' 
    });
    if (disableRlsError) {
        throw new Error(`Failed to disable RLS: ${disableRlsError.message}`);
    }
    rlsDisabled = true;
    console.log("[API Seed Connections] RLS disabled for connections.");

    console.log("[API Seed Connections] Fetching demo profiles...");
    const { data: profiles, error: profileError } = await supabaseAdmin
        .from('profiles')
        .select('id')
        .eq('is_demo', true);

    if (profileError || !profiles || profiles.length < 2) {
      throw new Error(profileError?.message || 'Need at least 2 demo profiles to create connections.');
    }
    const profileIds = profiles.map(p => p.id);
    console.log(`[API Seed Connections] Found ${profileIds.length} demo profiles.`);

    console.log("[API Seed Connections] Fetching existing demo connections...");
    // Fetch existing to avoid duplicates
    const { data: existingConnectionsData, error: existingConnError } = await supabaseAdmin
        .from('connections')
        .select('requester_id, receiver_id') // Select columns used for key
        .eq('is_demo', true); // Only check against existing *demo* connections

    if (existingConnError) {
      // Log warning but don't necessarily fail, maybe the table is empty
      console.warn(`[API Seed Connections] Warning fetching existing connections: ${existingConnError.message}`);
      // throw new Error(`Failed to fetch existing connections: ${existingConnError.message}`);
    }
    const existingConnections = new Set(
         existingConnectionsData?.map(c => `${c.requester_id}-${c.receiver_id}`) ?? []
    );
    console.log(`[API Seed Connections] Found ${existingConnections.size} existing demo connections.`);

    // Array to hold the raw pairs initially
    const connectionPairsToInsert: { requester_id: string; receiver_id: string; }[] = [];
    const maxAttempts = targetCount * 5; // Prevent infinite loop
    let attempts = 0;

    while (connectionPairsToInsert.length < targetCount && attempts < maxAttempts) {
        attempts++;
        const requesterId = profileIds[Math.floor(Math.random() * profileIds.length)];
        let receiverId = profileIds[Math.floor(Math.random() * profileIds.length)];

        // Ensure requester and receiver are different
        if (requesterId !== receiverId) {
             const key1 = `${requesterId}-${receiverId}`;
             const key2 = `${receiverId}-${requesterId}`; // Check both directions
             // Ensure this connection (or its reverse) doesn't already exist
             if (!existingConnections.has(key1) && !existingConnections.has(key2)) {
                 connectionPairsToInsert.push({
                     requester_id: requesterId,
                     receiver_id: receiverId,
                 });
                 // Add to set to prevent duplicates within this batch
                 existingConnections.add(key1);
             }
        }
    }

    // --- Direct Insert Block --- 
    if (connectionPairsToInsert.length > 0) {
        console.log(`[API Seed Connections] Attempting to insert ${connectionPairsToInsert.length} new connections directly...`);
        const connectionsWithDemoFlag = connectionPairsToInsert.map(conn => ({
            ...conn, status: 'accepted', is_demo: true
        }));
        const { data: insertedData, error: insertError } = await supabaseAdmin
            .from('connections').insert(connectionsWithDemoFlag).select();
        if (insertError) {
            console.error(`[API Seed Connections] Direct Insert Error:`, insertError);
            errors.push(`Insert Error: ${insertError.message}`);
            seededCount = 0;
        } else {
            seededCount = insertedData?.length ?? connectionPairsToInsert.length;
            console.log(`[API Seed Connections] Finished direct insert. Successful: ${seededCount}`);
        }
    } else {
        console.log("[API Seed Connections] No new connections generated to insert.");
    }
    // --- End Direct Insert Block ---

    console.log("[API Seed Connections] Seeding process finished before RLS re-enable.");
    // Note: RLS re-enabled in finally block

    if (errors.length > 0) {
       return NextResponse.json({ 
           message: `Completed with errors. ${errors.length} errors occurred. First error: ${errors[0]}`, 
           seededCount: seededCount,
           errors: errors 
       }, { status: 207 });
    }
    return NextResponse.json({ 
        message: `${seededCount} demo connections inserted successfully.`, 
        seededCount: seededCount
    }, { status: 200 });

  } catch (error: any) {
    console.error('[API Seed Connections] CRITICAL ERROR:', error);
    errors.push(error.message); // Add critical error to errors list
    return NextResponse.json({ 
        error: 'Internal server error during connection seeding', 
        details: error.message, 
        seededCount: seededCount, // Report count before critical error
        errors: errors
    }, { status: 500 });

  } finally {
    // --- Re-enable RLS --- 
    if (rlsDisabled) {
      console.log("[API Seed Connections] Attempting to RE-ENABLE RLS for connections...");
      const { error: enableRlsError } = await supabaseAdmin.rpc('execute_sql', { 
          sql: 'ALTER TABLE public.connections ENABLE ROW LEVEL SECURITY;' 
      });
      if (enableRlsError) {
        console.error('[API Seed Connections] FAILED TO RE-ENABLE RLS:', enableRlsError);
        // Decide how to handle this critical failure - maybe log to a monitoring service
      } else {
        console.log("[API Seed Connections] RLS re-enabled for connections.");
      }
    }
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