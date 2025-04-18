import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

// Define type for interaction inserts explicitly matching the DB schema
interface InteractionInsert {
  user_id: string;
  interaction_type: string;
  target_entity_type?: string | null;
  target_entity_id?: string | null;
  metadata?: any | null;
  is_demo: boolean; // Crucial: ensure the RPC function sets this
}

export async function POST(request: Request) {
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Auth Check ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  // --- Admin Role Check ---
  const { data: profile, error: profileError } = await supabaseUserClient
    .from('profiles')
    .select('role')
    .eq('id', session.user.id)
    .single();
  if (profileError || !profile || profile.role !== 'admin') {
    console.error('[API Seed Interactions] Forbidden: Admin role required');
    return NextResponse.json({ error: 'Forbidden: Admin role required' }, { status: 403 });
  }

  // Use admin client with service role key
  const supabaseAdmin = createAdminClient();

  // --- Get target count from request body ---
  let targetCount = 100; // Default count
  try {
    const body = await request.json();
    if (body.targetCount && typeof body.targetCount === 'number' && body.targetCount > 0) {
      targetCount = Math.min(body.targetCount, 2000);
    }
  } catch (e) { /* Ignore parsing errors, use default */ }

  console.log(`[API Seed Interactions] Starting process (Target: ${targetCount})...`);

  let seededCount = 0;
  const errors: string[] = [];
  let rlsDisabled = false; // Flag to track RLS state

  try {
    // --- Disable RLS --- 
    console.log("[API Seed Interactions] Attempting to DISABLE RLS for interaction_history...");
    const { error: disableRlsError } = await supabaseAdmin.rpc('execute_sql', { 
        sql: 'ALTER TABLE public.interaction_history DISABLE ROW LEVEL SECURITY;' 
    });
    if (disableRlsError) {
        throw new Error(`Failed to disable RLS for interactions: ${disableRlsError.message}`);
    }
    rlsDisabled = true;
    console.log("[API Seed Interactions] RLS disabled for interaction_history.");

    console.log("[API Seed Interactions] Fetching demo profiles, groups, events...");
    const [
      { data: profiles, error: profileError },
      { data: groups, error: groupError },
      { data: events, error: eventError }
    ] = await Promise.all([
      supabaseAdmin.from('profiles').select('id').eq('is_demo', true),
      supabaseAdmin.from('groups').select('id').eq('is_demo', true),
      supabaseAdmin.from('events').select('id').eq('is_demo', true),
    ]);

    if (profileError || !profiles || profiles.length === 0) {
      throw new Error(profileError?.message || 'No demo profiles found.');
    }
    if (groupError) console.warn('[API Seed Interactions] Warning fetching groups:', groupError.message);
    if (eventError) console.warn('[API Seed Interactions] Warning fetching events:', eventError.message);

    const profileIds = profiles.map(p => p.id);
    const groupIds = groups?.map(g => g.id) || [];
    const eventIds = events?.map(e => e.id) || [];

    console.log(`[API Seed Interactions] Found ${profileIds.length} profiles, ${groupIds.length} groups, ${eventIds.length} events.`);

    // This array holds objects matching the InteractionInsert interface
    const interactionsToInsert: InteractionInsert[] = [];
    const interactionTypes = ['PROFILE_VIEW', 'GROUP_JOIN', 'EVENT_RSVP', 'CONNECTION_REQUEST', 'CONNECTION_ACCEPT'];
    const maxAttempts = targetCount * 3;
    let attempts = 0;

    while (interactionsToInsert.length < targetCount && attempts < maxAttempts) {
      attempts++;
      const userId = profileIds[Math.floor(Math.random() * profileIds.length)];
      const interactionType = interactionTypes[Math.floor(Math.random() * interactionTypes.length)];
      let targetEntityType: string | null = null;
      let targetEntityId: string | null = null;

      switch (interactionType) {
        case 'PROFILE_VIEW':
        case 'CONNECTION_REQUEST':
        case 'CONNECTION_ACCEPT':
          targetEntityType = 'PROFILE';
          let targetProfileId = profileIds[Math.floor(Math.random() * profileIds.length)];
          if (profileIds.length > 1) {
             while (targetProfileId === userId) { targetProfileId = profileIds[Math.floor(Math.random() * profileIds.length)]; }
             targetEntityId = targetProfileId;
          } else { targetEntityId = null; }
          break;
        case 'GROUP_JOIN':
          if (groupIds.length > 0) {
            targetEntityType = 'GROUP';
            targetEntityId = groupIds[Math.floor(Math.random() * groupIds.length)];
          }
          break;
        case 'EVENT_RSVP':
          if (eventIds.length > 0) {
            targetEntityType = 'EVENT';
            targetEntityId = eventIds[Math.floor(Math.random() * eventIds.length)];
          }
          break;
      }

      if (targetEntityId) {
        interactionsToInsert.push({
          user_id: userId,
          interaction_type: interactionType,
          target_entity_type: targetEntityType,
          target_entity_id: targetEntityId,
          metadata: null,
          is_demo: true, // Keep this, though the RPC handles the insert
        });
      }
    }

    // --- Direct Insert Block --- 
    if (interactionsToInsert.length > 0) {
        console.log(`[API Seed Interactions] Attempting to insert ${interactionsToInsert.length} new interactions directly...`);
        const { data: insertedData, error: insertError } = await supabaseAdmin
            .from('interaction_history') // Target the correct table
            .insert(interactionsToInsert)
            .select();
        if (insertError) {
            console.error(`[API Seed Interactions] Direct Insert Error:`, insertError);
            errors.push(`Insert Error: ${insertError.message}`); // Capture insert error
            seededCount = 0;
        } else {
            seededCount = insertedData?.length ?? interactionsToInsert.length;
            console.log(`[API Seed Interactions] Finished direct insert. Successful: ${seededCount}`);
        }
    } else {
        console.log("[API Seed Interactions] No new interactions generated to insert.");
    }
    // --- End Direct Insert Block ---

    console.log("[API Seed Interactions] Seeding process finished before RLS re-enable.");

    if (errors.length > 0) {
       return NextResponse.json({
           message: `Completed with errors. ${errors.length} errors occurred. First error: ${errors[0]}`, 
           seededCount: seededCount,
           errors: errors
       }, { status: 207 });
    }
    return NextResponse.json({
        message: `${seededCount} demo interactions inserted successfully.`,
        seededCount: seededCount
    }, { status: 200 });

  } catch (error: any) {
    console.error('[API Seed Interactions] CRITICAL ERROR:', error);
    errors.push(error.message);
    return NextResponse.json({ 
        error: 'Internal server error during interaction seeding', 
        details: error.message, 
        seededCount: seededCount,
        errors: errors 
    }, { status: 500 });

  } finally {
     // --- Re-enable RLS --- 
    if (rlsDisabled) {
      console.log("[API Seed Interactions] Attempting to RE-ENABLE RLS for interaction_history...");
      const { error: enableRlsError } = await supabaseAdmin.rpc('execute_sql', { 
          sql: 'ALTER TABLE public.interaction_history ENABLE ROW LEVEL SECURITY;' 
      });
      if (enableRlsError) {
        console.error('[API Seed Interactions] FAILED TO RE-ENABLE RLS:', enableRlsError);
      } else {
        console.log("[API Seed Interactions] RLS re-enabled for interaction_history.");
      }
    }
  }
}

// Ensure the route is dynamically evaluated
export const dynamic = 'force-dynamic'; 