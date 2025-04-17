import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createClient, PostgrestError } from '@supabase/supabase-js';

// IMPORTANT: Ensure these environment variables are set in your deployment environment!
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

// Helper type for Supabase RPC results (adjust if needed based on your function)
interface RpcResult {
  success: boolean;
  message: string;
}

export async function POST(request: Request) {
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Authentication/Authorization Check ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized: Not logged in' }, { status: 401 });
  }
  // Optional: Add role-based admin check here if needed
  // --- End Authentication/Authorization Check ---

  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Reset Demo] Missing Supabase URL or Service Role Key.');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false }
  });

  console.log('[API Reset Demo] Starting demo environment reset process...');

  try {
    // --- Step 1: Get Demo Profile IDs (Changed from is_celebrity to is_demo) ---
    console.log('[API Reset Demo] Fetching demo profile IDs...');
    const { data: demoProfiles, error: idError } = await supabaseAdmin
      .from('profiles')
      .select('id')
      .eq('is_demo', true);

    if (idError) {
      console.error('[API Reset Demo] Error fetching demo profile IDs:', idError);
      throw new Error(`Failed to fetch demo profile IDs: ${idError.message}`);
    }

    const demoProfileIds = demoProfiles?.map(p => p.id) || [];
    let foundDemoProfiles = demoProfileIds.length > 0;

    if (!foundDemoProfiles) {
      console.log('[API Reset Demo] No demo profiles found to delete. Proceeding with general demo data cleanup.');
    } else {
        console.log(`[API Reset Demo] Found ${demoProfileIds.length} demo profile IDs to process.`);
    }

    // --- Step 2: Delete referencing data AND other general demo data ---
    console.log('[API Reset Demo] Deleting associated and general demo data...');
    const deletionPromises = [];

    // Deletions specifically related to fetched demo profile IDs (only if IDs exist)
    if (foundDemoProfiles) {
        // Connections involving demo profiles (using correct FK columns)
        deletionPromises.push(
            supabaseAdmin.from('connections').delete().in('requester_id', demoProfileIds)
                .then(result => ({ name: 'Demo Connections (Requester)', ...result }))
        );
        deletionPromises.push(
            supabaseAdmin.from('connections').delete().in('receiver_id', demoProfileIds)
                .then(result => ({ name: 'Demo Connections (Receiver)', ...result }))
        );
        // Groups created by demo profiles (using correct FK column)
        deletionPromises.push(
            supabaseAdmin.from('groups').delete().in('organizer_id', demoProfileIds)
                .then(result => ({ name: 'Demo Groups (organizer)', ...result }))
        );
        /* Events FK attempt removed */

        // --- Add deletions for other FKs found ---
        // Event Attendees
        deletionPromises.push(
            supabaseAdmin.from('event_attendees').delete().in('user_id', demoProfileIds)
                .then(result => ({ name: 'Demo Event Attendees', ...result }))
        );
        // Group Members
        deletionPromises.push(
            supabaseAdmin.from('group_members').delete().in('user_id', demoProfileIds)
                .then(result => ({ name: 'Demo Group Members', ...result }))
        );
        // Messages involving demo profiles
        deletionPromises.push(
            supabaseAdmin.from('messages').delete().in('sender_id', demoProfileIds)
                .then(result => ({ name: 'Demo Messages (Sender)', ...result }))
        );
        deletionPromises.push(
            supabaseAdmin.from('messages').delete().in('receiver_id', demoProfileIds)
                .then(result => ({ name: 'Demo Messages (Receiver)', ...result }))
        );
        // Profile Views involving demo profiles
        deletionPromises.push(
            supabaseAdmin.from('profile_views').delete().in('visitor_id', demoProfileIds)
                .then(result => ({ name: 'Demo Profile Views (Visitor)', ...result }))
        );
        deletionPromises.push(
            supabaseAdmin.from('profile_views').delete().in('profile_id', demoProfileIds)
                .then(result => ({ name: 'Demo Profile Views (Profile)', ...result }))
        );
        // Other potentially relevant tables
        deletionPromises.push(
            supabaseAdmin.from('interaction_history').delete().in('user_id', demoProfileIds)
                .then(result => ({ name: 'Demo Interaction History', ...result }))
        );
        deletionPromises.push(
            supabaseAdmin.from('activity_tracking').delete().in('user_id', demoProfileIds)
                .then(result => ({ name: 'Demo Activity Tracking', ...result }))
        );
         deletionPromises.push(
            supabaseAdmin.from('user_skills').delete().in('user_id', demoProfileIds)
                .then(result => ({ name: 'Demo User Skills', ...result }))
        );
         deletionPromises.push(
            supabaseAdmin.from('profile_skills').delete().in('profile_id', demoProfileIds)
                .then(result => ({ name: 'Demo Profile Skills', ...result }))
        );
        // Note: user_preferences, subscriptions might not be demo-specific
        // Decide if you want to delete these based on your demo data strategy.

        // Add other demo-profile-specific FK deletions here if needed
    }

    // General Demo Data Deletion (using is_demo flag)
    // This part remains important for data not directly linked via FKs we check
    deletionPromises.push(
        supabaseAdmin.from('groups').delete().eq('is_demo', true)
             .then(result => ({ name: 'General Demo Groups', ...result }))
    );
    deletionPromises.push(
        supabaseAdmin.from('events').delete().eq('is_demo', true)
             .then(result => ({ name: 'General Demo Events', ...result }))
    );
    deletionPromises.push(
        supabaseAdmin.from('connections').delete().eq('is_demo', true)
             .then(result => ({ name: 'General Demo Connections', ...result }))
    );
    /*
    deletionPromises.push(
        supabaseAdmin.from('interactions').delete().eq('is_demo', true)
             .then(result => ({ name: 'General Demo Interactions', ...result }))
    );
    */
     // Add other general demo data deletions here

    const deletionResults = await Promise.allSettled(deletionPromises);
    console.log('[API Reset Demo] Raw Deletion Results:', JSON.stringify(deletionResults, null, 2)); // More detailed log

    const deletionErrors: string[] = [];
    deletionResults.forEach((result, index) => {
      const opName = (result.status === 'fulfilled' && result.value?.name) ? result.value.name : `Operation ${index}`;
      if (result.status === 'rejected') {
        deletionErrors.push(`${opName}: Rejected - ${(result.reason as Error)?.message || result.reason}`);
      } else if (result.value?.error) {
        // Check for Supabase errors even if promise fulfilled
        const pgError = result.value.error as PostgrestError;
        deletionErrors.push(`${opName}: Failed - ${pgError.message} (Code: ${pgError.code}) Hint: ${pgError.hint}`);
      }
    });

    if (deletionErrors.length > 0) {
      console.error('[API Reset Demo] Errors occurred during associated/general data deletion:', deletionErrors);
      throw new Error(`Failed to delete some associated demo data. Errors: ${deletionErrors.join('; ')}`);
    }
    console.log('[API Reset Demo] Associated and general demo data deleted successfully (or no errors detected).');

    // --- Step 3: Delete Demo profiles LAST (only if IDs were found and Step 2 had no errors) ---
    let profileDeleteStatus = 'Skipped';
    if (foundDemoProfiles) {
      console.log('[API Reset Demo] Attempting to delete demo profiles via RPC...');
      // Call the database function using rpc()
      const { error: profileDeleteError } = await supabaseAdmin
        .rpc('delete_demo_profiles');

      if (profileDeleteError) {
        profileDeleteStatus = 'Failed';
        console.error('[API Reset Demo] Error calling delete_demo_profiles RPC:', profileDeleteError);
        throw new Error(`Failed to delete demo profiles via RPC: ${profileDeleteError.message}`);
      } else {
        profileDeleteStatus = 'Success';
        console.log('[API Reset Demo] RPC delete_demo_profiles called successfully.');
      }
    } else {
        console.log('[API Reset Demo] Skipping profile deletion as no demo profiles were found initially.');
    }

    console.log(`[API Reset Demo] Demo environment reset completed. Profile deletion status: ${profileDeleteStatus}`);
    return NextResponse.json({ message: 'Demo environment reset successfully' }, { status: 200 });

  } catch (error: any) {
    console.error('[API Reset Demo] CRITICAL ERROR during reset process:', error);
    // Ensure a 500 is returned for any unhandled error in the try block
    return NextResponse.json({ error: 'Internal server error during reset', details: error.message }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 