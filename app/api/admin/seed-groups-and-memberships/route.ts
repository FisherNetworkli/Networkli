import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Assumes execute_sql function exists

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

// Data Interfaces (Can be refined based on actual schema)
interface GroupSeedData {
  name: string;
  description?: string | null;
  category?: string | null;
  industry?: string | null;
  location?: string | null;
  is_demo: boolean;
}

interface MembershipInsert {
  group_id: string;
  user_id: string;
  role: string; 
  is_demo: boolean;
}

// Helper function for delay (might still be needed for reads, but not between writes)
// const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export async function POST(request: Request) {
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Auth Check ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  // --- End Auth Check ---

  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Combined] Missing Supabase URL or Service Role Key.');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false }
  });

  // --- Configurable options ---
  const minGroupsPerUser = 1;
  const maxGroupsPerUser = 3;
  // --- End Configurable options ---

  let groupsToSeed: GroupSeedData[] = [];
  try {
    const body = await request.json();
    if (body.groups && Array.isArray(body.groups)) {
        groupsToSeed = body.groups.filter((g: any) => g.name && g.is_demo === true);
    } else {
        return NextResponse.json({ error: 'Invalid request body: missing groups array.' }, { status: 400 });
    }
    if (groupsToSeed.length === 0) {
         return NextResponse.json({ message: 'No valid groups provided to seed.', seededGroups: 0, seededMemberships: 0 }, { status: 200 });
    }
  } catch (e) { 
      console.error('[API Seed Combined] Error parsing request body:', e);
      return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  console.log(`[API Seed Combined] Starting process for ${groupsToSeed.length} groups...`);

  let seededGroupsCount = 0;
  let seededMembershipsCount = 0;
  const errors: string[] = [];
  let groupsRlsDisabled = false;
  let membersRlsDisabled = false;

  try {
    // --- 1. Fetch Demo Users --- 
    console.log("[API Seed Combined] Fetching demo users...");
    const { data: users, error: userError } = await supabaseAdmin
        .from('profiles').select('id').eq('is_demo', true);
    if (userError) throw new Error(`Failed to fetch demo users: ${userError.message}`);
    if (!users || users.length === 0) throw new Error('No demo users found.');
    const userIds = users.map(u => u.id);
    console.log(`[API Seed Combined] Found ${userIds.length} users.`);

    // --- 2. Seed Groups and get IDs --- 
    console.log("[API Seed Combined] Attempting to DISABLE RLS for groups...");
    const { error: disableGroupsRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.groups DISABLE ROW LEVEL SECURITY;' });
    if (disableGroupsRlsError) throw new Error(`Failed to disable RLS for groups: ${disableGroupsRlsError.message}`);
    groupsRlsDisabled = true;
    console.log("[API Seed Combined] RLS disabled for groups.");

    console.log(`[API Seed Combined] Inserting ${groupsToSeed.length} groups and selecting IDs...`);
    const { data: insertedGroups, error: insertGroupsError } = await supabaseAdmin
        .from('groups') 
        .insert(groupsToSeed)
        .select('id'); // Select ONLY the ID of newly inserted groups

    if (insertGroupsError) {
        console.error(`[API Seed Combined] Group Insert Error:`, insertGroupsError);
        throw new Error(`Group Insert Error: ${insertGroupsError.message}`); // Throw to stop execution
    }
    seededGroupsCount = insertedGroups?.length ?? 0;
    const newGroupIds = insertedGroups?.map(g => g.id) || [];
    console.log(`[API Seed Combined] Inserted ${seededGroupsCount} groups.`);

    if (newGroupIds.length === 0) {
        console.log("[API Seed Combined] No new groups were inserted, skipping membership seeding.");
        // No need to throw error, just report 0 memberships
    } else {
        // --- 3. Generate & Seed Memberships --- 
        console.log("[API Seed Combined] Generating memberships...");
        const membershipsToInsert: MembershipInsert[] = [];
        const existingMemberships = new Set<string>(); 
        userIds.forEach(userId => {
            const groupsToJoinCount = Math.floor(Math.random() * (maxGroupsPerUser - minGroupsPerUser + 1)) + minGroupsPerUser;
            const shuffledGroupIds = [...newGroupIds].sort(() => 0.5 - Math.random());
            for (let i = 0; i < groupsToJoinCount && i < shuffledGroupIds.length; i++) {
                const groupId = shuffledGroupIds[i];
                const key = `${userId}-${groupId}`;
                if (!existingMemberships.has(key)) {
                    membershipsToInsert.push({ user_id: userId, group_id: groupId, role: 'member', is_demo: true });
                    existingMemberships.add(key);
                }
            }
        });

        if (membershipsToInsert.length > 0) {
            console.log("[API Seed Combined] Attempting to DISABLE RLS for group_members...");
            const { error: disableMembersRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.group_members DISABLE ROW LEVEL SECURITY;' });
            if (disableMembersRlsError) throw new Error(`Failed to disable RLS for group_members: ${disableMembersRlsError.message}`);
            membersRlsDisabled = true;
            console.log("[API Seed Combined] RLS disabled for group_members.");

            console.log(`[API Seed Combined] Inserting ${membershipsToInsert.length} memberships...`);
            const { data: insertedMemberships, error: insertMembersError } = await supabaseAdmin
                .from('group_members').insert(membershipsToInsert).select();
            
            if (insertMembersError) {
                console.error(`[API Seed Combined] Membership Insert Error:`, insertMembersError);
                // Don't throw, just record error and count
                errors.push(`Membership Insert Error: ${insertMembersError.message}`); 
                seededMembershipsCount = 0;
            } else {
                seededMembershipsCount = insertedMemberships?.length ?? membershipsToInsert.length;
                console.log(`[API Seed Combined] Inserted ${seededMembershipsCount} memberships.`);
            }
        } else {
            console.log("[API Seed Combined] No new memberships generated.");
        }
    }

    // --- Success Response --- 
    console.log("[API Seed Combined] Seeding process finished before RLS re-enable.");
    if (errors.length > 0) {
       // Report partial success with errors
       return NextResponse.json({ 
           message: `Completed with ${errors.length} errors during membership insert. First error: ${errors[0]}`, 
           seededGroups: seededGroupsCount,
           seededMemberships: seededMembershipsCount, // Report count even if errors occurred
           errors: errors 
       }, { status: 207 });
    }
    return NextResponse.json({ 
        message: `${seededGroupsCount} groups and ${seededMembershipsCount} memberships seeded successfully.`, 
        seededGroups: seededGroupsCount,
        seededMemberships: seededMembershipsCount
    }, { status: 200 });

  } catch (error: any) {
    console.error('[API Seed Combined] CRITICAL ERROR:', error);
    errors.push(error.message);
    // Ensure counts reflect state before the critical error if possible
    return NextResponse.json({ 
        error: 'Internal server error during combined group/membership seeding', 
        details: error.message, 
        seededGroups: seededGroupsCount, // Groups might have been seeded before error
        seededMemberships: 0, // Memberships likely failed
        errors: errors 
    }, { status: 500 });

  } finally {
     // --- Re-enable RLS --- 
    if (membersRlsDisabled) {
      console.log("[API Seed Combined] Attempting to RE-ENABLE RLS for group_members...");
      const { error: enableRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.group_members ENABLE ROW LEVEL SECURITY;' });
      if (enableRlsError) console.error('[API Seed Combined] FAILED TO RE-ENABLE RLS for group_members:', enableRlsError);
      else console.log("[API Seed Combined] RLS re-enabled for group_members.");
    }
    if (groupsRlsDisabled) {
      console.log("[API Seed Combined] Attempting to RE-ENABLE RLS for groups...");
      const { error: enableRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.groups ENABLE ROW LEVEL SECURITY;' });
      if (enableRlsError) console.error('[API Seed Combined] FAILED TO RE-ENABLE RLS for groups:', enableRlsError);
      else console.log("[API Seed Combined] RLS re-enabled for groups.");
    }
  }
}

export const dynamic = 'force-dynamic'; 