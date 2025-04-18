import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

// Define types for the group structure
interface GroupData {
  name: string;
  description: string;
  category: string;
  industry?: string;
  location?: string;
  is_demo: boolean;
}

export async function POST(request: Request) {
  console.log('[API Seed Groups] Route handler started');
  
  // Check environment variables early
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Groups] Missing Supabase environment variables');
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
      console.error('[API Seed Groups] No authenticated session found');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // --- Admin Role Check ---
    const { data: profile, error: profileError } = await supabaseUserClient
      .from('profiles')
      .select('role')
      .eq('id', session.user.id)
      .single();

    if (profileError) {
      console.error('[API Seed Groups] Error checking admin role:', profileError);
      return NextResponse.json({ error: 'Error verifying permissions' }, { status: 500 });
    }

    if (!profile || profile.role !== 'admin') {
      console.error('[API Seed Groups] User is not an admin:', session.user.id);
      return NextResponse.json({ error: 'Forbidden: Admin role required' }, { status: 403 });
    }

    // Parse request data
    let requestData;
    try {
      requestData = await request.json();
    } catch (e) {
      console.error('[API Seed Groups] Error parsing request body:', e);
      return NextResponse.json({ error: 'Invalid request format' }, { status: 400 });
    }

    if (!requestData.groups || !Array.isArray(requestData.groups)) {
      console.error('[API Seed Groups] Missing or invalid groups array in request body');
      return NextResponse.json({ error: 'Missing groups array in request' }, { status: 400 });
    }

    const groups = requestData.groups as GroupData[];
    console.log(`[API Seed Groups] Received ${groups.length} groups to seed`);

    // --- Create Admin Client ---
    const supabaseAdmin = createAdminClient();
    console.log('[API Seed Groups] Admin client created successfully');

    // --- Get Demo Profile IDs ---
    const { data: demoProfiles, error: profilesError } = await supabaseAdmin
      .from('profiles')
      .select('id')
      .eq('is_demo', true)
      .limit(100);

    if (profilesError) {
      console.error('[API Seed Groups] Error fetching demo profiles:', profilesError);
      return NextResponse.json({ error: 'Error fetching demo profiles' }, { status: 500 });
    }

    if (!demoProfiles || demoProfiles.length < 1) {
      console.error('[API Seed Groups] No demo profiles found to assign as group creators/members');
      return NextResponse.json({ 
        error: 'No demo profiles found', 
        details: 'Need at least one demo profile to create groups',
      }, { status: 400 });
    }

    const profileIds = demoProfiles.map(p => p.id);
    console.log(`[API Seed Groups] Found ${profileIds.length} demo profiles to use as organizers/members`);

    // --- Create Groups ---
    const createdGroups = [];
    const errors = [];
    let seededGroupsCount = 0;
    let seededMembershipsCount = 0;

    for (const group of groups) {
      try {
        // Assign a random profile as the creator
        const creatorId = profileIds[Math.floor(Math.random() * profileIds.length)];
        
        // Insert the group
        const { data: insertedGroup, error: groupError } = await supabaseAdmin
          .from('groups')
          .insert({
            name: group.name,
            description: group.description,
            category: group.category,
            industry: group.industry || group.category,
            location: group.location || 'Global',
            organizer_id: creatorId,
            is_demo: true
          })
          .select()
          .single();
          
        if (groupError) {
          console.error(`[API Seed Groups] Error inserting group ${group.name}:`, groupError);
          errors.push({ 
            type: 'group', 
            name: group.name, 
            message: groupError.message, 
            code: groupError.code 
          });
          continue; // Skip to next group
        }
        
        if (!insertedGroup) {
          console.error(`[API Seed Groups] Group ${group.name} inserted but no data returned`);
          continue; // Skip to next group
        }
        
        seededGroupsCount++;
        createdGroups.push(insertedGroup);
        console.log(`[API Seed Groups] Created group: ${insertedGroup.name} (${insertedGroup.id})`);
        
        // Add 3-10 random members to each group
        const memberCount = 3 + Math.floor(Math.random() * 8); // 3-10 members
        const groupMembers = new Set<string>();
        groupMembers.add(creatorId); // Creator is always a member
        
        // Add random unique members
        while (groupMembers.size < Math.min(memberCount, profileIds.length)) {
          const randomId = profileIds[Math.floor(Math.random() * profileIds.length)];
          groupMembers.add(randomId);
        }
        
        // Insert memberships
        for (const memberId of groupMembers) {
          try {
            const { error: membershipError } = await supabaseAdmin
              .from('group_memberships')
              .insert({
                group_id: insertedGroup.id,
                user_id: memberId,
                role: memberId === creatorId ? 'admin' : 'member',
                status: 'active',
                is_demo: true
              });
              
            if (membershipError) {
              console.error(`[API Seed Groups] Error adding member ${memberId} to group ${insertedGroup.id}:`, membershipError);
              errors.push({
                type: 'membership',
                group: insertedGroup.name,
                message: membershipError.message,
                code: membershipError.code
              });
            } else {
              seededMembershipsCount++;
            }
          } catch (err) {
            console.error(`[API Seed Groups] Exception adding member to group:`, err);
            errors.push({
              type: 'membership',
              group: insertedGroup.name,
              message: err instanceof Error ? err.message : String(err)
            });
          }
        }
        
        console.log(`[API Seed Groups] Added ${groupMembers.size} members to group ${insertedGroup.name}`);
        
      } catch (err) {
        console.error(`[API Seed Groups] Unexpected error processing group ${group.name}:`, err);
        errors.push({
          type: 'group',
          name: group.name,
          message: err instanceof Error ? err.message : String(err)
        });
      }
    }

    console.log(`[API Seed Groups] Created ${seededGroupsCount} groups with ${seededMembershipsCount} memberships`);
    console.log(`[API Seed Groups] Encountered ${errors.length} errors`);
    
    return NextResponse.json({
      message: `Created ${seededGroupsCount} groups and ${seededMembershipsCount} memberships`,
      seededGroups: seededGroupsCount,
      seededMemberships: seededMembershipsCount,
      errors: errors.length > 0 ? errors : undefined,
      success: errors.length === 0
    }, errors.length > 0 ? { status: 207 } : { status: 200 });

  } catch (error: any) {
    console.error('[API Seed Groups] Unhandled error:', error);
    
    // Log environment variable status (safely)
    const envCheck = {
      hasUrl: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasAnonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      hasServiceKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY
    };
    console.error('[API Seed Groups] Environment check:', envCheck);
    
    return NextResponse.json({ 
      error: 'Internal server error', 
      details: error.message,
      code: error.code || 'unknown'
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 