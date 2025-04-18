import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

export async function GET(request: Request) {
  // Debug environment vars
  console.log('[API Demo Status] DEBUG ENV:', {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL,
    hasAnonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
    hasServiceKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
  });
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Auth Check ---
  const { data: { session }, error: sessionError } = await supabaseUserClient.auth.getSession();
  console.log('[API Demo Status] Session:', session, 'Error:', sessionError);
  if (!session) {
    console.warn('[API Demo Status] Unauthorized: no session');
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  // Optional: Add Admin Role check here
  // --- End Auth Check ---

  console.log('[API Demo Status] Fetching demo data counts and profiles...');

  try {
    // Create an admin client for privileged operations
    const supabaseAdmin = createAdminClient();
    console.log('[API Demo Status] Admin client created.');

    const [
      { data: profilesData, error: profileError },
      { count: connectionsCount, error: connError },
      { count: interactionsCount, error: intError },
      { count: groupsCount, error: groupError },
      { count: eventsCount, error: eventError }
    ] = await Promise.all([
      supabaseAdmin.from('profiles').select('*').eq('is_demo', true),
      supabaseAdmin.from('connections').select('*', { count: 'exact', head: true }).eq('is_demo', true),
      supabaseAdmin.from('interaction_history').select('*', { count: 'exact', head: true }).eq('is_demo', true),
      supabaseAdmin.from('groups').select('*', { count: 'exact', head: true }).eq('is_demo', true),
      supabaseAdmin.from('events').select('*', { count: 'exact', head: true }).eq('is_demo', true)
    ]);

    // Error handling for individual counts (optional, could return partial data)
    if (profileError) console.error('[API Demo Status] Error fetching profile data:', profileError);
    if (connError) console.error('[API Demo Status] Error fetching connection count:', connError);
    if (intError) console.error('[API Demo Status] Error fetching interaction count:', intError);
    if (groupError) console.error('[API Demo Status] Error fetching group count:', groupError);
    if (eventError) console.error('[API Demo Status] Error fetching event count:', eventError);

    const responseData = {
        profilesData: profilesData ?? [],
        counts: {
            profiles: profilesData?.length ?? 0,
            connections: connectionsCount ?? 0,
            interactions: interactionsCount ?? 0,
            groups: groupsCount ?? 0,
            events: eventsCount ?? 0,
        }
    };

    console.log('[API Demo Status] Data fetched:', responseData.counts);
    return NextResponse.json(responseData, { status: 200 });

  } catch (error: any) {
    console.error('[API Demo Status] CRITICAL ERROR fetching counts:', error);
    return NextResponse.json({ error: 'Internal server error fetching demo status', details: error.message }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 