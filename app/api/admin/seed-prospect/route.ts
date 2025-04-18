import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

export async function POST(request: Request) {
  console.log('[API Seed Prospect] Route handler started');

  // Check environment
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Prospect] Missing Supabase environment variables');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  // Authenticate user session
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    console.error('[API Seed Prospect] Unauthorized');
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // Check admin role
  const { data: userProfile, error: profileError } = await supabaseUserClient
    .from('profiles')
    .select('role')
    .eq('id', session.user.id)
    .single();
  if (profileError || !userProfile || userProfile.role !== 'admin') {
    console.error('[API Seed Prospect] Forbidden: Admin role required');
    return NextResponse.json({ error: 'Forbidden: Admin role required' }, { status: 403 });
  }

  // Parse request body
  let body: any;
  try {
    body = await request.json();
  } catch (e) {
    console.error('[API Seed Prospect] Invalid JSON body', e);
    return NextResponse.json({ error: 'Invalid request format' }, { status: 400 });
  }
  const { profileData, arrayInputs } = body;
  if (!profileData || !arrayInputs) {
    return NextResponse.json({ error: 'Missing profileData or arrayInputs' }, { status: 400 });
  }

  // Convert array strings to actual arrays
  const parseArray = (str: string) =>
    str.split(',').map(s => s.trim()).filter(s => s);
  const skills = parseArray(arrayInputs.skillsString || '');
  const interests = parseArray(arrayInputs.interestsString || '');
  const professional_goals = parseArray(arrayInputs.goalsString || '');
  const values = parseArray(arrayInputs.valuesString || '');

  const supabaseAdmin = createAdminClient();

  // Delete existing demo prospects (non-celebrities)
  try {
    const { error: delErr } = await supabaseAdmin
      .from('profiles')
      .delete()
      .eq('is_demo', true)
      .not('is_celebrity', 'eq', true);
    if (delErr) console.warn('[API Seed Prospect] Warning deleting old prospects', delErr);
  } catch (err) {
    console.warn('[API Seed Prospect] Exception deleting old prospects', err);
  }

  // Insert new prospect profile
  try {
    const { data, error: insertError } = await supabaseAdmin
      .from('profiles')
      .insert([
        {
          ...profileData,
          skills,
          interests,
          professional_goals,
          values,
          is_demo: true,
        }
      ])
      .select('id, first_name, last_name')
      .single();
    if (insertError || !data) {
      console.error('[API Seed Prospect] Insert error', insertError);
      return NextResponse.json({ error: insertError?.message || 'Insert failed' }, { status: 500 });
    }

    const seededProspectId = data.id;
    const seededProspectName = `${data.first_name} ${data.last_name}`;
    console.log(`[API Seed Prospect] Seeded prospect ${seededProspectId}`);

    return NextResponse.json({ message: 'Prospect seeded', seededProspectId, seededProspectName, success: true });
  } catch (err: any) {
    console.error('[API Seed Prospect] Unexpected error', err);
    return NextResponse.json({ error: err.message || 'Unexpected error' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 