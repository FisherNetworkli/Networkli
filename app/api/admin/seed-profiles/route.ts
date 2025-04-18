import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

interface CelebrityProfile {
  first_name: string;
  last_name: string;
  email: string;
  role: string;
  title: string;
  bio: string;
  location: string;
  industry: string;
  company: string;
  avatar_url?: string;
  skills: string[];
  interests: string[];
  professional_goals?: string[];
  values?: string[];
  website?: string | null;
  linkedin_url?: string | null;
  github_url?: string | null;
  is_demo?: boolean;
  is_celebrity?: boolean;
}

export async function POST(request: Request) {
  console.log('[API Seed Profiles] Route handler started');

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Profiles] Missing Supabase environment variables');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    console.error('[API Seed Profiles] No authenticated session found');
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data: profile, error: profileError } = await supabaseUserClient
    .from('profiles')
    .select('role')
    .eq('id', session.user.id)
    .single();
  if (profileError) {
    console.error('[API Seed Profiles] Error verifying permissions:', profileError);
    return NextResponse.json({ error: 'Error verifying permissions' }, { status: 500 });
  }
  if (!profile || profile.role !== 'admin') {
    console.error('[API Seed Profiles] Forbidden: Admin role required');
    return NextResponse.json({ error: 'Forbidden: Admin role required' }, { status: 403 });
  }

  let requestData;
  try {
    requestData = await request.json();
  } catch (e) {
    console.error('[API Seed Profiles] Error parsing request body:', e);
    return NextResponse.json({ error: 'Invalid request format' }, { status: 400 });
  }
  const celebrities: CelebrityProfile[] = requestData.celebrities;
  if (!Array.isArray(celebrities)) {
    return NextResponse.json({ error: "Missing 'celebrities' array in request" }, { status: 400 });
  }

  const supabaseAdmin = createAdminClient();
  let seededCount = 0;
  const errors: string[] = [];

  for (const celeb of celebrities) {
    try {
      const { error: insertError } = await supabaseAdmin
        .from('profiles')
        .insert([{ ...celeb, is_demo: true }]);
      if (insertError) {
        console.error('[API Seed Profiles] Insert error for', celeb.email, insertError);
        errors.push(`Failed insert for ${celeb.email}: ${insertError.message}`);
      } else {
        seededCount++;
      }
    } catch (err: any) {
      console.error('[API Seed Profiles] Exception inserting', celeb.email, err);
      errors.push(`Exception for ${celeb.email}: ${err.message}`);
    }
  }

  console.log(`[API Seed Profiles] Seeded ${seededCount} profiles with ${errors.length} errors`);
  return NextResponse.json({ message: `Seeded ${seededCount} profiles`, seededCount, errors, success: errors.length === 0 });
}

export const dynamic = 'force-dynamic';