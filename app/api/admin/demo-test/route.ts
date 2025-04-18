import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

export async function GET(request: Request) {
  console.log('[API Demo Test] Test route started');
  console.log('[API Demo Test] ENV:', {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL,
    anonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
    serviceKeyPresent: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
    serviceKeyPrefix: process.env.SUPABASE_SERVICE_ROLE_KEY?.substring(0, 5) ?? 'none'
  });
  try {
    const supabaseAdmin = createAdminClient();
    console.log('[API Demo Test] Admin client created');

    // Fetch up to 10 demo profiles
    const { data, error } = await supabaseAdmin
      .from('profiles')
      .select('id, first_name, last_name')
      .eq('is_demo', true)
      .limit(10);

    console.log('[API Demo Test] Query result:', { data, error });
    if (error) {
      return NextResponse.json({ success: false, error: error.message }, { status: 500 });
    }

    return NextResponse.json({
      success: true,
      demoCount: data?.length ?? 0,
      sample: data ?? []
    });
  } catch (err: any) {
    console.error('[API Demo Test] Unexpected error:', err);
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 