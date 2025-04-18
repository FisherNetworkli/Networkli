import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createAdminClient } from '@/utils/supabase/server';

// Tables to clean up, in the order they should be deleted
const DEMO_TABLES = [
  'interaction_history',
  'event_attendance',
  'group_memberships',
  'events',
  'groups',
  'connections',
  // profiles should be last as others depend on it
  'profiles'
];

export async function POST(request: Request) {
  console.log('[API Reset Demo] Reset handler started');
  
  // Check environment variables early
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Reset Demo] Missing Supabase environment variables');
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
      console.error('[API Reset Demo] No authenticated session found');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // --- Admin Role Check ---
    const { data: profile, error: profileError } = await supabaseUserClient
      .from('profiles')
      .select('role')
      .eq('id', session.user.id)
      .single();

    if (profileError) {
      console.error('[API Reset Demo] Error checking admin role:', profileError);
      return NextResponse.json({ error: 'Error verifying permissions' }, { status: 500 });
    }

    if (!profile || profile.role !== 'admin') {
      console.error('[API Reset Demo] User is not an admin:', session.user.id);
      return NextResponse.json({ error: 'Forbidden: Admin role required' }, { status: 403 });
    }

    // --- Create Admin Client ---
    const supabaseAdmin = createAdminClient();
    console.log('[API Reset Demo] Admin client created successfully');

    // --- Delete Demo Data ---
    const results = {};
    const errors = [];

    for (const table of DEMO_TABLES) {
      try {
        console.log(`[API Reset Demo] Deleting data from table: ${table}`);
        
        const { error, count } = await supabaseAdmin
          .from(table)
          .delete({ count: 'exact' })
          .eq('is_demo', true);
        
        if (error) {
          console.error(`[API Reset Demo] Error deleting from ${table}:`, error);
          errors.push({ table, message: error.message, code: error.code });
          results[table] = { success: false, error: error.message };
        } else {
          console.log(`[API Reset Demo] Successfully deleted ${count || 'unknown'} rows from ${table}`);
          results[table] = { success: true, count };
        }
      } catch (e) {
        console.error(`[API Reset Demo] Exception deleting from ${table}:`, e);
        errors.push({ table, message: e.message });
        results[table] = { success: false, error: e.message };
      }
    }

    if (errors.length > 0) {
      console.warn(`[API Reset Demo] Reset completed with ${errors.length} errors`);
      return NextResponse.json({
        message: `Reset completed with ${errors.length} errors`,
        results,
        errors
      }, { status: 207 }); // 207 Multi-Status
    }
    
    console.log('[API Reset Demo] Reset completed successfully');
    return NextResponse.json({
      message: 'Demo environment has been reset successfully',
      results
    });

  } catch (error: any) {
    console.error('[API Reset Demo] Unhandled error:', error);
    
    // Log environment variable status (safely)
    const envCheck = {
      hasUrl: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasAnonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      hasServiceKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY
    };
    console.error('[API Reset Demo] Environment check:', envCheck);
    
    return NextResponse.json({ 
      error: 'Internal server error', 
      details: error.message,
      code: error.code || 'unknown'
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 