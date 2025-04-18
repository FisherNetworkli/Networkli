import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export async function GET() {
  try {
    // Check environment variables
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    
    // Environment variable report
    const report = {
      hasUrl: !!supabaseUrl,
      urlFirstChars: supabaseUrl ? supabaseUrl.substring(0, 12) + '...' : 'MISSING',
      hasServiceKey: !!supabaseServiceRoleKey,
      serviceKeyFirstChars: supabaseServiceRoleKey 
        ? supabaseServiceRoleKey.substring(0, 5) + '...' 
        : 'MISSING',
      serviceKeyLength: supabaseServiceRoleKey?.length || 0,
      createdClientSuccessfully: false,
      testQuerySuccessful: false,
      error: null as string | null
    };
    
    // If we have the required environment variables, try to create a client
    if (supabaseUrl && supabaseServiceRoleKey) {
      try {
        const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
          auth: { autoRefreshToken: false, persistSession: false }
        });
        
        report.createdClientSuccessfully = true;
        
        // Try a simple query that requires admin privileges - just count profiles
        const { count, error } = await supabaseAdmin
          .from('profiles')
          .select('*', { count: 'exact', head: true });
        
        if (error) {
          report.error = `Admin Query Error: ${error.message}`;
        } else {
          report.testQuerySuccessful = true;
        }
      } catch (clientError: any) {
        report.error = `Failed to create or use admin client: ${clientError.message}`;
      }
    }
    
    return NextResponse.json(report, { status: 200 });
  } catch (error: any) {
    return NextResponse.json({ 
      error: `Diagnostic error: ${error.message}`,
      stack: error.stack
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 