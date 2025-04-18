import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Ensure these are set in your environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

export async function POST(request: Request) {
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Auth Check (Ensure only Admins can call this) ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  
  // Fetch the caller's profile to check their role
  const { data: callerProfile, error: callerError } = await supabaseUserClient
    .from('profiles')
    .select('role')
    .eq('id', session.user.id)
    .single();

  if (callerError || !callerProfile || callerProfile.role !== 'admin') {
      console.warn(`[API GenLink] Unauthorized attempt by user ${session.user.id}, role: ${callerProfile?.role}`);
      return NextResponse.json({ error: 'Forbidden: Requires admin privileges' }, { status: 403 });
  }
  // --- End Auth Check ---

  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API GenLink] Missing Supabase URL or Service Role Key.');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  // Parse request body for userId and optional redirectTo
  let targetUserId: string;
  let redirectTo: string | undefined;
  try {
    const body = await request.json();
    targetUserId = body.userId;
    redirectTo = body.redirectTo;
    if (!targetUserId || typeof targetUserId !== 'string') {
      throw new Error('Missing or invalid target userId in request body.');
    }
    console.log(`[API GenLink] Received request for user ID: ${targetUserId}, redirectTo: ${redirectTo}`);
  } catch (e: any) {
    console.error('[API GenLink] Error parsing request body:', e);
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  // Use Admin client for elevated privileges
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false }
  });

  try {
    // --- Step 1: Get the target user's email address --- 
    console.log(`[API GenLink] Fetching user data for ID: ${targetUserId}`);
    const { data: targetUserData, error: getUserError } = await supabaseAdmin.auth.admin.getUserById(targetUserId);
    
    if (getUserError) {
        console.error('[API GenLink] Error fetching target user:', getUserError);
        // Check if it's a specific "not found" error
        if (getUserError.message.toLowerCase().includes('not found')) {
            return NextResponse.json({ error: `User with ID ${targetUserId} not found.` }, { status: 404 });
        }
        throw new Error(`Failed to fetch target user: ${getUserError.message}`);
    }

    if (!targetUserData?.user?.email) {
        console.error('[API GenLink] Target user found but email is missing.');
        throw new Error('Target user email could not be retrieved.');
    }
    const targetEmail = targetUserData.user.email;
    console.log(`[API GenLink] Found target email: ${targetEmail}`);

    // --- Step 2: Generate the Magic Link --- 
    console.log(`[API GenLink] Generating magic link for ${targetEmail}...`);
    // Generate magic link, optionally redirecting to a specific path after login
    const genParams: any = {
      type: 'magiclink',
      email: targetEmail
    };
    if (redirectTo && typeof redirectTo === 'string') {
      genParams.options = { redirectTo };
    }
    const { data: linkData, error: linkError } = await supabaseAdmin.auth.admin.generateLink(genParams);

    if (linkError) {
        console.error('[API GenLink] Error generating magic link:', linkError);
        throw new Error(`Failed to generate sign-in link: ${linkError.message}`);
    }

    // The linkData contains various properties, including the full link in `properties.action_link`
    const magicLink = linkData?.properties?.action_link;

    if (!magicLink) {
        console.error('[API GenLink] Link generated but action_link property is missing:', linkData);
        throw new Error('Failed to extract magic link from response.');
    }

    console.log(`[API GenLink] Magic link generated successfully.`);

    // --- Success Response ---
    return NextResponse.json({ 
        message: `Sign-in link generated for ${targetEmail}.`, 
        signInLink: magicLink 
    }, { status: 200 });

  } catch (error: any) {
    console.error('[API GenLink] CRITICAL ERROR:', error);
    return NextResponse.json({
        error: 'Internal server error generating sign-in link',
        details: error.message
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic'; 