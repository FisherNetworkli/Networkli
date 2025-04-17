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

  // --- Auth Check ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  // TODO: Add more robust admin role check if necessary
  // --- End Auth Check ---

  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Prospect] Missing Supabase URL or Service Role Key.');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  // Use Admin client for elevated privileges needed for delete/insert
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false }
  });

  let prospectData: any;
  try {
    prospectData = await request.json();
    if (!prospectData || typeof prospectData !== 'object' || !prospectData.email) {
         throw new Error('Invalid prospect data received.');
    }
    console.log('[API Seed Prospect] Received prospect data:', prospectData);
  } catch (e) {
      console.error('[API Seed Prospect] Error parsing request body:', e);
      return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const prospectEmail = prospectData.email;

  let rlsDisabled = false;
  let authUserId: string | null = null; 
  let userExists = false;

  try {
     // --- Step 0: Check/Create Authentication User --- 
     console.log(`[API Seed Prospect] Attempting to create auth user for email: ${prospectEmail}...`);
     try {
       const { data: createData, error: createError } = await supabaseAdmin.auth.admin.createUser({
         email: prospectEmail,
         email_confirm: true,
         password: `DummyPass${Math.random().toString(36).slice(-6)}!`,
         user_metadata: { 
           first_name: prospectData.first_name, 
           last_name: prospectData.last_name,
         }
       });

       if (createError) {
         // Rethrow the original error to see if it's more specific than our catch block
         throw createError; 
       } 
       
       if (createData?.user?.id) {
          authUserId = createData.user.id;
          console.log(`[API Seed Prospect] Successfully created auth user with ID: ${authUserId}`);
       } else {
          throw new Error('User creation call succeeded but no user data/ID returned.');
       }

     } catch (error: any) {
        // Check if the specific error is "already registered" using the error code
        if (error?.code === 'email_exists') {
            userExists = true;
            console.log(`[API Seed Prospect] Auth user already exists for ${prospectEmail}. Attempting to fetch ID...`);
            
            // Try to fetch the existing user ID
            const { data: existingUserData, error: fetchError } = await supabaseAdmin.auth.admin.listUsers({
                 page: 1,
                 perPage: 1,
             });

             // Filter client-side if needed (less efficient)
             const foundUser = existingUserData?.users?.find(u => u.email === prospectEmail);

            if (fetchError) {
                console.error('[API Seed Prospect] Error fetching existing auth user ID:', fetchError);
                throw new Error(`Auth user exists but failed to fetch ID: ${fetchError.message}`); // Throw specific error
            }
            if (!foundUser || !foundUser.id) {
                 console.error('[API Seed Prospect] Failed to find existing user ID among fetched users.');
                 throw new Error('Auth user exists but could not retrieve ID.');
            }
            authUserId = foundUser.id;
            console.log(`[API Seed Prospect] Found existing auth user ID: ${authUserId}`);
            
        } else {
             // It's a different error, re-throw the original error from createUser
             console.error('[API Seed Prospect] Error during auth user creation (not email_exists):', error);
             throw error; 
        }
     }
     // --- End Step 0 ---
     
     // We must have an authUserId to proceed if profiles.id is linked
     if (!authUserId) {
         throw new Error('Could not determine authentication user ID to link profile.');
     }

    // --- Disable RLS before profile operations --- 
    console.log("[API Seed Prospect] Attempting to DISABLE RLS for profiles...");
    const { error: disableRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY;' });
    if (disableRlsError) {
        console.error('[API Seed Prospect] FAILED to disable RLS:', disableRlsError.message);
        throw new Error(`Failed to disable RLS for profiles: ${disableRlsError.message}`);
    }
    rlsDisabled = true;
    console.log("[API Seed Prospect] RLS disabled for profiles.");
    
    // --- Step 1: Prepare Profile Data --- 
    const profileToUpsert: any = {
        id: authUserId, // *** Assume profiles.id is linked to auth.users.id ***
        ...prospectData,
        email: prospectEmail,
        is_demo: true,
        is_prospect: true,
        is_celebrity: false
    };
    delete profileToUpsert.skillsString; 
    delete profileToUpsert.interestsString;
    delete profileToUpsert.goalsString;
    delete profileToUpsert.valuesString;

    // --- Step 2: Upsert Profile --- 
    const conflictTarget = 'email'; // *** Upsert based on the email constraint ***
    console.log(`[API Seed Prospect] Upserting profile for ID: ${authUserId} on conflict: ${conflictTarget}...`);
    const { data: upsertedData, error: upsertError } = await supabaseAdmin
        .from('profiles')
        .upsert(profileToUpsert, { onConflict: conflictTarget }) // Ensure profileToUpsert includes id: authUserId
        .select('id')
        .single();

    if (upsertError) {
        console.error('[API Seed Prospect] Error upserting profile:', upsertError);
        throw new Error(`Failed to upsert profile: ${upsertError.message}`);
    }
    if (!upsertedData || !upsertedData.id) {
         console.error('[API Seed Prospect] Profile Upsert succeeded but no ID returned.');
         throw new Error('Profile Upsert succeeded but failed to retrieve profile ID.');
    }
    const profileIdConfirmed = upsertedData.id;
    console.log(`[API Seed Prospect] Successfully upserted profile with ID: ${profileIdConfirmed}`);

    // --- Step 3: Clean up old prospect flags (remains the same) ---
    console.log(`[API Seed Prospect] Cleaning up old is_prospect flags...`);
    const { error: cleanupError } = await supabaseAdmin
        .from('profiles')
        .update({ is_prospect: false })
        // Use the confirmed ID for exclusion
        .neq('id', profileIdConfirmed) 
        .eq('is_prospect', true);
    if (cleanupError) {
        console.warn('[API Seed Prospect] Error cleaning up old prospect flags:', cleanupError.message);
    }

    // --- Success Response ---
    return NextResponse.json({
        message: `Prospect profile for ${prospectData.first_name || 'N/A'} seeded/updated successfully.`, 
        profileId: profileIdConfirmed // Return the confirmed ID
    }, { status: 200 });

  } catch (error: any) {
    // Log the potentially more specific error from the try block
    console.error('[API Seed Prospect] CRITICAL ERROR in main try block:', error);
    return NextResponse.json({
        error: 'Internal server error during prospect seeding',
        // Return the actual error message if available
        details: error.message || 'Unknown error' 
    }, { status: 500 });
  } finally {
    // --- Re-enable RLS in finally block --- 
    if (rlsDisabled) {
      console.log("[API Seed Prospect] Attempting to RE-ENABLE RLS for profiles...");
      const { error: enableRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;' });
      if (enableRlsError) {
          console.error('[API Seed Prospect] FAILED TO RE-ENABLE RLS for profiles:', enableRlsError.message);
          // Potentially alert monitoring, but don't fail the original request if it succeeded
      } else {
          console.log("[API Seed Prospect] RLS re-enabled for profiles.");
      }
    }
  }
}

export const dynamic = 'force-dynamic'; 