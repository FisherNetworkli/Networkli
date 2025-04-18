import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!SUPABASE_URL || !SERVICE_ROLE_KEY) {
  console.error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false }
});

async function main() {
  // 1. Get all demo profiles
  const { data: profiles, error } = await supabase
    .from('profiles')
    .select('id, email, first_name, last_name')
    .eq('is_demo', true);

  if (error) throw error;
  if (!profiles || profiles.length === 0) {
    console.log('No demo profiles found.');
    return;
  }

  for (const profile of profiles) {
    // 2. Check if Auth user exists
    const { data: user, error: userError } = await supabase.auth.admin.getUserById(profile.id);
    if (userError || !user) {
      // 3. Create Auth user with the same ID and email
      console.log(`Creating Auth user for profile: ${profile.email} (ID: ${profile.id})`);
      const { data: createdUser, error: createError } = await supabase.auth.admin.createUser({
        email: profile.email,
        user_metadata: {
          first_name: profile.first_name,
          last_name: profile.last_name,
        },
        email_confirm: true,
        id: profile.id,
      });
      if (createError) {
        console.error(`Failed to create Auth user for ${profile.email}:`, createError.message);
      } else {
        console.log(`Created Auth user for ${profile.email}`);
      }
    } else {
      console.log(`Auth user already exists for: ${profile.email}`);
    }
  }
}

main().catch((err) => {
  console.error('Script failed:', err);
  process.exit(1);
}); 