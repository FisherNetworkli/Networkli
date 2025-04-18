import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const DEMO_EMAIL = 'theprospect@yourdomain.com';
const DEMO_PASSWORD = 'DemoPassword123!';

if (!SUPABASE_URL || !SERVICE_ROLE_KEY) {
  console.error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false }
});

async function main() {
  // 1. Try to find existing Auth user by listing users and filtering by email
  let userId: string | undefined;
  let user = null;
  let getUserError = null;
  try {
    const { data, error } = await supabase.auth.admin.listUsers();
    if (error) throw error;
    user = data.users.find((u: any) => u.email === DEMO_EMAIL);
  } catch (e) {
    getUserError = e;
  }

  if (!user) {
    // 2. Create Auth user if not found
    console.log(`Creating demo Auth user: ${DEMO_EMAIL}`);
    const { data: created, error: createError } = await supabase.auth.admin.createUser({
      email: DEMO_EMAIL,
      password: DEMO_PASSWORD,
      email_confirm: true,
      user_metadata: {
        first_name: 'The',
        last_name: 'Prospect',
        is_demo: true,
      },
    });
    if (createError || !created?.user) {
      console.error('Failed to create demo Auth user:', createError?.message);
      process.exit(1);
    }
    userId = created.user.id;
    console.log('Created demo Auth user:', DEMO_EMAIL);
  } else {
    userId = user.id;
    console.log('Demo Auth user already exists:', DEMO_EMAIL);
  }

  // 3. Upsert profile row
  const { error: upsertError } = await supabase.from('profiles').upsert({
    id: userId,
    email: DEMO_EMAIL,
    first_name: 'The',
    last_name: 'Prospect',
    is_demo: true,
    is_celebrity: false,
    role: 'user',
  });
  if (upsertError) {
    console.error('Failed to upsert demo profile:', upsertError.message);
    process.exit(1);
  }
  console.log('Demo profile upserted for:', DEMO_EMAIL);
}

main().catch((err) => {
  console.error('Script failed:', err);
  process.exit(1);
}); 