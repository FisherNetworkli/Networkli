import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

// --- Assumed Data Interfaces ---
interface EventSeedData {
  title: string;
  description?: string | null;
  category?: string | null;
  industry?: string | null;
  date: string; // ISO string
  location?: string | null;
  format?: string | null;
  organizer_id?: string | null; // Assign randomly from demo users
  is_demo: boolean;
}

interface AttendanceInsert {
  event_id: string;
  user_id: string;
  role: string; // e.g., 'attendee'
  is_demo: boolean;
}

// Helper function to pick random element
const getRandomElement = <T>(arr: T[]): T | undefined => {
  if (arr.length === 0) return undefined;
  return arr[Math.floor(Math.random() * arr.length)];
};

export async function POST(request: Request) {
  const cookieStore = cookies();
  const supabaseUserClient = createRouteHandlerClient({ cookies: () => cookieStore });

  // --- Auth Check ---
  const { data: { session } } = await supabaseUserClient.auth.getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  // --- End Auth Check ---

  if (!supabaseUrl || !supabaseServiceRoleKey) {
    console.error('[API Seed Events Combined] Missing Supabase URL or Service Role Key.');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false }
  });

  // --- Configurable options ---
  const minAttendeesPerEvent = 5;
  const maxAttendeesPerEvent = 10; // Keep relatively low for demo
  // --- End Configurable options ---

  let eventsToSeed: EventSeedData[] = [];
  try {
    const body = await request.json();
    if (body.events && Array.isArray(body.events)) {
        eventsToSeed = body.events.filter((e: any) => e.title && e.is_demo === true && e.date);
    } else {
        return NextResponse.json({ error: 'Invalid request body: missing events array.' }, { status: 400 });
    }
    if (eventsToSeed.length === 0) {
         return NextResponse.json({ message: 'No valid events provided to seed.', seededEvents: 0, seededAttendance: 0 }, { status: 200 });
    }
  } catch (e) {
      console.error('[API Seed Events Combined] Error parsing request body:', e);
      return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  console.log(`[API Seed Events Combined] Starting process for ${eventsToSeed.length} events...`);

  let seededEventsCount = 0;
  let seededAttendanceCount = 0;
  const errors: string[] = [];
  let eventsRlsDisabled = false;
  let attendanceRlsDisabled = false;

  try {
    // --- 1. Fetch Demo Users ---
    console.log("[API Seed Events Combined] Fetching demo users...");
    const { data: users, error: userError } = await supabaseAdmin
        .from('profiles').select('id').eq('is_demo', true);
    if (userError) throw new Error(`Failed to fetch demo users: ${userError.message}`);
    if (!users || users.length === 0) throw new Error('No demo users found.');
    const userIds = users.map(u => u.id);
    console.log(`[API Seed Events Combined] Found ${userIds.length} users.`);

    // --- 2. Prepare Event Data (Assign random organizers) ---
    const eventsWithOrganizers = eventsToSeed.map(event => ({
        ...event,
        organizer_id: getRandomElement(userIds), // Assign a random demo user as organizer
    }));

    // --- 3. Seed Events and get IDs ---
    console.log("[API Seed Events Combined] Attempting to DISABLE RLS for events...");
    const { error: disableEventsRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.events DISABLE ROW LEVEL SECURITY;' });
    if (disableEventsRlsError) throw new Error(`Failed to disable RLS for events: ${disableEventsRlsError.message}`);
    eventsRlsDisabled = true;
    console.log("[API Seed Events Combined] RLS disabled for events.");

    console.log(`[API Seed Events Combined] Inserting ${eventsWithOrganizers.length} events and selecting IDs...`);
    const { data: insertedEvents, error: insertEventsError } = await supabaseAdmin
        .from('events')
        .insert(eventsWithOrganizers)
        .select('id'); // Select ONLY the ID of newly inserted events

    if (insertEventsError) {
        console.error(`[API Seed Events Combined] Event Insert Error:`, insertEventsError);
        throw new Error(`Event Insert Error: ${insertEventsError.message}`); // Throw to stop execution
    }
    seededEventsCount = insertedEvents?.length ?? 0;
    const newEventIds = insertedEvents?.map(e => e.id) || [];
    console.log(`[API Seed Events Combined] Inserted ${seededEventsCount} events.`);

    if (newEventIds.length === 0) {
        console.log("[API Seed Events Combined] No new events were inserted, skipping attendance seeding.");
    } else {
        // --- 4. Generate & Seed Attendance ---
        console.log("[API Seed Events Combined] Generating attendance records...");
        const attendanceToInsert: AttendanceInsert[] = [];
        const existingAttendance = new Set<string>(); // To prevent duplicate user-event pairs

        newEventIds.forEach(eventId => {
            const attendeeCount = Math.floor(Math.random() * (maxAttendeesPerEvent - minAttendeesPerEvent + 1)) + minAttendeesPerEvent;
            const shuffledUserIds = [...userIds].sort(() => 0.5 - Math.random()); // Shuffle users for variety

            for (let i = 0; i < attendeeCount && i < shuffledUserIds.length; i++) {
                const userId = shuffledUserIds[i];
                const key = `${userId}-${eventId}`;
                if (!existingAttendance.has(key)) {
                    attendanceToInsert.push({
                        event_id: eventId,
                        user_id: userId,
                        role: 'attendee', // Assuming 'attendee' role
                        is_demo: true
                    });
                    existingAttendance.add(key);
                }
            }
        });

        if (attendanceToInsert.length > 0) {
            console.log("[API Seed Events Combined] Attempting to DISABLE RLS for event_attendees...");
            // *** ASSUMING table name is 'event_attendees' ***
            const { error: disableAttendRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.event_attendees DISABLE ROW LEVEL SECURITY;' });
            if (disableAttendRlsError) throw new Error(`Failed to disable RLS for event_attendees: ${disableAttendRlsError.message}`);
            attendanceRlsDisabled = true;
            console.log("[API Seed Events Combined] RLS disabled for event_attendees.");

            console.log(`[API Seed Events Combined] Inserting ${attendanceToInsert.length} attendance records...`);
            // *** ASSUMING table name is 'event_attendees' ***
            const { data: insertedAttendance, error: insertAttendError } = await supabaseAdmin
                .from('event_attendees') // *** Check table name ***
                .insert(attendanceToInsert)
                .select(); // Select to confirm insertion

            if (insertAttendError) {
                console.error(`[API Seed Events Combined] Attendance Insert Error:`, insertAttendError);
                // Don't throw, just record error and count
                errors.push(`Attendance Insert Error: ${insertAttendError.message}`);
                seededAttendanceCount = 0; // Indicate failure for this part
            } else {
                seededAttendanceCount = insertedAttendance?.length ?? attendanceToInsert.length; // Use length of what was meant to be inserted as fallback count
                console.log(`[API Seed Events Combined] Inserted ${seededAttendanceCount} attendance records.`);
            }
        } else {
            console.log("[API Seed Events Combined] No new attendance records generated.");
        }
    }

    // --- Success Response ---
    console.log("[API Seed Events Combined] Seeding process finished before RLS re-enable.");
    if (errors.length > 0) {
       // Report partial success with errors
       return NextResponse.json({
           message: `Completed with ${errors.length} errors during attendance insert. First error: ${errors[0]}`,
           seededEvents: seededEventsCount,
           seededAttendance: seededAttendanceCount, // Report count even if errors occurred
           errors: errors
       }, { status: 207 }); // 207 Multi-Status for partial success
    }
    return NextResponse.json({
        message: `${seededEventsCount} events and ${seededAttendanceCount} attendance records seeded successfully.`,
        seededEvents: seededEventsCount,
        seededAttendance: seededAttendanceCount
    }, { status: 200 });

  } catch (error: any) {
    console.error('[API Seed Events Combined] CRITICAL ERROR:', error);
    errors.push(error.message);
    // Ensure counts reflect state before the critical error if possible
    return NextResponse.json({
        error: 'Internal server error during combined event/attendance seeding',
        details: error.message,
        seededEvents: seededEventsCount, // Events might have been seeded before error
        seededAttendance: 0, // Attendance likely failed
        errors: errors
    }, { status: 500 });

  } finally {
     // --- Re-enable RLS ---
    if (attendanceRlsDisabled) {
      console.log("[API Seed Events Combined] Attempting to RE-ENABLE RLS for event_attendees...");
      // *** ASSUMING table name is 'event_attendees' ***
      const { error: enableRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.event_attendees ENABLE ROW LEVEL SECURITY;' });
      if (enableRlsError) console.error('[API Seed Events Combined] FAILED TO RE-ENABLE RLS for event_attendees:', enableRlsError);
      else console.log("[API Seed Events Combined] RLS re-enabled for event_attendees.");
    }
    if (eventsRlsDisabled) {
      console.log("[API Seed Events Combined] Attempting to RE-ENABLE RLS for events...");
      const { error: enableRlsError } = await supabaseAdmin.rpc('execute_sql', { sql: 'ALTER TABLE public.events ENABLE ROW LEVEL SECURITY;' });
      if (enableRlsError) console.error('[API Seed Events Combined] FAILED TO RE-ENABLE RLS for events:', enableRlsError);
      else console.log("[API Seed Events Combined] RLS re-enabled for events.");
    }
  }
}

export const dynamic = 'force-dynamic'; 