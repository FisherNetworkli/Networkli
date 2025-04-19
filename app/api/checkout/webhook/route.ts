import { NextRequest, NextResponse } from 'next/server';
import Stripe from 'stripe';
import { stripe } from '@/lib/stripe';
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';

// Stripe webhook secret from env
const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET || '';

export async function POST(request: NextRequest) {
  const sig = request.headers.get('stripe-signature') || '';
  const body = await request.text();
  let event: Stripe.Event;
  
  try {
    event = stripe.webhooks.constructEvent(body, sig, webhookSecret);
  } catch (err: any) {
    console.error('Webhook signature verification failed:', err.message);
    return NextResponse.json({ error: 'Invalid signature' }, { status: 400 });
  }

  // Only handle checkout session completed
  if (event.type === 'checkout.session.completed') {
    const session = event.data.object as Stripe.Checkout.Session;
    const metadata = session.metadata || {};
    const eventId = metadata.eventId as string;
    const userId = metadata.userId as string;

    // Initialize Supabase client using cookies
    const supabase = createServerComponentClient({ cookies });

    try {
      // Insert attendance record
      const { error } = await supabase
        .from('event_attendance')
        .insert({ event_id: eventId, user_id: userId, status: 'attending' });
      if (error) {
        console.error('Error inserting attendance:', error);
      }
    } catch (dbErr) {
      console.error('Database error on webhook:', dbErr);
    }
  }

  return NextResponse.json({ received: true });
} 