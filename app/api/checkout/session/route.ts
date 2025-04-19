import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { stripe } from '@/lib/stripe';

export async function POST(req: NextRequest) {
  // Initialize Supabase client with cookies
  const supabase = createServerComponentClient({ cookies });

  try {
    const { eventId } = await req.json();
    // Fetch event with price and organizer Stripe account
    const { data: eventData, error: eventErr } = await supabase
      .from('events')
      .select(`
        id,
        title,
        price,
        organizer:organizer_id(stripe_account_id)
      `)
      .eq('id', eventId)
      .single();
    if (eventErr || !eventData) {
      return NextResponse.json({ error: 'Event not found' }, { status: 404 });
    }

    const priceCents = eventData.price || 0;
    const title = eventData.title;
    const stripeAccount = eventData.organizer?.stripe_account_id;
    if (!stripeAccount) {
      return NextResponse.json(
        { error: 'Organizer Stripe account not configured' },
        { status: 400 }
      );
    }

    // Get current user for metadata
    const {
      data: { user }
    } = await supabase.auth.getUser();
    const userId = user?.id;
    const customerEmail = user?.email;

    // Calculate application fee (10% commission)
    const applicationFee = Math.round(priceCents * 0.1);

    // Build Stripe Checkout session
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: { name: title },
            unit_amount: priceCents
          },
          quantity: 1
        }
      ],
      mode: 'payment',
      payment_intent_data: {
        application_fee_amount: applicationFee,
        transfer_data: { destination: stripeAccount }
      },
      success_url: `${baseUrl}/events/${eventId}?success=1`,
      cancel_url: `${baseUrl}/events/${eventId}?canceled=1`,
      customer_email: customerEmail || undefined,
      metadata: { eventId, userId }
    });

    return NextResponse.json({ url: session.url });
  } catch (err: any) {
    console.error('Error creating event checkout session:', err);
    return NextResponse.json(
      { error: 'Failed to create checkout session' },
      { status: 500 }
    );
  }
} 