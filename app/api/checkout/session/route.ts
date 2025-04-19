import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { stripe } from '@/lib/stripe';

export async function POST(req: NextRequest) {
  // Initialize Supabase client with cookies
  const supabase = createServerComponentClient({ cookies });

  try {
    const { eventId } = await req.json();
    // Fetch event basics including organizer_id
    const { data: eventData, error: eventErr } = await supabase
      .from('events')
      .select('id, title, price, organizer_id')
      .eq('id', eventId)
      .single();
    if (eventErr || !eventData) {
      return NextResponse.json({ error: 'Event not found' }, { status: 404 });
    }

    const priceCents = eventData.price || 0;
    const title = eventData.title;
    // Manually fetch organizer's Stripe account from profiles
    const { data: organizerData, error: orgErr } = await supabase
      .from('profiles')
      .select('stripe_account_id')
      .eq('id', eventData.organizer_id)
      .single();
    if (orgErr || !organizerData) {
      return NextResponse.json(
        { error: 'Organizer Stripe account not configured' },
        { status: 400 }
      );
    }
    const stripeAccount = (organizerData as any).stripe_account_id as string;

    // Get current user for metadata
    const {
      data: { user }
    } = await supabase.auth.getUser();
    const userId = user?.id;
    const customerEmail = user?.email;

    // Prepare metadata for Stripe session
    const metadataForStripe: { [key: string]: string } = { eventId };
    if (userId) metadataForStripe.userId = userId;

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
      customer_email: customerEmail,
      metadata: metadataForStripe
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