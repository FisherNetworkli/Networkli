import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { stripe } from '@/lib/stripe';
import Stripe from 'stripe';

export async function POST(req: Request) {
  try {
    const session = await getServerSession(authOptions);
    const { plan, email } = await req.json();

    if (!plan) {
      return NextResponse.json(
        { error: 'Plan is required' },
        { status: 400 }
      );
    }

    let priceData: Stripe.Checkout.SessionCreateParams.LineItem.PriceData;
    
    if (plan === 'premium') {
      priceData = {
        currency: 'usd',
        product_data: {
          name: 'Premium Plan',
          description: 'Access to all premium features',
        },
        unit_amount: 1299, // $12.99
        recurring: {
          interval: 'month' as const,
        },
      };
    } else if (plan === 'organizer') {
      priceData = {
        currency: 'usd',
        product_data: {
          name: 'Organizer Plan',
          description: 'Full access to group and event management features',
        },
        unit_amount: 9900, // $99.00
        recurring: {
          interval: 'month' as const,
        },
      };
    } else {
      return NextResponse.json(
        { error: 'Invalid plan' },
        { status: 400 }
      );
    }

    // Get the base URL from environment or use a fallback
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3001';

    const checkoutSession = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: priceData,
          quantity: 1,
        },
      ],
      mode: 'subscription',
      success_url: `${baseUrl}/signup?plan=${plan}&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${baseUrl}/pricing?canceled=true`,
      customer_email: email || undefined,
      metadata: {
        userId: session?.user?.id || 'anonymous',
        plan,
      },
    });

    return NextResponse.json({ url: checkoutSession.url });
  } catch (error) {
    console.error('Error creating checkout session:', error);
    return NextResponse.json(
      { error: 'Failed to create checkout session' },
      { status: 500 }
    );
  }
} 