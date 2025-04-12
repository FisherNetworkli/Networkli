import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2025-03-31.basil',
});

export async function POST(req: Request) {
  try {
    const session = await getServerSession(authOptions);

    if (!session?.user) {
      return new NextResponse('Unauthorized', { status: 401 });
    }

    const { plan } = await req.json();

    let priceData: Stripe.Checkout.SessionCreateParams.LineItem.PriceData;
    
    if (plan === 'premium') {
      priceData = {
        currency: 'usd',
        product_data: {
          name: 'Premium Plan',
          description: 'Access to all premium features',
        },
        unit_amount: 1299, // $12.99 in cents
        recurring: {
          interval: 'month' as const,
        },
      };
    } else if (plan === 'organizer') {
      // For organizer plan, we'll redirect to a contact form
      return NextResponse.json({ 
        url: '/contact?plan=organizer',
        isContactForm: true 
      });
    } else {
      return new NextResponse('Invalid plan selected', { status: 400 });
    }

    const checkoutSession = await stripe.checkout.sessions.create({
      mode: 'subscription',
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: priceData,
          quantity: 1,
        },
      ],
      success_url: `${process.env.NEXTAUTH_URL}/dashboard?success=true`,
      cancel_url: `${process.env.NEXTAUTH_URL}/subscription?canceled=true`,
      client_reference_id: session.user.id,
      metadata: {
        userId: session.user.id,
        plan: plan,
      },
    });

    return NextResponse.json({ url: checkoutSession.url });
  } catch (error) {
    console.error('Error creating checkout session:', error);
    return new NextResponse('Error creating checkout session', { status: 500 });
  }
} 