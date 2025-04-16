import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import Stripe from 'stripe';
import { createClient } from '@/lib/supabase/server';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2025-03-31.basil',
});

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!;

interface StripeSubscription extends Omit<Stripe.Subscription, 'items'> {
  current_period_end: number;
  current_period_start: number;
  cancel_at_period_end: boolean;
  cancel_at: number | null;
  canceled_at: number | null;
  trial_start: number | null;
  trial_end: number | null;
  items: {
    data: Array<{
      price: Stripe.Price;
      quantity: number;
    }>;
  };
}

const handleCheckoutSession = async (session: Stripe.Checkout.Session) => {
  if (!session?.metadata?.userId) {
    throw new Error('No user ID in session metadata');
  }

  const subscriptionResponse = await stripe.subscriptions.retrieve(session.subscription as string);
  const subscription = subscriptionResponse as unknown as StripeSubscription;
  const supabase = createClient();
  
  await supabase
    .from('subscriptions')
    .upsert({
      user_id: session.metadata.userId,
      subscription_id: subscription.id,
      status: subscription.status,
      price_id: subscription.items.data[0].price.id,
      quantity: subscription.items.data[0].quantity || 1,
      current_period_start: new Date(subscription.current_period_start * 1000).toISOString(),
      current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
      cancel_at_period_end: subscription.cancel_at_period_end,
      cancel_at: subscription.cancel_at ? new Date(subscription.cancel_at * 1000).toISOString() : null,
      canceled_at: subscription.canceled_at ? new Date(subscription.canceled_at * 1000).toISOString() : null,
      trial_start: subscription.trial_start ? new Date(subscription.trial_start * 1000).toISOString() : null,
      trial_end: subscription.trial_end ? new Date(subscription.trial_end * 1000).toISOString() : null,
    });
  
  console.log('Subscription data saved to database');
}

const handleSubscriptionUpdate = async (subscription: StripeSubscription) => {
  const supabase = createClient();
  await supabase
    .from('subscriptions')
    .update({
      status: subscription.status,
      current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
    })
    .eq('subscription_id', subscription.id);
    
  console.log('Subscription updated in database');
}

const handleSubscriptionDelete = async (subscription: StripeSubscription) => {
  const supabase = createClient();
  await supabase
    .from('subscriptions')
    .update({
      status: 'canceled',
      current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
    })
    .eq('subscription_id', subscription.id);
    
  console.log('Subscription marked as canceled in database');
}

export async function POST(req: Request) {
  try {
    const body = await req.text();
    const signature = headers().get('stripe-signature')!;

    let event: Stripe.Event;

    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    } catch (err) {
      return new NextResponse(`Webhook Error: ${err instanceof Error ? err.message : 'Unknown Error'}`, { status: 400 });
    }

    switch (event.type) {
      case 'checkout.session.completed': {
        await handleCheckoutSession(event.data.object as Stripe.Checkout.Session);
        break;
      }
      case 'customer.subscription.updated': {
        await handleSubscriptionUpdate(event.data.object as unknown as StripeSubscription);
        break;
      }
      case 'customer.subscription.deleted': {
        await handleSubscriptionDelete(event.data.object as unknown as StripeSubscription);
        break;
      }
    }

    return new NextResponse(null, { status: 200 });
  } catch (error) {
    console.error('Error in webhook handler:', error);
    return new NextResponse('Webhook handler failed', { status: 500 });
  }
} 