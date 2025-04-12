import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import { stripe } from '@/lib/stripe';
import { createClient } from '@/lib/supabase/server';

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!;

export async function POST(req: Request) {
  try {
    const body = await req.text();
    const signature = headers().get('stripe-signature');

    if (!signature) {
      return new NextResponse('No signature', { status: 400 });
    }

    const event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    const supabase = createClient();

    // Log the event type for debugging
    console.log(`Processing webhook event: ${event.type}`);

    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as any;
        console.log('Checkout session completed:', session.id);
        
        // Only proceed if this is a subscription checkout
        if (session.mode === 'subscription' && session.subscription) {
          try {
            const subscription = await stripe.subscriptions.retrieve(session.subscription);
            console.log('Subscription retrieved:', subscription.id);
            
            // Update user's subscription status in database
            await supabase
              .from('subscriptions')
              .upsert({
                user_id: session.metadata?.userId || 'anonymous',
                stripe_subscription_id: subscription.id,
                stripe_customer_id: subscription.customer as string,
                stripe_price_id: subscription.items.data[0].price.id,
                status: subscription.status,
                plan: session.metadata?.plan || 'unknown',
                current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
              });
            
            console.log('Subscription data saved to database');
          } catch (error) {
            console.error('Error processing subscription:', error);
          }
        }
        break;
      }

      case 'customer.subscription.updated': {
        const subscription = event.data.object as any;
        console.log('Subscription updated:', subscription.id);
        
        try {
          // Update subscription status in database
          await supabase
            .from('subscriptions')
            .update({
              status: subscription.status,
              current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
            })
            .eq('stripe_subscription_id', subscription.id);
          
          console.log('Subscription update saved to database');
        } catch (error) {
          console.error('Error updating subscription:', error);
        }
        break;
      }

      case 'customer.subscription.deleted': {
        const subscription = event.data.object as any;
        console.log('Subscription deleted:', subscription.id);
        
        try {
          // Update subscription status to canceled
          await supabase
            .from('subscriptions')
            .update({
              status: 'canceled',
              current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
            })
            .eq('stripe_subscription_id', subscription.id);
          
          console.log('Subscription cancellation saved to database');
        } catch (error) {
          console.error('Error canceling subscription:', error);
        }
        break;
      }
    }

    return new NextResponse(null, { status: 200 });
  } catch (error) {
    console.error('Webhook error:', error);
    return new NextResponse(
      'Webhook error: ' + (error as Error).message,
      { status: 400 }
    );
  }
} 