import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { stripe } from '@/lib/stripe'
import { hash } from 'bcrypt'

export async function POST(req: Request) {
  try {
    const { name, email, password, plan, sessionId } = await req.json()

    if (!name || !email || !password) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      )
    }

    const supabase = createClient()

    // Check if user already exists
    const { data: existingUser } = await supabase
      .from('users')
      .select('id')
      .eq('email', email)
      .single()

    if (existingUser) {
      return NextResponse.json(
        { error: 'User already exists' },
        { status: 400 }
      )
    }

    // Hash password
    const hashedPassword = await hash(password, 10)

    // Create user
    const { data: user, error: userError } = await supabase
      .from('users')
      .insert({
        name,
        email,
        password: hashedPassword,
        role: plan === 'organizer' ? 'organizer' : 'user',
      })
      .select()
      .single()

    if (userError) {
      console.error('Error creating user:', userError)
      return NextResponse.json(
        { error: 'Failed to create user' },
        { status: 500 }
      )
    }

    // If this is a paid plan signup, update the subscription
    if (plan && sessionId) {
      try {
        // Retrieve the checkout session
        const session = await stripe.checkout.sessions.retrieve(sessionId)
        
        if (session.customer) {
          // Update the subscription with the user ID
          await supabase
            .from('subscriptions')
            .update({
              user_id: user.id,
            })
            .eq('stripe_customer_id', session.customer)
        }
      } catch (error) {
        console.error('Error updating subscription:', error)
        // Continue with registration even if subscription update fails
      }
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Registration error:', error)
    return NextResponse.json(
      { error: 'Registration failed' },
      { status: 500 }
    )
  }
} 