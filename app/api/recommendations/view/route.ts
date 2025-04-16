import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const cookieStore = cookies();
  const supabase = createRouteHandlerClient({ cookies: () => cookieStore });
  
  try {
    // Get the current session
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    
    if (sessionError || !session) {
      console.error('Authentication error:', sessionError);
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }
    
    // Parse the request body
    const requestData = await request.json();
    const { profile_id, source, referrer } = requestData;
    
    if (!profile_id) {
      return NextResponse.json(
        { error: 'Missing required profile_id field' },
        { status: 400 }
      );
    }
    
    // Save to interaction_history in Supabase
    await supabase.from('interaction_history').insert({
      user_id: session.user.id,
      interaction_type: 'PROFILE_VIEW',
      target_entity_type: 'PROFILE',
      target_entity_id: profile_id,
      metadata: {
        source: source || 'direct',
        referrer,
        from_recommendation: source === 'recommendation'
      }
    });
    
    // Call our backend API for profile view tracking
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      await fetch(`${API_URL}/profiles/${profile_id}/record-view`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`
        },
        body: JSON.stringify({
          source: source || 'direct',
          referrer,
          from_recommendation: source === 'recommendation'
        })
      });
    } catch (apiError) {
      console.error('Error calling backend API for profile view:', apiError);
      // Non-blocking - continue even if the API call fails
    }
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error in profile view API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 