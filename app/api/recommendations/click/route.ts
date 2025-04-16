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
    const { profile_id, source_page, rank, algorithm_version } = requestData;
    
    if (!profile_id) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    // Save to interaction_history in Supabase
    await supabase.from('interaction_history').insert({
      user_id: session.user.id,
      interaction_type: 'RECOMMENDATION_CLICK',
      target_entity_type: 'PROFILE',
      target_entity_id: profile_id,
      metadata: {
        source_page,
        rank,
        algorithm_version
      }
    });
    
    // Also call our backend API for ML tracking
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      await fetch(`${API_URL}/recommendations/profile/click`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`
        },
        body: JSON.stringify({
          profile_id,
          source_page,
          rank,
          section: 'recommended_connections',
          algorithm_version
        })
      });
    } catch (apiError) {
      console.error('Error calling backend API:', apiError);
      // Non-blocking - continue even if the API call fails
    }
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error in recommendation click API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 