import { NextRequest, NextResponse } from 'next/server';
import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';

export async function GET(request: NextRequest) {
  try {
    // Get authenticated user token
    const cookieStore = cookies();
    const supabase = createRouteHandlerClient({ cookies: () => cookieStore });
    
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (!session) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Get the time_period parameter from the request URL
    const { searchParams } = new URL(request.url);
    const timePeriod = searchParams.get('time_period') || '30d';

    // Forward the request to our backend API
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const backendResponse = await fetch(`${apiUrl}/organizer/dashboard?time_period=${timePeriod}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session.access_token}`
      }
    });

    if (!backendResponse.ok) {
      // If the organizer API fails, check if it's because the user isn't an organizer
      if (backendResponse.status === 403) {
        return NextResponse.json(
          { error: 'Access denied. This dashboard is only available to organizers.' },
          { status: 403 }
        );
      }
      
      const errorData = await backendResponse.json();
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch organizer dashboard data' },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in organizer dashboard API route:', error);
    return NextResponse.json(
      { error: 'Failed to fetch organizer dashboard data' },
      { status: 500 }
    );
  }
} 