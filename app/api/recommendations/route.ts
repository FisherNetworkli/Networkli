import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';
import { getUserProfile } from '@/lib/profile';

export async function GET(req: Request) {
  try {
    // Initialize Supabase client
    const cookieStore = cookies();
    const supabase = createClient(cookieStore);
    
    // Check authentication
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
    }
    
    // Get user profile for matching preferences
    const userProfile = await getUserProfile(user.id);
    if (!userProfile) {
      return NextResponse.json({ error: 'User profile not found' }, { status: 404 });
    }
    
    // Parse URL parameters
    const url = new URL(req.url);
    const limit = parseInt(url.searchParams.get('limit') || '10', 10);
    const exclude = url.searchParams.get('exclude')?.split(',') || [];
    
    // Add user id to exclude list to avoid recommending self
    if (!exclude.includes(user.id)) {
      exclude.push(user.id);
    }
    
    // Get profiles that match user interests and industry
    // Simple implementation - will be enhanced in Phase 3 & 4
    let query = supabase
      .from('profiles')
      .select(`
        id,
        user_id,
        first_name,
        last_name,
        headline,
        bio,
        avatar_url,
        current_company,
        current_role,
        industry,
        experience_level,
        skills,
        education,
        interests,
        location,
        availability_status,
        created_at,
        updated_at
      `)
      .not('user_id', 'in', `(${exclude.join(',')})`)
      .limit(limit);
    
    // Apply filters based on user profile if available
    if (userProfile.industry) {
      query = query.eq('industry', userProfile.industry);
    }
    
    // Execute query
    const { data: recommendations, error } = await query;
    
    if (error) {
      console.error('Error fetching recommendations:', error);
      return NextResponse.json({ error: 'Failed to fetch recommendations' }, { status: 500 });
    }
    
    // Check if we found enough recommendations
    if (recommendations.length < limit / 2 && userProfile.industry) {
      // If not enough matches with same industry, try without industry filter
      const { data: moreRecommendations, error: moreError } = await supabase
        .from('profiles')
        .select(`
          id,
          user_id,
          first_name,
          last_name,
          headline,
          bio,
          avatar_url,
          current_company,
          current_role,
          industry,
          experience_level,
          skills,
          education,
          interests,
          location,
          availability_status,
          created_at,
          updated_at
        `)
        .not('user_id', 'in', `(${exclude.join(',')})`)
        .not('user_id', 'in', `(${recommendations.map(r => r.user_id).join(',')})`)
        .limit(limit - recommendations.length);
      
      if (!moreError && moreRecommendations) {
        recommendations.push(...moreRecommendations);
      }
    }
    
    // Log this recommendation batch
    const { error: logError } = await supabase
      .from('recommendation_batches')
      .insert({
        user_id: user.id,
        count: recommendations.length,
        algorithm_version: 'v1-simple-industry-match'
      });
    
    if (logError) {
      console.error('Error logging recommendation batch:', logError);
    }
    
    return NextResponse.json({
      recommendations,
      metadata: {
        count: recommendations.length,
        algorithm: 'v1-simple-industry-match'
      }
    });
  } catch (error) {
    console.error('Recommendation API error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
} 