import { NextResponse, NextRequest } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';
import { getUserProfile } from '@/lib/profile';
import { User } from '@supabase/supabase-js';

export async function GET(req: NextRequest) {
  let user: User | null = null;
  let authError: any = null;
  let recommendations: any[] = []; // Define recommendations array outside the try block

  try {
    // Initialize Supabase client
    const cookieStore = cookies();
    const supabase = createClient(cookieStore);
    
    // --- Prioritize Authorization Header --- 
    const authHeader = req.headers.get('Authorization');
    if (authHeader && authHeader.startsWith('Bearer ')) {
        const token = authHeader.split(' ')[1];
        console.log("[API Recs] Found Bearer token in header. Verifying...");
        const { data: { user: userFromToken }, error: tokenError } = await supabase.auth.getUser(token);
        if (tokenError) {
            console.error("[API Recs] Token verification error:", tokenError.message);
            authError = tokenError; // Store error but don't immediately fail
        } else {
            user = userFromToken; // Assign user if token is valid
            console.log("[API Recs] Token verified successfully.");
        }
    }
    // --- End Prioritize Header ---

    // --- Fallback to Cookie Auth if no user yet --- 
    if (!user && !authError) { // Only try cookie auth if header didn't work or wasn't present
        console.log("[API Recs] No valid Bearer token found or provided. Falling back to cookie auth...");
        const { data: { user: userFromCookie }, error: cookieError } = await supabase.auth.getUser();
        if (cookieError) {
            console.error("[API Recs] Cookie auth error:", cookieError.message);
            authError = cookieError;
        } else {
            user = userFromCookie;
             console.log("[API Recs] Cookie auth successful.");
        }
    }
    // --- End Fallback --- 
    
    // --- Final Auth Check --- 
    if (!user) {
      const errorMessage = authError ? `Authentication failed: ${authError.message}` : 'Not authenticated';
      console.error("[API Recs] Final auth check failed:", errorMessage);
      return NextResponse.json({ error: errorMessage }, { status: 401 });
    }
    // --- End Final Auth Check ---
    
    console.log(`[API Recs] Authenticated user ID: ${user.id}`);

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
        first_name,
        last_name,
        headline,
        bio,
        avatar_url,
        industry,
        experience_level,
        skills,
        interests,
        location,
        created_at,
        updated_at
      `)
      .not('id', 'in', `(${exclude.join(',')})`)
      .limit(limit);
    
    // Apply filters based on user profile if available
    if (userProfile.industry) {
      query = query.eq('industry', userProfile.industry);
    }
    
    // Execute query
    // Assign to the outer recommendations variable
    const { data: initialRecommendations, error } = await query;
    
    if (error) {
      console.error('Error fetching recommendations:', error);
      return NextResponse.json({ error: 'Failed to fetch recommendations' }, { status: 500 });
    }
    
    // Assign initial results
    recommendations = initialRecommendations || [];

    // Check if we found enough recommendations
    if (recommendations.length < limit / 2 && userProfile.industry) {
      // If not enough matches with same industry, try without industry filter
      console.log('[API Recs] Not enough matches with industry filter, trying without...');
      const { data: moreRecommendations, error: moreError } = await supabase
        .from('profiles')
        .select(`
          id,
          first_name,
          last_name,
          headline,
          bio,
          avatar_url,
          industry,
          experience_level,
          skills,
          interests,
          location,
          created_at,
          updated_at
        `)
        .not('id', 'in', `(${exclude.join(',')})`)
        .not('id', 'in', `(${recommendations.map(r => r.id).join(',')})`) // Exclude already fetched recommendations
        .limit(limit - recommendations.length);
      
      if (!moreError && moreRecommendations) {
        console.log(`[API Recs] Fetched ${moreRecommendations.length} additional recommendations.`);
        recommendations.push(...moreRecommendations);
      } else if (moreError) {
         console.error('Error fetching more recommendations:', moreError);
      }
    }
    
    // Log this recommendation batch
    const { error: logError } = await supabase
      .from('recommendation_batches')
      .insert({
        user_id: user.id,
        count: recommendations.length,
        algorithm_version: 'v1-simple-industry-match' // Update if algorithm changes
      });
    
    if (logError) {
      // Log but don't fail the request
      console.error('Error logging recommendation batch:', logError);
    }
    
  } catch (error) {
    console.error('Recommendation API error:', error);
    // Return empty recommendations on general error
    return NextResponse.json({ 
        recommendations: [], 
        metadata: { count: 0, algorithm: 'error' },
        error: 'Internal server error' 
    }, { status: 500 });
  }

  // Return successful response with recommendations
  return NextResponse.json({
    recommendations,
    metadata: {
      count: recommendations.length,
      algorithm: 'v1-simple-industry-match' // Update if algorithm changes
    }
  });
}