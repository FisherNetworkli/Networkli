import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

export interface RecommendationScore {
  profileId: string;
  name: string;
  title: string | null;
  avatarUrl: string | null;
  company: string | null;
  score: number;
  matchReasons: string[];
}

/**
 * Get personalized recommendations from the API
 * Uses the enhanced recommendation engine on the backend
 */
export async function getRecommendations(userId: string, limit: number = 10): Promise<RecommendationScore[]> {
  try {
    // Call the API endpoint
    const response = await fetch(`/api/recommendations?userId=${userId}&limit=${limit}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    
    // Transform API response to the expected format
    return data.recommendations.map((rec: any) => ({
      profileId: rec.id,
      name: rec.full_name || `${rec.first_name || ''} ${rec.last_name || ''}`.trim(),
      title: rec.title,
      avatarUrl: rec.avatar_url,
      company: rec.company,
      score: Math.round(rec.similarity_score * 100), // Convert similarity score to a 0-100 scale
      matchReasons: rec.reason ? [rec.reason] : []
    }));
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    
    // Fall back to local recommendations if API fails
    return await getLocalRecommendations(userId, limit);
  }
}

/**
 * Fallback function that calculates recommendations locally
 * This is used when the API endpoint fails
 */
async function getLocalRecommendations(userId: string, limit: number = 10): Promise<RecommendationScore[]> {
  const supabase = createClientComponentClient();
  
  try {
    // Get current user's profile with skills and interests
    const { data: currentUser, error: userError } = await supabase
      .from('profiles')
      .select('*, user_preferences(*)')
      .eq('id', userId)
      .single();
    
    if (userError) throw userError;
    
    // Get all other profiles
    const { data: otherProfiles, error: profilesError } = await supabase
      .from('profiles')
      .select('*, user_preferences(*)')
      .neq('id', userId)
      .limit(50); // Get a reasonable amount to process
    
    if (profilesError) throw profilesError;
    if (!otherProfiles || otherProfiles.length === 0) {
      return [];
    }
    
    // Check if we already have connections with these users
    const { data: existingConnections, error: connectionsError } = await supabase
      .from('connections')
      .select('*')
      .or(`requester_id.eq.${userId},receiver_id.eq.${userId}`)
      .in('status', ['accepted', 'pending']);
    
    if (connectionsError) throw connectionsError;
    
    // Get connected user IDs to exclude
    const connectedUserIds = new Set<string>();
    existingConnections?.forEach(conn => {
      if (conn.requester_id === userId) {
        connectedUserIds.add(conn.receiver_id);
      } else {
        connectedUserIds.add(conn.requester_id);
      }
    });
    
    // Parse current user preferences
    const userSkills = Array.isArray(currentUser?.user_preferences?.skills) 
      ? currentUser.user_preferences.skills 
      : [];
      
    const userInterests = Array.isArray(currentUser?.user_preferences?.interests) 
      ? currentUser.user_preferences.interests 
      : [];
    
    const userIndustry = currentUser?.industry || '';
    
    // Calculate scores
    const recommendations: RecommendationScore[] = otherProfiles
      .filter(profile => !connectedUserIds.has(profile.id)) // Exclude existing connections
      .map(profile => {
        const profileSkills = Array.isArray(profile.user_preferences?.skills) 
          ? profile.user_preferences.skills 
          : [];
          
        const profileInterests = Array.isArray(profile.user_preferences?.interests) 
          ? profile.user_preferences.interests 
          : [];
        
        const profileIndustry = profile.industry || '';
        
        // Calculate match score
        let score = 0;
        const matchReasons: string[] = [];
        
        // Industry match (important)
        if (userIndustry && profileIndustry && userIndustry === profileIndustry) {
          score += 30;
          matchReasons.push('Same industry');
        }
        
        // Skills match
        const skillOverlap = userSkills.filter((skill: string) => 
          profileSkills.includes(skill)
        );
        
        if (skillOverlap.length > 0) {
          score += skillOverlap.length * 10;
          matchReasons.push(`${skillOverlap.length} shared skills`);
        }
        
        // Interests match
        const interestOverlap = userInterests.filter((interest: string) => 
          profileInterests.includes(interest)
        );
        
        if (interestOverlap.length > 0) {
          score += interestOverlap.length * 8;
          matchReasons.push(`${interestOverlap.length} shared interests`);
        }
        
        // Same company bonus
        if (userIndustry && profileIndustry && currentUser.company === profile.company) {
          score += 15;
          matchReasons.push('Same company');
        }
        
        return {
          profileId: profile.id,
          name: `${profile.first_name || ''} ${profile.last_name || ''}`.trim(),
          title: profile.title,
          avatarUrl: profile.avatar_url,
          company: profile.company,
          score,
          matchReasons
        };
      })
      .filter(rec => rec.score > 0) // Only include if there's some relevance
      .sort((a, b) => b.score - a.score) // Sort by score descending
      .slice(0, limit); // Limit results
      
    return recommendations;
  } catch (error) {
    console.error('Error in local recommendation algorithm:', error);
    return [];
  }
} 