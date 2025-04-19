'use client';

import { useState, useEffect } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';
import { Loader2, Mail, MapPin, Building, Briefcase, Link as LinkIcon, UserPlus, Users, UserCheck, Lightbulb, Sparkles, Calendar, Tv } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { toast } from 'react-hot-toast';
import Link from 'next/link';

interface UserProfile {
  id: string;
  full_name: string;
  headline: string;
  bio: string;
  location: string;
  website: string;
  avatar_url: string | null;
  title: string;
  company: string;
  industry: string;
  skills: string[];
  interests: string[];
  limited_view?: boolean;
}

type Recommendation = {
  id: string;
  name?: string;
  title?: string;
  avatar_url?: string;
  reason?: string;
  score?: number;
};

export default function ProfileViewPage({ params }: { params: { id: string } }) {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const supabase = createClientComponentClient();
  const profileId = params.id;

  const [profileRecommendations, setProfileRecommendations] = useState<Recommendation[]>([]);
  const [groupRecommendations, setGroupRecommendations] = useState<Recommendation[]>([]);
  const [eventRecommendations, setEventRecommendations] = useState<Recommendation[]>([]);
  const [recommendationLoading, setRecommendationLoading] = useState(false);

  useEffect(() => {
    const getUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setCurrentUser(session.user);
      } else {
        console.warn("No active session found for current user.");
      }
    };

    getUser();
  }, [supabase.auth]);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!profileId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const { data: sessionData } = await supabase.auth.getSession();
        if (!sessionData.session?.access_token) {
          throw new Error('No authenticated session or token');
        }

        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/profiles/${profileId}?source=profile_page`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${sessionData.session.access_token}`
          }
        });

        if (!response.ok) {
          let errorMsg = `Failed to fetch profile: ${response.statusText}`;
          try { const errorJson = await response.json(); errorMsg = errorJson.error || errorMsg; } catch(e){}
          throw new Error(errorMsg);
        }

        const profileData = await response.json();
        setProfile(profileData);

        if (window.location.search.includes('from=recommendation')) {
          try {
            await fetch('/api/recommendations/view', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionData.session.access_token}`
              },
              body: JSON.stringify({
                profile_id: profileId,
                source: 'recommendation'
              }),
            });
          } catch (viewError) {
            console.error('Error logging profile view:', viewError);
          }
        }
      } catch (err) {
        console.error('Error fetching profile:', err);
        setError(err instanceof Error ? err.message : 'Failed to load profile');
        toast.error('Could not load the profile');
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, [profileId, supabase]);

  useEffect(() => {
    const fetchRecommendations = async () => {
      if (!profileId || !currentUser) return;
      
      setRecommendationLoading(true);
      console.log(`[Profile Page] Fetching recommendations for profile: ${profileId}`);
      
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (!session?.access_token) {
          console.warn("[Profile Page Recs] No access token found for fetching recommendations.");
          setRecommendationLoading(false);
          return;
        }
        const accessToken = session.access_token;
        
        const headers: HeadersInit = {
          'Authorization': `Bearer ${accessToken}`
        };
        
        const recTypes: ('profile' | 'group' | 'event')[] = ['profile', 'group', 'event'];
        const promises = recTypes.map(type => 
          fetch(`/api/recommendations?profile_id=${profileId}&type=${type}&limit=3`, { headers })
            .then(res => {
               if (!res.ok) {
                   console.error(`[Profile Page Recs] API Error (${res.status}) fetching ${type} recs for ${profileId}`);
                   return { type, data: { recommendations: [] } };
               }
               return res.json().then(data => ({ type, data }));
            })
        );
        
        const results = await Promise.allSettled(promises);
        
        const fetchedRecommendations: { [key: string]: Recommendation[] } = {
          profile: [],
          group: [],
          event: []
        };

        results.forEach(result => {
          if (result.status === 'fulfilled' && result.value) {
            const { type, data } = result.value;
            fetchedRecommendations[type] = data.recommendations || [];
          }
        });

        console.log("[Profile Page Recs] Fetched recommendations:", fetchedRecommendations);
        setProfileRecommendations(fetchedRecommendations.profile);
        setGroupRecommendations(fetchedRecommendations.group);
        setEventRecommendations(fetchedRecommendations.event);

      } catch (err) {
        console.error('Error fetching recommendations on profile page:', err);
        toast.error('Could not load recommendations');
      } finally {
        setRecommendationLoading(false);
      }
    };

    if (profile && currentUser) { 
        fetchRecommendations();
    }
  }, [profileId, profile, currentUser, supabase]);

  const handleConnect = async () => {
    if (!currentUser || !profile) return;
    
    setConnecting(true);
    try {
      const { error } = await supabase
        .from('connections')
        .insert({
          requester_id: currentUser.id,
          receiver_id: profile.id,
          status: 'pending'
        });

      if (error) throw error;
      
      toast.success('Connection request sent!');
    } catch (error) {
      console.error('Error sending connection request:', error);
      toast.error('Failed to send connection request');
    } finally {
      setConnecting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600 mb-2" />
          <p>Loading profile...</p>
        </div>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="text-center py-10">
        <h2 className="text-xl font-semibold mb-2">Could not load profile</h2>
        <p className="text-gray-600 mb-4">{error || 'Profile not found'}</p>
        <Button onClick={() => window.history.back()}>Go Back</Button>
      </div>
    );
  }

  const isLimitedView = profile.limited_view === true;

  return (
    <div className="container mx-auto py-6 max-w-5xl">
      <div className="bg-white rounded-lg border shadow-sm overflow-hidden mb-6">
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 h-40 relative">
          <div className="absolute bottom-0 left-8 transform translate-y-1/2">
            <div className="w-32 h-32 rounded-full border-4 border-white bg-white overflow-hidden flex items-center justify-center">
              {profile.avatar_url ? (
                <img 
                  src={profile.avatar_url} 
                  alt={profile.full_name} 
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-gray-200 text-gray-600 text-4xl font-bold">
                  {profile.full_name.charAt(0)}
                </div>
              )}
            </div>
          </div>
          
          <div className="absolute bottom-4 right-4">
            {currentUser && currentUser.id !== profileId && (
              <Button 
                onClick={handleConnect}
                disabled={connecting}
                className="px-4 py-2 rounded-md text-sm font-medium shadow"
              >
                {connecting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <UserPlus className="h-4 w-4 mr-2" />
                    Connect
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
        
        <div className="mt-20 p-8">
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold">{profile.full_name}</h2>
              <p className="text-gray-600 mt-1">
                {profile.title}
                {profile.company && ` at ${profile.company}`}
              </p>
              
              {profile.location && (
                <div className="flex items-center mt-2 text-gray-500 text-sm">
                  <MapPin className="h-4 w-4 mr-1" />
                  {profile.location}
                </div>
              )}
              
              {profile.website && !isLimitedView && (
                <div className="flex items-center mt-1 text-blue-600 text-sm">
                  <LinkIcon className="h-4 w-4 mr-1" />
                  <a href={profile.website} target="_blank" rel="noopener noreferrer">
                    {profile.website}
                  </a>
                </div>
              )}
            </div>
            
            {!isLimitedView && profile.bio && (
              <div>
                <h3 className="text-lg font-medium mb-2">About</h3>
                <p className="text-gray-600">{profile.bio}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {!isLimitedView && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {profile.skills && profile.skills.length > 0 && (
            <Card>
              <CardHeader>
                <h3 className="text-lg font-medium">Skills</h3>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {profile.skills.map((skill, index) => (
                    <Badge key={index} variant="secondary">{skill}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {profile.interests && profile.interests.length > 0 && (
            <Card>
              <CardHeader>
                <h3 className="text-lg font-medium">Interests</h3>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {profile.interests.map((interest, index) => (
                    <Badge key={index} variant="outline">{interest}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {!isLimitedView && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-4">Suggestions for You</h3>
          {recommendationLoading ? (
            <div className="flex justify-center items-center p-6">
              <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              <span className="ml-2">Loading suggestions...</span>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {profileRecommendations.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center">
                      <Users className="h-5 w-5 mr-2 text-blue-600"/>
                      Suggested Connections
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-3">
                      {profileRecommendations.map(rec => (
                        <li key={rec.id} className="flex items-center space-x-3 text-sm">
                          <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                             {rec.avatar_url ? (
                                <img src={rec.avatar_url} alt={rec.name} className="h-full w-full rounded-full object-cover"/>
                             ) : <UserCheck className="h-4 w-4 text-gray-500"/>}
                          </div>
                          <div>
                            <Link href={`/dashboard/profile/${rec.id}?from=recommendation`} className="font-medium text-gray-800 hover:text-blue-600">
                              {rec.name || rec.title || 'View Profile'}
                            </Link>
                            {rec.reason && <p className="text-xs text-gray-500 italic">{rec.reason}</p>}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {groupRecommendations.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center">
                      <Lightbulb className="h-5 w-5 mr-2 text-purple-600"/>
                      Recommended Groups
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                     <ul className="space-y-3">
                      {groupRecommendations.map(rec => (
                        <li key={rec.id} className="flex items-center space-x-3 text-sm">
                           <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                             <Sparkles className="h-4 w-4 text-gray-500"/>
                           </div>
                          <div>
                            <Link href={`/groups/${rec.id}`} className="font-medium text-gray-800 hover:text-purple-600">
                              {rec.name || 'View Group'}
                            </Link>
                            {rec.reason && <p className="text-xs text-gray-500 italic">{rec.reason}</p>}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              {eventRecommendations.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center">
                      <Calendar className="h-5 w-5 mr-2 text-pink-600"/>
                      Recommended Events
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                     <ul className="space-y-3">
                      {eventRecommendations.map(rec => (
                        <li key={rec.id} className="flex items-center space-x-3 text-sm">
                           <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                             <Tv className="h-4 w-4 text-gray-500"/>
                           </div>
                          <div>
                             <Link href={`/events/${rec.id}`} className="font-medium text-gray-800 hover:text-pink-600">
                              {rec.title || 'View Event'}
                            </Link>
                            {rec.reason && <p className="text-xs text-gray-500 italic">{rec.reason}</p>}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>
      )}

      {isLimitedView && (
        <Card className="mt-6">
          <CardContent className="py-6">
            <div className="text-center">
              <h3 className="text-lg font-medium mb-2">Limited Profile View</h3>
              <p className="text-gray-600 mb-4">
                Upgrade to Premium to see full profile details and get more insights.
              </p>
              <Button asChild>
                <a href="/pricing">Upgrade to Premium</a>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 