'use client';

import { useState, useEffect } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';
import { Loader2, Mail, MapPin, Building, Briefcase, Link as LinkIcon, UserPlus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { toast } from 'react-hot-toast';

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

export default function ProfileViewPage({ params }: { params: { id: string } }) {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const supabase = createClientComponentClient();
  const profileId = params.id;

  useEffect(() => {
    const getUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setCurrentUser(session.user);
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
        if (!sessionData.session) {
          throw new Error('No authenticated session');
        }

        // Fetch the profile from the API
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/profiles/${profileId}?source=profile_page`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${sessionData.session.access_token}`
          }
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch profile: ${response.statusText}`);
        }

        const profileData = await response.json();
        setProfile(profileData);

        // Track profile view from recommendation
        if (window.location.search.includes('from=recommendation')) {
          try {
            await fetch('/api/recommendations/view', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                profile_id: profileId,
                source: 'recommendation'
              }),
            });
          } catch (viewError) {
            console.error('Error logging profile view:', viewError);
            // Non-blocking - continue even if logging fails
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

  const handleConnect = async () => {
    if (!currentUser || !profile) return;
    
    setConnecting(true);
    try {
      // Create connection request
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

  // Check if this is a limited view
  const isLimitedView = profile.limited_view === true;

  return (
    <div className="container mx-auto py-6 max-w-5xl">
      {/* Profile Header Section */}
      <div className="bg-white rounded-lg border shadow-sm overflow-hidden mb-6">
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 h-40 relative">
          {/* Avatar */}
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
          
          {/* Connect Button */}
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
        
        {/* Profile Content */}
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

      {/* Additional Profile Information */}
      {!isLimitedView && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Skills */}
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

          {/* Interests */}
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

      {/* Limited View Message */}
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