'use client';

import { useState, useEffect } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { getRecommendations, RecommendationScore } from '@/app/utils/recommendationEngine';
import { Loader2, UserPlus, Info, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import toast from 'react-hot-toast';

export default function RecommendedConnectionsPage() {
  const [recommendations, setRecommendations] = useState<RecommendationScore[]>([]);
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState<Set<string>>(new Set());
  const supabase = createClientComponentClient();

  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        // Get current user
        const { data: userData, error: userError } = await supabase.auth.getUser();
        if (userError || !userData?.user) {
          console.error('Error getting user:', userError);
          return;
        }

        const userId = userData.user.id;
        
        // Get recommendations
        const recommendationResults = await getRecommendations(userId, 20);
        setRecommendations(recommendationResults);
      } catch (error) {
        console.error('Error fetching recommendations:', error);
        toast.error('Failed to load recommendations');
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [supabase]);

  // Track when user clicks on a recommendation
  const logRecommendationClick = async (profileId: string, rank: number) => {
    try {
      // Get current user 
      const { data: userData } = await supabase.auth.getUser();
      if (!userData?.user) return;

      // Log the click to the API
      await fetch('/api/recommendations/click', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          profile_id: profileId,
          source_page: 'network_recommended',
          rank: rank,
          algorithm_version: 'simple-attribute-matching-v1'
        }),
      });
    } catch (error) {
      console.error('Error logging recommendation click:', error);
      // Non-blocking - continue even if logging fails
    }
  };

  const handleConnect = async (profileId: string, rank: number) => {
    try {
      // Log the click
      await logRecommendationClick(profileId, rank);
      
      // Update UI immediately
      setConnecting(prev => new Set(prev).add(profileId));
      
      // Get current user
      const { data: userData, error: userError } = await supabase.auth.getUser();
      if (userError || !userData?.user) {
        throw new Error('User not authenticated');
      }

      const userId = userData.user.id;
      
      // Create connection request
      const { error } = await supabase
        .from('connections')
        .insert({
          requester_id: userId,
          receiver_id: profileId,
          status: 'pending'
        });

      if (error) throw error;
      
      // Show success toast
      toast.success('Connection request sent!');
      
      // Remove from recommendations
      setRecommendations(prev => prev.filter(rec => rec.profileId !== profileId));
    } catch (error) {
      console.error('Error connecting:', error);
      toast.error('Failed to send connection request');
    } finally {
      // Remove from connecting state
      setConnecting(prev => {
        const newSet = new Set(prev);
        newSet.delete(profileId);
        return newSet;
      });
    }
  };

  const handleProfileClick = async (profileId: string, rank: number) => {
    await logRecommendationClick(profileId, rank);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">Recommended Connections</h1>
        <p className="text-gray-600">
          People you might want to connect with based on your profile, skills, and interests.
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center items-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
          <span className="ml-2 text-lg">Finding your best matches...</span>
        </div>
      ) : recommendations.length === 0 ? (
        <div className="text-center py-20 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-medium mb-2">No recommendations found</h3>
          <p className="text-gray-600 mb-4">
            Complete your profile with more details to get better recommendations
          </p>
          <Button asChild>
            <Link href="/dashboard/profile">Update Profile</Link>
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {recommendations.map((recommendation, index) => (
            <Card key={recommendation.profileId} className="overflow-hidden">
              <Link 
                href={`/dashboard/profile/${recommendation.profileId}`}
                onClick={() => handleProfileClick(recommendation.profileId, index + 1)}
                className="block cursor-pointer hover:bg-gray-50 transition-colors"
              >
                <CardHeader className="bg-gray-50 pb-2">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center">
                      <div className="w-12 h-12 rounded-full overflow-hidden bg-gray-200 mr-3 flex items-center justify-center text-gray-500">
                        {recommendation.avatarUrl ? (
                          <Image
                            src={recommendation.avatarUrl}
                            alt={recommendation.name}
                            width={48}
                            height={48}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          recommendation.name.charAt(0).toUpperCase()
                        )}
                      </div>
                      <div>
                        <CardTitle className="text-base flex items-center">
                          {recommendation.name}
                          <ExternalLink className="h-3.5 w-3.5 ml-1.5 text-gray-400" />
                        </CardTitle>
                        {recommendation.title && (
                          <CardDescription className="text-sm">
                            {recommendation.title}
                            {recommendation.company && ` at ${recommendation.company}`}
                          </CardDescription>
                        )}
                      </div>
                    </div>
                    <div 
                      className="bg-blue-100 text-blue-800 rounded-full w-8 h-8 flex items-center justify-center"
                      title="Match score based on profile compatibility"
                    >
                      <span className="text-sm font-medium">{recommendation.score}</span>
                    </div>
                  </div>
                </CardHeader>
              </Link>
              <CardContent className="pt-4">
                <div className="mb-4">
                  <h4 className="text-sm font-medium flex items-center mb-2">
                    <Info className="h-4 w-4 mr-1" /> Why we recommend this connection
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {recommendation.matchReasons.map((reason, idx) => (
                      <Badge key={idx} variant="secondary">
                        {reason}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
              <CardFooter className="border-t pt-3">
                {connecting.has(recommendation.profileId) ? (
                  <Button disabled className="w-full">
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Connecting...
                  </Button>
                ) : (
                  <Button 
                    className="w-full" 
                    onClick={() => handleConnect(recommendation.profileId, index + 1)}
                  >
                    <UserPlus className="h-4 w-4 mr-2" />
                    Connect
                  </Button>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
} 