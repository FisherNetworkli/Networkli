'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, useAnimation, PanInfo } from 'framer-motion';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { User } from '@supabase/supabase-js';

import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Skeleton } from '@/app/components/ui/skeleton';
import { Badge } from '@/app/components/ui/badge';

interface Profile {
  id: string;
  full_name: string;
  title?: string;
  company?: string;
  bio?: string;
  industry?: string;
  location?: string;
  avatar_url?: string;
  skills?: string[];
  interests?: string[];
}

export default function SwipePage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [user, setUser] = useState<User | null>(null);
  const [swipeAnimation, setSwipeAnimation] = useState<'none' | 'left' | 'right'>('none');
  const controls = useAnimation();
  const cardRef = useRef<HTMLDivElement>(null);
  const supabase = createClientComponentClient();
  const router = useRouter();

  // Load user session and recommendations
  useEffect(() => {
    const fetchUserAndProfiles = async () => {
      try {
        // Get current user
        const { data: { session }, error: userError } = await supabase.auth.getSession();
        if (userError || !session) {
          console.error('Error fetching user:', userError);
          return;
        }
        
        setUser(session.user);
        
        // Fetch recommendations from the API
        const response = await fetch(`/api/recommendations?limit=10`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch recommendations');
        }
        
        const data = await response.json();
        setProfiles(data.recommendations || []);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchUserAndProfiles();
  }, [supabase]);

  // Track swipe interactions
  const recordInteraction = async (profileId: string, action: 'swipe_left' | 'swipe_right') => {
    try {
      const response = await fetch('/api/recommendations/interaction', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          interaction_type: action.toUpperCase(),
          target_entity_type: 'PROFILE',
          target_entity_id: profileId,
          metadata: {
            source_page: 'swipe_interface',
            algorithm_version: 'simple-attribute-matching-v1',
          }
        }),
      });
      
      if (!response.ok) {
        console.error('Failed to record interaction:', await response.text());
      }
    } catch (error) {
      console.error('Error recording interaction:', error);
    }
  };

  // Handle card swipe
  const handleDragEnd = async (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    const threshold = 100; // Min drag distance for a swipe
    const profile = profiles[currentIndex];
    
    if (!profile) return;
    
    if (info.offset.x > threshold) {
      // Swipe right - like
      setSwipeAnimation('right');
      await controls.start({ x: 500, opacity: 0, transition: { duration: 0.5 } });
      recordInteraction(profile.id, 'swipe_right');
      showNextProfile();
    } else if (info.offset.x < -threshold) {
      // Swipe left - pass
      setSwipeAnimation('left');
      await controls.start({ x: -500, opacity: 0, transition: { duration: 0.5 } });
      recordInteraction(profile.id, 'swipe_left');
      showNextProfile();
    } else {
      // Return to center if not enough movement
      controls.start({ x: 0, opacity: 1, transition: { duration: 0.5 } });
    }
  };

  // Move to the next profile
  const showNextProfile = () => {
    if (currentIndex < profiles.length - 1) {
      setCurrentIndex(prev => prev + 1);
      controls.start({ x: 0, opacity: 1, scale: 1, transition: { duration: 0 } });
      setSwipeAnimation('none');
    } else {
      // No more profiles
      setProfiles([]);
    }
  };

  // Button handlers
  const handleLike = async () => {
    const profile = profiles[currentIndex];
    if (!profile) return;
    
    setSwipeAnimation('right');
    await controls.start({ x: 500, opacity: 0, transition: { duration: 0.5 } });
    recordInteraction(profile.id, 'swipe_right');
    showNextProfile();
  };

  const handlePass = async () => {
    const profile = profiles[currentIndex];
    if (!profile) return;
    
    setSwipeAnimation('left');
    await controls.start({ x: -500, opacity: 0, transition: { duration: 0.5 } });
    recordInteraction(profile.id, 'swipe_left');
    showNextProfile();
  };

  // View profile details
  const viewProfile = (id: string) => {
    router.push(`/dashboard/profile/${id}`);
  };

  const currentProfile = profiles[currentIndex];

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-2xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-8">Network Connect</h1>
      
      {loading ? (
        <div className="w-full max-w-md">
          <Skeleton className="h-[500px] w-full rounded-xl mb-6" />
          <div className="flex justify-center space-x-4">
            <Skeleton className="h-12 w-12 rounded-full" />
            <Skeleton className="h-12 w-12 rounded-full" />
          </div>
        </div>
      ) : profiles.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-8 border border-gray-200 rounded-xl shadow-sm">
          <h2 className="text-xl font-medium mb-4">No more profiles to show</h2>
          <p className="text-gray-500 mb-4 text-center">
            We&apos;ve run out of recommendations for now. Check back later!
          </p>
          <Button onClick={() => router.push('/dashboard/network')}>
            Back to Network
          </Button>
        </div>
      ) : (
        <div className="w-full max-w-md relative h-[600px]">
          <motion.div
            ref={cardRef}
            drag="x"
            dragConstraints={{ left: 0, right: 0 }}
            onDragEnd={handleDragEnd}
            animate={controls}
            initial={{ opacity: 1, x: 0 }}
            className="absolute w-full"
          >
            <Card className="overflow-hidden rounded-xl shadow-md bg-white h-[500px] flex flex-col">
              <div className="relative h-72">
                {currentProfile?.avatar_url ? (
                  <Image
                    src={currentProfile.avatar_url}
                    alt={currentProfile.full_name}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-blue-300 to-purple-400 flex items-center justify-center">
                    <span className="text-4xl font-bold text-white">
                      {currentProfile?.full_name?.charAt(0) || '?'}
                    </span>
                  </div>
                )}
              </div>
              
              <div className="p-5 flex-1 overflow-y-auto">
                <div className="flex justify-between items-start mb-2">
                  <h2 className="text-xl font-bold">{currentProfile?.full_name}</h2>
                  <Badge variant="outline" className="bg-blue-50">
                    {currentProfile?.industry || 'Professional'}
                  </Badge>
                </div>
                
                {currentProfile?.title && (
                  <p className="text-gray-700 font-medium">
                    {currentProfile.title} {currentProfile.company ? `at ${currentProfile.company}` : ''}
                  </p>
                )}
                
                {currentProfile?.location && (
                  <p className="text-gray-500 text-sm mb-3">
                    📍 {currentProfile.location}
                  </p>
                )}
                
                {currentProfile?.bio && (
                  <p className="text-gray-600 mb-4 line-clamp-3">
                    {currentProfile.bio}
                  </p>
                )}
                
                {currentProfile?.skills && currentProfile.skills.length > 0 && (
                  <div className="mb-3">
                    <p className="text-sm font-medium text-gray-700 mb-1">Skills</p>
                    <div className="flex flex-wrap gap-1">
                      {currentProfile.skills.slice(0, 5).map((skill, i) => (
                        <Badge key={i} variant="secondary" className="bg-gray-100 text-gray-700">
                          {skill}
                        </Badge>
                      ))}
                      {currentProfile.skills.length > 5 && (
                        <Badge variant="secondary" className="bg-gray-100 text-gray-700">
                          +{currentProfile.skills.length - 5} more
                        </Badge>
                      )}
                    </div>
                  </div>
                )}
                
                <Button 
                  variant="outline" 
                  className="w-full mt-2"
                  onClick={() => viewProfile(currentProfile?.id)}
                >
                  View Full Profile
                </Button>
              </div>
            </Card>
            
            {swipeAnimation === 'left' && (
              <div className="absolute top-5 left-5 bg-red-500 text-white px-3 py-1 rounded-md transform -rotate-12">
                NOT NOW
              </div>
            )}
            
            {swipeAnimation === 'right' && (
              <div className="absolute top-5 right-5 bg-green-500 text-white px-3 py-1 rounded-md transform rotate-12">
                CONNECT
              </div>
            )}
          </motion.div>
          
          <div className="absolute bottom-0 left-0 right-0 flex justify-center space-x-10 p-4">
            <Button 
              variant="outline" 
              className="h-14 w-14 rounded-full border-2 border-gray-300"
              onClick={handlePass}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </Button>
            
            <Button 
              variant="outline" 
              className="h-14 w-14 rounded-full border-2 border-gray-300"
              onClick={handleLike}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </Button>
          </div>
        </div>
      )}
      
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>Swipe right to connect, left to skip</p>
        <p className="mt-1">Or use the buttons below to grow your network</p>
      </div>
    </div>
  );
} 