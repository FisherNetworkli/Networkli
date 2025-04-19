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

// --- Define unified swipe item type ---
type SwipeItem = {
  id: string;
  type: 'user' | 'group' | 'event';
  name?: string;         // for user full name or group name
  title?: string;        // for profile title or event title
  avatar_url?: string;   // for user avatar
  industry?: string;     // for profile or group metadata
  date?: string;         // for event date
  location?: string;     // for profile or event location
  company?: string;
  skills?: string[];
};

export default function SwipePage() {
  const [items, setItems] = useState<SwipeItem[]>([]);
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
        // 1) Authenticate and get user ID
        const { data: { session }, error: userError } = await supabase.auth.getSession();
        if (userError || !session) throw new Error('Not authenticated');
        setUser(session.user);
        const userId = session.user.id;
        // Demo override for our showcase account
        const DEMO_USER_ID = 'b2ebcc2a-74db-4f27-b313-7b6031f7e610';
        if (userId === DEMO_USER_ID) {
          // Hardcoded demo swipe items
          const demoItems: SwipeItem[] = [
            { id: 'demo-user-1', type: 'user', name: 'Elon Musk', title: 'CEO @ SpaceX', avatar_url: '/placeholder-avatar.png', skills: ['Innovation', 'Leadership'] },
            { id: 'demo-user-2', type: 'user', name: 'Marie Curie', title: 'Physicist', avatar_url: '/placeholder-avatar.png', skills: ['Research', 'Chemistry'] },
            { id: 'demo-grp-1', type: 'group', name: 'Tech Pioneers', industry: 'Technology' },
            { id: 'demo-grp-2', type: 'group', name: 'Design Gurus', industry: 'Design' },
            { id: 'demo-evt-1', type: 'event', title: 'AI Future Summit', date: '2025-08-01', location: 'San Francisco' },
            { id: 'demo-evt-2', type: 'event', title: 'Product Hackathon', date: '2025-09-15', location: 'New York' }
          ];
          setItems(demoItems);
          setLoading(false);
          return;
        }

        // 2) Fetch peers, groups, events in parallel for non-demo users
        const [uRes, gRes, eRes] = await Promise.all([
          fetch(`/api/recommend/users/${userId}?limit=10`),
          fetch(`/api/recommend/groups/${userId}?limit=10`),
          fetch(`/api/recommend/events/${userId}?limit=10`)
        ]);
        const [uJson, gJson, eJson] = await Promise.all([uRes.json(), gRes.json(), eRes.json()]);

        // 3) Normalize into SwipeItem[]
        const userItems: SwipeItem[] = (uJson as any[]).map(u => ({
          id: u.profile_id || u.id,
          type: 'user',
          name: u.first_name + ' ' + u.last_name,
          title: u.headline || u.title,
          company: u.company,
          avatar_url: u.avatar_url,
          industry: u.industry,
          skills: u.skills
        }));
        const groupItems: SwipeItem[] = (gJson as any[]).map(g => ({
          id: g.group_id || g.id,
          type: 'group',
          name: g.name,
          industry: g.category
        }));
        const eventItems: SwipeItem[] = (eJson as any[]).map(e => ({
          id: e.event_id || e.id,
          type: 'event',
          title: e.name || e.title,
          date: e.date,
          location: e.location
        }));

        // 4) Combine and set into state
        setItems([...userItems, ...groupItems, ...eventItems]);
      } catch (error) {
        console.error('Error fetching swipe items:', error);
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
    const item = items[currentIndex];
    
    if (!item) return;
    
    if (info.offset.x > threshold) {
      // Swipe right - like
      setSwipeAnimation('right');
      await controls.start({ x: 500, opacity: 0, transition: { duration: 0.5 } });
      recordInteraction(item.id, 'swipe_right');
      showNextProfile();
    } else if (info.offset.x < -threshold) {
      // Swipe left - pass
      setSwipeAnimation('left');
      await controls.start({ x: -500, opacity: 0, transition: { duration: 0.5 } });
      recordInteraction(item.id, 'swipe_left');
      showNextProfile();
    } else {
      // Return to center if not enough movement
      controls.start({ x: 0, opacity: 1, transition: { duration: 0.5 } });
    }
  };

  // Move to the next profile
  const showNextProfile = () => {
    if (currentIndex < items.length - 1) {
      setCurrentIndex(prev => prev + 1);
      controls.start({ x: 0, opacity: 1, scale: 1, transition: { duration: 0 } });
      setSwipeAnimation('none');
    } else {
      // No more profiles
      setItems([]);
    }
  };

  // Button handlers
  const handleLike = async () => {
    const item = items[currentIndex];
    if (!item) return;
    
    setSwipeAnimation('right');
    await controls.start({ x: 500, opacity: 0, transition: { duration: 0.5 } });
    recordInteraction(item.id, 'swipe_right');
    showNextProfile();
  };

  const handlePass = async () => {
    const item = items[currentIndex];
    if (!item) return;
    
    setSwipeAnimation('left');
    await controls.start({ x: -500, opacity: 0, transition: { duration: 0.5 } });
    recordInteraction(item.id, 'swipe_left');
    showNextProfile();
  };

  // View profile details
  const viewProfile = (id: string) => {
    router.push(`/dashboard/profile/${id}`);
  };

  const currentItem = items[currentIndex];

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
      ) : items.length === 0 ? (
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
              {/* Type Label */}
              <div className="absolute top-4 left-4 z-10 bg-white bg-opacity-80 px-3 py-1 rounded text-sm font-semibold">
                {currentItem?.type === 'user' ? 'Individual' : currentItem?.type === 'group' ? 'Group' : 'Event'}
              </div>
              <div className="relative h-72">
                {/* Render card differently by item.type */}
                {currentItem?.type === 'user' && currentItem.avatar_url ? (
                  <Image
                    src={currentItem.avatar_url!}
                    alt={currentItem.name!}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-blue-300 to-purple-400 flex items-center justify-center">
                    <span className="text-4xl font-bold text-white">
                      {currentItem?.name?.charAt(0) || '?'}
                    </span>
                  </div>
                )}
              </div>
              
              <div className="p-5 flex-1 overflow-y-auto">
                <div className="flex justify-between items-start mb-2">
                  <h2 className="text-xl font-bold">
                    {currentItem?.type === 'user' && currentItem.name}
                    {currentItem?.type === 'group' && currentItem.name}
                    {currentItem?.type === 'event' && currentItem.title}
                  </h2>
                  <Badge variant="outline" className="bg-blue-50">
                    {currentItem?.industry || 'Professional'}
                  </Badge>
                </div>
                
                {/* Show subtitle for user or event */}
                {currentItem?.type === 'user' && currentItem.title && (
                  <p className="text-gray-700 font-medium">
                    {currentItem.title} {currentItem.company ? `at ${currentItem.company}` : ''}
                  </p>
                )}
                {currentItem?.type === 'event' && currentItem.date && (
                  <p className="text-gray-700 font-medium">
                    {new Date(currentItem.date).toLocaleDateString()}
                  </p>
                )}
                
                {/* Show location for user and event */}
                {currentItem?.location && (
                  <p className="text-gray-500 text-sm mb-3">📍 {currentItem.location}</p>
                )}
                
                {/* Show skills for user */}
                {currentItem?.type === 'user' && (currentItem.skills || []).length > 0 && (
                  <div className="mb-3">
                    <p className="text-sm font-medium text-gray-700 mb-1">Skills</p>
                    <div className="flex flex-wrap gap-1">
                      {(currentItem.skills || []).slice(0,5).map((skill, i) => (<Badge key={i}>{skill}</Badge>))}
                      {( (currentItem.skills || []).length > 5 ) && (
                        <Badge>+{(currentItem.skills || []).length - 5} more</Badge>
                      )}
                    </div>
                  </div>
                )}
                
                <Button 
                  variant="outline" 
                  className="w-full mt-2"
                  onClick={() => router.push('/dashboard/network/swipe')}
                >
                  View Details
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