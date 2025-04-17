'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { 
  Users, 
  Calendar, 
  Map, 
  Tag, 
  Share2, 
  MessageSquare, 
  UserPlus, 
  ChevronRight,
  Check 
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Skeleton } from '@/components/ui/skeleton';

interface GroupData {
  id: string;
  name: string;
  description: string;
  category: string;
  location: string;
  image_url: string;
  created_at: string;
  organizer_id: string;
  member_count: number;
  tags: string[];
  is_private: boolean;
  organizer?: {
    id: string;
    full_name: string;
    avatar_url: string;
  };
}

interface Member {
  id: string;
  full_name: string;
  avatar_url: string | null;
  title: string | null;
  joined_at: string;
  is_organizer: boolean;
}

export default function GroupDetailPage() {
  const router = useRouter();
  const params = useParams();
  const groupId = params?.id as string;
  
  const [group, setGroup] = useState<GroupData | null>(null);
  const [members, setMembers] = useState<Member[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isMember, setIsMember] = useState(false);
  const [isJoining, setIsJoining] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('about');
  
  const supabase = createClientComponentClient();
  
  useEffect(() => {
    const fetchUser = async () => {
      const { data } = await supabase.auth.getUser();
      setUser(data.user);
    };
    
    fetchUser();
  }, [supabase]);
  
  useEffect(() => {
    const fetchGroupData = async () => {
      if (!groupId) return;
      
      try {
        setIsLoading(true);
        
        // Fetch group details
        const { data: groupData, error: groupError } = await supabase
          .from('groups')
          .select(`
            *,
            organizer:organizer_id(id, full_name, avatar_url)
          `)
          .eq('id', groupId)
          .single();
        
        if (groupError) throw groupError;
        
        // Fetch members
        const { data: membersData, error: membersError } = await supabase
          .from('group_members')
          .select(`
            user_id,
            joined_at,
            profiles:user_id(id, full_name, avatar_url, title)
          `)
          .eq('group_id', groupId)
          .limit(10);
        
        if (membersError) throw membersError;
        
        // Check if current user is a member
        if (user?.id) {
          const { data: membershipData } = await supabase
            .from('group_members')
            .select('*')
            .eq('group_id', groupId)
            .eq('user_id', user.id)
            .maybeSingle();
          
          setIsMember(!!membershipData);
        }
        
        // Format members data
        const formattedMembers = membersData
          .filter(m => m.profiles)
          .map(m => ({
            id: m.profiles.id,
            full_name: m.profiles.full_name,
            avatar_url: m.profiles.avatar_url,
            title: m.profiles.title,
            joined_at: m.joined_at,
            is_organizer: m.profiles.id === groupData.organizer_id
          }));
        
        setGroup(groupData);
        setMembers(formattedMembers);
      } catch (error) {
        console.error('Error fetching group data:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchGroupData();
  }, [groupId, supabase, user?.id]);
  
  const handleJoinGroup = async () => {
    if (!user?.id) {
      router.push('/login?redirect=' + encodeURIComponent(`/groups/${groupId}`));
      return;
    }
    
    try {
      setIsJoining(true);
      
      const { error } = await supabase
        .from('group_members')
        .insert({
          group_id: groupId,
          user_id: user.id
        });
      
      if (error) throw error;
      
      setIsMember(true);
      
      // Optionally redirect to alignment page
      router.push(`/groups/${groupId}/members/alignment`);
    } catch (error) {
      console.error('Error joining group:', error);
    } finally {
      setIsJoining(false);
    }
  };
  
  const handleShareGroup = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: group?.name || 'Join my group on Networkli',
          text: `Check out ${group?.name} on Networkli: ${group?.description?.substring(0, 100)}...`,
          url: window.location.href
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      // Fallback for browsers that don't support navigator.share
      navigator.clipboard.writeText(window.location.href);
      // Show some feedback (would implement a toast notification in a real app)
      alert('Group link copied to clipboard!');
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8">
        <Skeleton className="h-64 w-full rounded-lg mb-6" />
        <Skeleton className="h-10 w-1/3 rounded-md mb-4" />
        <Skeleton className="h-4 w-full rounded-md mb-2" />
        <Skeleton className="h-4 w-full rounded-md mb-2" />
        <Skeleton className="h-4 w-2/3 rounded-md mb-6" />
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <Skeleton className="h-10 w-full rounded-md mb-4" />
            <Skeleton className="h-32 w-full rounded-md" />
          </div>
          <div>
            <Skeleton className="h-10 w-full rounded-md mb-4" />
            <Skeleton className="h-64 w-full rounded-md" />
          </div>
        </div>
      </div>
    );
  }
  
  if (!group) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8 text-center">
        <h1 className="text-2xl font-bold mb-4">Group not found</h1>
        <p className="mb-6">The group you're looking for doesn't exist or has been removed.</p>
        <Button asChild>
          <Link href="/groups">Browse Groups</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Group Header */}
      <div className="relative mb-8">
        <div className="aspect-[3/1] w-full rounded-lg overflow-hidden bg-muted mb-6">
          {group.image_url ? (
            <Image
              src={group.image_url}
              alt={group.name}
              fill
              className="object-cover"
              priority
            />
          ) : (
            <div className="w-full h-full bg-gradient-to-r from-blue-100 to-indigo-100 flex items-center justify-center">
              <Users className="h-24 w-24 text-blue-300" />
            </div>
          )}
        </div>
        
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">{group.name}</h1>
            <div className="flex flex-wrap items-center gap-3 mt-2">
              <Badge variant="outline">{group.category}</Badge>
              {group.location && (
                <div className="flex items-center text-sm text-muted-foreground">
                  <Map className="h-4 w-4 mr-1" />
                  {group.location}
                </div>
              )}
              <div className="flex items-center text-sm text-muted-foreground">
                <Users className="h-4 w-4 mr-1" />
                {group.member_count || members.length} members
              </div>
            </div>
            
            {group.tags && group.tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {group.tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}
          </div>
          
          <div className="flex flex-wrap gap-3">
            {isMember ? (
              <Button variant="outline" onClick={handleShareGroup}>
                <Share2 className="h-4 w-4 mr-2" />
                Share
              </Button>
            ) : (
              <>
                <Button onClick={handleJoinGroup} disabled={isJoining}>
                  {isJoining ? (
                    <>Joining...</>
                  ) : (
                    <>
                      <UserPlus className="h-4 w-4 mr-2" />
                      Join Group
                    </>
                  )}
                </Button>
                <Button variant="outline" onClick={handleShareGroup}>
                  <Share2 className="h-4 w-4 mr-2" />
                  Share
                </Button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Group Content */}
        <div className="lg:col-span-2">
          <Tabs defaultValue="about" className="mb-8" onValueChange={setActiveTab}>
            <TabsList className="mb-4">
              <TabsTrigger value="about">About</TabsTrigger>
              <TabsTrigger value="discussions">Discussions</TabsTrigger>
              <TabsTrigger value="events">Events</TabsTrigger>
            </TabsList>
            
            <TabsContent value="about" className="space-y-6">
              <Card>
                <CardContent className="pt-6">
                  <h2 className="text-xl font-semibold mb-4">About this group</h2>
                  <div className="prose max-w-none">
                    <p>{group.description}</p>
                  </div>
                </CardContent>
              </Card>
              
              {isMember && (
                <Card>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-xl font-semibold">Meet similar members</h2>
                      <Button variant="ghost" size="sm" className="text-primary" asChild>
                        <Link href={`/groups/${groupId}/members/alignment`}>
                          View all
                          <ChevronRight className="h-4 w-4 ml-1" />
                        </Link>
                      </Button>
                    </div>
                    <p className="text-muted-foreground mb-4">
                      Connect with members that match your professional background and interests.
                    </p>
                    
                    {/* Placeholder for aligned members preview */}
                    <div className="bg-muted/30 rounded-lg p-6 text-center">
                      <Users className="h-10 w-10 mx-auto mb-2 text-muted-foreground" />
                      <h3 className="font-medium">Discover aligned members</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        Find members with similar skills and interests
                      </p>
                      <Button asChild>
                        <Link href={`/groups/${groupId}/members/alignment`}>
                          View Aligned Members
                        </Link>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
            
            <TabsContent value="discussions">
              <Card>
                <CardContent className="pt-6">
                  {isMember ? (
                    <>
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-semibold">Discussions</h2>
                        <Button size="sm">
                          <MessageSquare className="h-4 w-4 mr-2" />
                          New Post
                        </Button>
                      </div>
                      
                      <div className="bg-muted/30 rounded-lg p-6 text-center">
                        <MessageSquare className="h-10 w-10 mx-auto mb-2 text-muted-foreground" />
                        <h3 className="font-medium">No discussions yet</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                          Be the first to start a conversation in this group
                        </p>
                        <Button>Start a Discussion</Button>
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-8">
                      <MessageSquare className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                      <h3 className="text-lg font-medium mb-2">Join to view discussions</h3>
                      <p className="text-muted-foreground mb-4">
                        Become a member to participate in group discussions
                      </p>
                      <Button onClick={handleJoinGroup} disabled={isJoining}>
                        {isJoining ? 'Joining...' : 'Join Group'}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="events">
              <Card>
                <CardContent className="pt-6">
                  <h2 className="text-xl font-semibold mb-4">Upcoming Events</h2>
                  
                  <div className="bg-muted/30 rounded-lg p-6 text-center">
                    <Calendar className="h-10 w-10 mx-auto mb-2 text-muted-foreground" />
                    <h3 className="font-medium">No upcoming events</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Stay tuned for future events from this group
                    </p>
                    {group.organizer?.id === user?.id && (
                      <Button>Create Event</Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        
        {/* Sidebar */}
        <div className="space-y-6">
          {/* Organizer */}
          <Card>
            <CardContent className="pt-6">
              <h3 className="font-medium mb-4">Organizer</h3>
              {group.organizer && (
                <div className="flex items-center gap-3">
                  <Avatar className="h-10 w-10 border">
                    {group.organizer.avatar_url ? (
                      <Image
                        src={group.organizer.avatar_url}
                        alt={group.organizer.full_name}
                        fill
                        className="object-cover"
                      />
                    ) : (
                      <div className="h-full w-full bg-muted flex items-center justify-center">
                        <span className="text-sm font-medium">
                          {group.organizer.full_name.charAt(0)}
                        </span>
                      </div>
                    )}
                  </Avatar>
                  <div>
                    <Link
                      href={`/profile/${group.organizer.id}`}
                      className="font-medium hover:underline"
                    >
                      {group.organizer.full_name}
                    </Link>
                    <p className="text-xs text-muted-foreground">Organizer</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Members */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium">Members</h3>
                <span className="text-sm text-muted-foreground">
                  {group.member_count || members.length} members
                </span>
              </div>
              
              <div className="space-y-3">
                {members.slice(0, 5).map((member) => (
                  <div key={member.id} className="flex items-center gap-3">
                    <Avatar className="h-8 w-8 border">
                      {member.avatar_url ? (
                        <Image
                          src={member.avatar_url}
                          alt={member.full_name}
                          fill
                          className="object-cover"
                        />
                      ) : (
                        <div className="h-full w-full bg-muted flex items-center justify-center">
                          <span className="text-xs font-medium">
                            {member.full_name.charAt(0)}
                          </span>
                        </div>
                      )}
                    </Avatar>
                    <div className="min-w-0">
                      <Link
                        href={`/profile/${member.id}`}
                        className="text-sm font-medium hover:underline truncate block"
                      >
                        {member.full_name}
                      </Link>
                      {member.title && (
                        <p className="text-xs text-muted-foreground truncate">
                          {member.title}
                        </p>
                      )}
                    </div>
                    {member.is_organizer && (
                      <Badge variant="outline" className="ml-auto text-xs">
                        Organizer
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
              
              {members.length > 5 && (
                <Button variant="ghost" size="sm" className="w-full mt-4" asChild>
                  <Link href={`/groups/${groupId}/members`}>
                    View all members
                  </Link>
                </Button>
              )}
              
              {members.length === 0 && (
                <div className="text-center py-4">
                  <p className="text-sm text-muted-foreground">No members yet</p>
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Related Groups */}
          <Card>
            <CardContent className="pt-6">
              <h3 className="font-medium mb-4">Similar Groups</h3>
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <p className="text-sm text-muted-foreground">
                  Explore more groups like this one
                </p>
                <Button variant="outline" size="sm" className="mt-2" asChild>
                  <Link href="/groups">Browse Groups</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 