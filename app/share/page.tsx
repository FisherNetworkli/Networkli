'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { 
  ArrowRight, 
  Calendar, 
  Users, 
  MapPin, 
  Clock, 
  Share2, 
  UserPlus 
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';

export default function ShareLandingPage() {
  const searchParams = useSearchParams();
  const type = searchParams.get('type') || 'group'; // 'group' or 'event'
  const id = searchParams.get('id');
  
  const [data, setData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);
  const [isJoining, setIsJoining] = useState(false);
  
  const supabase = createClientComponentClient();
  
  useEffect(() => {
    const fetchUser = async () => {
      const { data } = await supabase.auth.getUser();
      setUser(data.user);
    };
    
    fetchUser();
  }, [supabase]);
  
  useEffect(() => {
    const fetchData = async () => {
      if (!id) {
        setError('No ID provided');
        setIsLoading(false);
        return;
      }
      
      try {
        setIsLoading(true);
        setError(null);
        
        if (type === 'group') {
          // Fetch group details
          const { data: groupData, error: groupError } = await supabase
            .from('groups')
            .select(`
              *,
              organizer:organizer_id(id, full_name, avatar_url),
              member_count:group_members(count)
            `)
            .eq('id', id)
            .single();
          
          if (groupError) throw groupError;
          
          // Fetch recent members
          const { data: membersData, error: membersError } = await supabase
            .from('group_members')
            .select(`
              user_id,
              profiles:user_id(id, full_name, avatar_url)
            `)
            .eq('group_id', id)
            .order('joined_at', { ascending: false })
            .limit(5);
          
          if (membersError) throw membersError;
          
          // Check if current user is a member
          let isMember = false;
          if (user?.id) {
            const { data: memberData } = await supabase
              .from('group_members')
              .select('*')
              .eq('group_id', id)
              .eq('user_id', user.id)
              .maybeSingle();
            
            isMember = !!memberData;
          }
          
          setData({
            ...groupData,
            members: membersData.map(m => m.profiles),
            isMember,
            type: 'group'
          });
        } else if (type === 'event') {
          // Fetch event details
          const { data: eventData, error: eventError } = await supabase
            .from('events')
            .select(`
              *,
              organizer:organizer_id(id, full_name, avatar_url),
              group:group_id(id, name),
              attendee_count:event_attendance(count)
            `)
            .eq('id', id)
            .single();
          
          if (eventError) throw eventError;
          
          // Fetch recent attendees
          const { data: attendeesData, error: attendeesError } = await supabase
            .from('event_attendance')
            .select(`
              user_id,
              profiles:user_id(id, full_name, avatar_url)
            `)
            .eq('event_id', id)
            .eq('status', 'attending')
            .order('created_at', { ascending: false })
            .limit(5);
          
          if (attendeesError) throw attendeesError;
          
          // Check if current user is attending
          let isAttending = false;
          if (user?.id) {
            const { data: attendeeData } = await supabase
              .from('event_attendance')
              .select('*')
              .eq('event_id', id)
              .eq('user_id', user.id)
              .eq('status', 'attending')
              .maybeSingle();
            
            isAttending = !!attendeeData;
          }
          
          // Check if event is past
          const isPastEvent = new Date(eventData.end_time || eventData.start_time) < new Date();
          
          setData({
            ...eventData,
            attendees: attendeesData.map(a => a.profiles),
            isAttending,
            isPastEvent,
            type: 'event'
          });
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to load details');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [id, type, supabase, user?.id]);
  
  const handleJoinGroup = async () => {
    if (!user?.id) {
      window.location.href = `/login?redirect=${encodeURIComponent(window.location.href)}`;
      return;
    }
    
    try {
      setIsJoining(true);
      
      const { error } = await supabase
        .from('group_members')
        .insert({
          group_id: id,
          user_id: user.id
        });
      
      if (error) throw error;
      
      // Update local state
      setData(prevData => ({
        ...prevData,
        isMember: true
      }));
      
      // Redirect to group page
      window.location.href = `/groups/${id}`;
    } catch (error) {
      console.error('Error joining group:', error);
    } finally {
      setIsJoining(false);
    }
  };
  
  const handleAttendEvent = async () => {
    if (!user?.id) {
      window.location.href = `/login?redirect=${encodeURIComponent(window.location.href)}`;
      return;
    }
    
    try {
      setIsJoining(true);
      
      const { error } = await supabase
        .from('event_attendance')
        .insert({
          event_id: id,
          user_id: user.id,
          status: 'attending'
        });
      
      if (error) throw error;
      
      // Update local state
      setData(prevData => ({
        ...prevData,
        isAttending: true
      }));
      
      // Redirect to event page
      window.location.href = `/events/${id}`;
    } catch (error) {
      console.error('Error attending event:', error);
    } finally {
      setIsJoining(false);
    }
  };
  
  const formatEventDate = (start: string, end?: string) => {
    const startDate = new Date(start);
    const formattedStart = startDate.toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
      year: 'numeric'
    });
    
    if (!end) return formattedStart;
    
    const endDate = new Date(end);
    
    // If same day event
    if (startDate.toDateString() === endDate.toDateString()) {
      return formattedStart;
    }
    
    // Multi-day event
    const formattedEnd = endDate.toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric'
    });
    
    return `${formattedStart} - ${formattedEnd}`;
  };

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        <Skeleton className="h-64 w-full rounded-lg mb-6" />
        <Skeleton className="h-10 w-1/3 rounded-md mb-4" />
        <Skeleton className="h-4 w-full rounded-md mb-2" />
        <Skeleton className="h-4 w-full rounded-md mb-2" />
        <Skeleton className="h-4 w-2/3 rounded-md mb-6" />
        <Skeleton className="h-12 w-40 rounded-md" />
      </div>
    );
  }
  
  if (error || !data) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <div className="bg-red-50 text-red-800 p-4 rounded-md mb-6">
          {error || 'Unable to load content'}
        </div>
        <p className="mb-6">
          Please check the link and try again, or browse our available content.
        </p>
        <div className="flex flex-wrap gap-4 justify-center">
          <Button asChild>
            <Link href="/groups">Browse Groups</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/events">Browse Events</Link>
          </Button>
        </div>
      </div>
    );
  }

  if (data.type === 'group') {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Group Header */}
        <div className="relative mb-8">
          <div className="aspect-video w-full rounded-lg overflow-hidden bg-muted mb-6">
            {data.image_url ? (
              <Image
                src={data.image_url}
                alt={data.name}
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
          
          <div>
            <h1 className="text-4xl font-bold mb-2">{data.name}</h1>
            
            <div className="flex flex-wrap items-center gap-3 mt-2 mb-4">
              <Badge variant="outline">{data.category}</Badge>
              {data.location && (
                <div className="flex items-center text-sm text-muted-foreground">
                  <MapPin className="h-4 w-4 mr-1" />
                  {data.location}
                </div>
              )}
              <div className="flex items-center text-sm text-muted-foreground">
                <Users className="h-4 w-4 mr-1" />
                {data.member_count?.[0]?.count || 0} members
              </div>
            </div>
            
            <div className="prose max-w-none mb-8">
              <p>{data.description}</p>
            </div>
            
            {data.isMember ? (
              <div className="flex gap-4">
                <Button asChild size="lg">
                  <Link href={`/groups/${id}`}>
                    Go to Group
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" onClick={() => {
                  if (navigator.share) {
                    navigator.share({
                      title: data.name,
                      text: `Join our group: ${data.name}`,
                      url: window.location.href
                    });
                  } else {
                    navigator.clipboard.writeText(window.location.href);
                    alert('Link copied to clipboard!');
                  }
                }}>
                  <Share2 className="mr-2 h-4 w-4" />
                  Share
                </Button>
              </div>
            ) : (
              <div className="flex gap-4">
                <Button 
                  size="lg" 
                  onClick={handleJoinGroup}
                  disabled={isJoining}
                >
                  {isJoining ? (
                    <>Joining...</>
                  ) : (
                    <>
                      <UserPlus className="mr-2 h-4 w-4" />
                      Join this Group
                    </>
                  )}
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link href="/groups">
                    Explore More Groups
                  </Link>
                </Button>
              </div>
            )}
          </div>
        </div>
        
        <Separator className="my-10" />
        
        {/* Organizer Info */}
        <div className="mb-10">
          <h2 className="text-xl font-semibold mb-4">About the Organizer</h2>
          
          {data.organizer && (
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16 border">
                {data.organizer.avatar_url ? (
                  <Image
                    src={data.organizer.avatar_url}
                    alt={data.organizer.full_name}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="h-full w-full bg-muted flex items-center justify-center">
                    <span className="text-lg font-medium">
                      {data.organizer.full_name.charAt(0)}
                    </span>
                  </div>
                )}
              </Avatar>
              <div>
                <h3 className="font-medium text-lg">{data.organizer.full_name}</h3>
                <p className="text-muted-foreground">Group Organizer</p>
                <Button variant="link" className="p-0 h-auto" asChild>
                  <Link href={`/profile/${data.organizer.id}`}>
                    View Profile
                  </Link>
                </Button>
              </div>
            </div>
          )}
        </div>
        
        {/* Members Preview */}
        {data.members && data.members.length > 0 && (
          <div className="mb-10">
            <h2 className="text-xl font-semibold mb-4">Members</h2>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              {data.members.map((member: any) => (
                <Card key={member.id}>
                  <CardContent className="p-4 flex items-center gap-3">
                    <Avatar className="h-10 w-10 border">
                      {member.avatar_url ? (
                        <Image
                          src={member.avatar_url}
                          alt={member.full_name}
                          fill
                          className="object-cover"
                        />
                      ) : (
                        <div className="h-full w-full bg-muted flex items-center justify-center">
                          <span className="text-sm font-medium">
                            {member.full_name.charAt(0)}
                          </span>
                        </div>
                      )}
                    </Avatar>
                    <div>
                      <Link
                        href={`/profile/${member.id}`}
                        className="font-medium hover:underline"
                      >
                        {member.full_name}
                      </Link>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
        
        {/* Join CTA */}
        {!data.isMember && (
          <Card className="bg-primary/5 border-primary/20">
            <CardContent className="p-6">
              <h2 className="text-2xl font-bold mb-2">Ready to join?</h2>
              <p className="text-muted-foreground mb-4">
                Connect with like-minded professionals in this group
              </p>
              <Button 
                size="lg" 
                onClick={handleJoinGroup}
                disabled={isJoining}
              >
                {isJoining ? 'Joining...' : 'Join this Group'}
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    );
  } else if (data.type === 'event') {
    // Check if event is past
    const isPastEvent = data.isPastEvent;
    
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Event Header */}
        <div className="relative mb-8">
          <div className="aspect-video w-full rounded-lg overflow-hidden bg-muted mb-6">
            {data.image_url ? (
              <Image
                src={data.image_url}
                alt={data.title}
                fill
                className="object-cover"
                priority
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-r from-blue-100 to-indigo-100 flex items-center justify-center">
                <Calendar className="h-24 w-24 text-blue-300" />
              </div>
            )}
          </div>
          
          <div>
            <h1 className="text-4xl font-bold mb-2">{data.title}</h1>
            
            <div className="flex flex-wrap items-center gap-3 mt-2 mb-4">
              <Badge variant="outline" className="capitalize">{data.format || 'In Person'}</Badge>
              <Badge variant="outline">{data.category}</Badge>
              
              {data.group && (
                <Link href={`/groups/${data.group.id}`} className="text-sm hover:underline flex items-center text-primary">
                  Hosted by {data.group.name}
                </Link>
              )}
            </div>
            
            <div className="space-y-3 mb-6">
              <div className="flex items-center text-sm">
                <Calendar className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{formatEventDate(data.start_time, data.end_time)}</span>
              </div>
              
              <div className="flex items-center text-sm">
                <Clock className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>
                  {new Date(data.start_time).toLocaleTimeString('en-US', {
                    hour: 'numeric',
                    minute: '2-digit',
                    hour12: true
                  })}
                  {data.end_time && ` - ${new Date(data.end_time).toLocaleTimeString('en-US', {
                    hour: 'numeric',
                    minute: '2-digit',
                    hour12: true
                  })}`}
                </span>
              </div>
              
              {data.location && (
                <div className="flex items-center text-sm">
                  <MapPin className="h-4 w-4 mr-2 text-muted-foreground" />
                  <span>{data.is_virtual ? 'Virtual Event' : data.location}</span>
                </div>
              )}
              
              <div className="flex items-center text-sm">
                <Users className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{data.attendee_count?.[0]?.count || 0} attending</span>
                {data.max_attendees && (
                  <span className="ml-1 text-muted-foreground">
                    (Limit: {data.max_attendees})
                  </span>
                )}
              </div>
            </div>
            
            <div className="prose max-w-none mb-8">
              <p>{data.description}</p>
            </div>
            
            {isPastEvent ? (
              <div className="bg-muted/80 p-4 rounded-md mb-6">
                <p className="font-medium">This event has already taken place.</p>
              </div>
            ) : data.isAttending ? (
              <div className="flex gap-4">
                <Button asChild size="lg">
                  <Link href={`/events/${id}`}>
                    View Event Details
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" onClick={() => {
                  if (navigator.share) {
                    navigator.share({
                      title: data.title,
                      text: `Join me at this event: ${data.title}`,
                      url: window.location.href
                    });
                  } else {
                    navigator.clipboard.writeText(window.location.href);
                    alert('Link copied to clipboard!');
                  }
                }}>
                  <Share2 className="mr-2 h-4 w-4" />
                  Share
                </Button>
              </div>
            ) : (
              <div className="flex gap-4">
                <Button 
                  size="lg" 
                  onClick={handleAttendEvent}
                  disabled={isJoining}
                >
                  {isJoining ? (
                    <>Processing...</>
                  ) : (
                    <>
                      <UserPlus className="mr-2 h-4 w-4" />
                      Attend This Event
                    </>
                  )}
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link href="/events">
                    Explore More Events
                  </Link>
                </Button>
              </div>
            )}
          </div>
        </div>
        
        <Separator className="my-10" />
        
        {/* Organizer Info */}
        <div className="mb-10">
          <h2 className="text-xl font-semibold mb-4">Organized by</h2>
          
          {data.organizer && (
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16 border">
                {data.organizer.avatar_url ? (
                  <Image
                    src={data.organizer.avatar_url}
                    alt={data.organizer.full_name}
                    fill
                    className="object-cover"
                  />
                ) : (
                  <div className="h-full w-full bg-muted flex items-center justify-center">
                    <span className="text-lg font-medium">
                      {data.organizer.full_name.charAt(0)}
                    </span>
                  </div>
                )}
              </Avatar>
              <div>
                <h3 className="font-medium text-lg">{data.organizer.full_name}</h3>
                <p className="text-muted-foreground">Event Organizer</p>
                <Button variant="link" className="p-0 h-auto" asChild>
                  <Link href={`/profile/${data.organizer.id}`}>
                    View Profile
                  </Link>
                </Button>
              </div>
            </div>
          )}
        </div>
        
        {/* Attendees Preview */}
        {data.attendees && data.attendees.length > 0 && (
          <div className="mb-10">
            <h2 className="text-xl font-semibold mb-4">People Attending</h2>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              {data.attendees.map((attendee: any) => (
                <Card key={attendee.id}>
                  <CardContent className="p-4 flex items-center gap-3">
                    <Avatar className="h-10 w-10 border">
                      {attendee.avatar_url ? (
                        <Image
                          src={attendee.avatar_url}
                          alt={attendee.full_name}
                          fill
                          className="object-cover"
                        />
                      ) : (
                        <div className="h-full w-full bg-muted flex items-center justify-center">
                          <span className="text-sm font-medium">
                            {attendee.full_name.charAt(0)}
                          </span>
                        </div>
                      )}
                    </Avatar>
                    <div>
                      <Link
                        href={`/profile/${attendee.id}`}
                        className="font-medium hover:underline"
                      >
                        {attendee.full_name}
                      </Link>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
        
        {/* Attend CTA */}
        {!isPastEvent && !data.isAttending && (
          <Card className="bg-primary/5 border-primary/20">
            <CardContent className="p-6">
              <h2 className="text-2xl font-bold mb-2">Ready to attend?</h2>
              <p className="text-muted-foreground mb-4">
                Join this event and connect with other attendees
              </p>
              <Button 
                size="lg" 
                onClick={handleAttendEvent}
                disabled={isJoining}
              >
                {isJoining ? 'Processing...' : 'Attend This Event'}
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    );
  }
  
  return null;
} 