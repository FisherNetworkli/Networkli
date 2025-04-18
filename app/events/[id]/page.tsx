'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { 
  Calendar, 
  Clock, 
  MapPin, 
  Users, 
  Share2, 
  Tag, 
  ExternalLink, 
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

interface EventData {
  id: string;
  title: string;
  description: string;
  start_time: string;
  end_time: string;
  location: string;
  address: string;
  is_virtual: boolean;
  meeting_link?: string;
  image_url?: string;
  category: string;
  format: string;
  created_at: string;
  organizer_id: string;
  group_id?: string;
  max_attendees?: number;
  attendee_count: number;
  tags: string[];
  is_private: boolean;
  organizer?: {
    id: string;
    full_name: string;
    avatar_url: string;
  };
  group?: {
    id: string;
    name: string;
  };
}

interface Attendee {
  id: string;
  full_name: string;
  avatar_url: string | null;
  title: string | null;
  rsvp_status: string;
  is_organizer: boolean;
}

export default function EventDetailPage() {
  const router = useRouter();
  const params = useParams();
  const eventId = params?.id as string;
  
  const [event, setEvent] = useState<EventData | null>(null);
  const [attendees, setAttendees] = useState<Attendee[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isAttending, setIsAttending] = useState(false);
  const [isRsvping, setIsRsvping] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [userAttendeeData, setUserAttendeeData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('details');
  
  const supabase = createClientComponentClient();
  
  useEffect(() => {
    const fetchUser = async () => {
      const { data } = await supabase.auth.getUser();
      setUser(data.user);
    };
    
    fetchUser();
  }, [supabase]);
  
  useEffect(() => {
    const fetchEventData = async () => {
      if (!eventId) return;
      
      try {
        setIsLoading(true);
        
        // Fetch event details
        const { data: eventData, error: eventError } = await supabase
          .from('events')
          .select(`
            *,
            organizer:organizer_id(id, full_name, avatar_url),
            group:group_id(id, name)
          `)
          .eq('id', eventId)
          .single();
        
        if (eventError) throw eventError;
        
        // Fetch attendees
        const { data: attendeesData, error: attendeesError } = await supabase
          .from('event_attendance')
          .select(`
            user_id,
            status,
            profiles:user_id(
              id,
              first_name,
              last_name,
              avatar_url,
              title
            )
          `)
          .eq('event_id', eventId)
          .eq('status', 'attending')
          .limit(10);
        
        if (attendeesError) throw attendeesError;
        
        // Check if current user is attending
        if (user?.id) {
          const { data: attendanceData } = await supabase
            .from('event_attendance')
            .select('*')
            .eq('event_id', eventId)
            .eq('user_id', user.id)
            .maybeSingle();
          
          setIsAttending(attendanceData?.status === 'attending');
          setUserAttendeeData(attendanceData);
        }
        
        // Format attendees data
        const formattedAttendees = attendeesData
          ?.filter(a => Array.isArray(a.profiles) && a.profiles.length > 0)
          .map(a => {
            const profile = a.profiles[0];
            return {
              id: profile.id,
              full_name: `${profile.first_name || ''} ${profile.last_name || ''}`.trim(),
              avatar_url: profile.avatar_url,
              title: profile.title,
              rsvp_status: a.status,
              is_organizer: profile.id === eventData?.organizer_id
            };
          }) || [];
        
        setEvent(eventData);
        setAttendees(formattedAttendees);
      } catch (error) {
        console.error('Error fetching event data:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchEventData();
  }, [eventId, supabase, user?.id]);
  
  const handleRsvp = async (status: 'attending' | 'interested' | 'not_attending') => {
    if (!user?.id) {
      router.push('/login?redirect=' + encodeURIComponent(`/events/${eventId}`));
      return;
    }
    
    try {
      setIsRsvping(true);
      
      if (userAttendeeData) {
        // Update existing RSVP
        const { error } = await supabase
          .from('event_attendance')
          .update({ status })
          .eq('id', userAttendeeData.id);
        
        if (error) throw error;
      } else {
        // Create new RSVP
        const { error } = await supabase
          .from('event_attendance')
          .insert({
            event_id: eventId,
            user_id: user.id,
            status
          });
        
        if (error) throw error;
      }
      
      setIsAttending(status === 'attending');
      
      // If user is attending, optionally redirect to the alignment page
      if (status === 'attending') {
        router.push(`/events/${eventId}/alignment`);
      }
    } catch (error) {
      console.error('Error updating RSVP:', error);
    } finally {
      setIsRsvping(false);
    }
  };
  
  const handleShareEvent = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: event?.title || 'Join me at this event on Networkli',
          text: `Join me at ${event?.title} on ${new Date(event?.start_time || '').toLocaleDateString()}`,
          url: window.location.href
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      // Fallback for browsers that don't support navigator.share
      navigator.clipboard.writeText(window.location.href);
      // Show some feedback (would implement a toast notification in a real app)
      alert('Event link copied to clipboard!');
    }
  };
  
  // Format date and time
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
  
  const formatEventTime = (start: string, end?: string) => {
    const startDate = new Date(start);
    const formattedStart = startDate.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
    
    if (!end) return formattedStart;
    
    const endDate = new Date(end);
    const formattedEnd = endDate.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
    
    return `${formattedStart} - ${formattedEnd}`;
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
  
  if (!event) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8 text-center">
        <h1 className="text-2xl font-bold mb-4">Event not found</h1>
        <p className="mb-6">The event you're looking for doesn't exist or has been removed.</p>
        <Button asChild>
          <Link href="/events">Browse Events</Link>
        </Button>
      </div>
    );
  }

  // Check if event is past
  const isPastEvent = new Date(event.end_time || event.start_time) < new Date();

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Event Header */}
      <div className="relative mb-8">
        <div className="aspect-[3/1] w-full rounded-lg overflow-hidden bg-muted mb-6">
          {event.image_url ? (
            <Image
              src={event.image_url}
              alt={event.title}
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
        
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">{event.title}</h1>
            
            <div className="flex flex-wrap items-center gap-3 mt-2">
              <Badge variant="outline" className="capitalize">{event.format || 'In Person'}</Badge>
              <Badge variant="outline">{event.category}</Badge>
              
              {event.group && (
                <Link href={`/groups/${event.group.id}`} className="text-sm hover:underline flex items-center text-primary">
                  Hosted by {event.group.name}
                </Link>
              )}
            </div>
            
            <div className="flex flex-col gap-2 mt-4">
              <div className="flex items-center text-sm">
                <Calendar className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{formatEventDate(event.start_time, event.end_time)}</span>
              </div>
              
              <div className="flex items-center text-sm">
                <Clock className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{formatEventTime(event.start_time, event.end_time)}</span>
              </div>
              
              {event.location && (
                <div className="flex items-center text-sm">
                  <MapPin className="h-4 w-4 mr-2 text-muted-foreground" />
                  <span>{event.is_virtual ? 'Virtual Event' : event.location}</span>
                </div>
              )}
              
              <div className="flex items-center text-sm">
                <Users className="h-4 w-4 mr-2 text-muted-foreground" />
                <span>{event.attendee_count || attendees.length} attending</span>
                {event.max_attendees && (
                  <span className="ml-1 text-muted-foreground">
                    (Limit: {event.max_attendees})
                  </span>
                )}
              </div>
            </div>
            
            {event.tags && event.tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {event.tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}
          </div>
          
          <div className="flex flex-wrap gap-3">
            {isPastEvent ? (
              <Badge variant="outline" className="text-muted-foreground px-3 py-1">
                Past Event
              </Badge>
            ) : isAttending ? (
              <>
                <Button variant="outline" onClick={handleShareEvent}>
                  <Share2 className="h-4 w-4 mr-2" />
                  Share
                </Button>
                <Button variant="outline" className="bg-green-50">
                  <Check className="h-4 w-4 mr-2 text-green-600" />
                  <span className="text-green-600">Attending</span>
                </Button>
              </>
            ) : (
              <>
                <Button 
                  onClick={() => handleRsvp('attending')} 
                  disabled={isRsvping}
                >
                  {isRsvping ? (
                    <>Processing...</>
                  ) : (
                    <>
                      <UserPlus className="h-4 w-4 mr-2" />
                      Attend
                    </>
                  )}
                </Button>
                <Button variant="outline" onClick={handleShareEvent}>
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
        {/* Event Content */}
        <div className="lg:col-span-2">
          <Tabs defaultValue="details" className="mb-8" onValueChange={setActiveTab}>
            <TabsList className="mb-4">
              <TabsTrigger value="details">Details</TabsTrigger>
              <TabsTrigger value="attendees">Attendees</TabsTrigger>
              <TabsTrigger value="discussions">Discussions</TabsTrigger>
            </TabsList>
            
            <TabsContent value="details" className="space-y-6">
              <Card>
                <CardContent className="pt-6">
                  <h2 className="text-xl font-semibold mb-4">About this event</h2>
                  <div className="prose max-w-none">
                    <p>{event.description}</p>
                  </div>
                  
                  {event.is_virtual && event.meeting_link && (
                    <div className="mt-6 p-4 border rounded-lg">
                      <h3 className="font-medium mb-2">Virtual Event Link</h3>
                      <div className="flex flex-wrap items-center gap-2">
                        <a 
                          href={event.meeting_link} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-primary hover:underline flex items-center"
                        >
                          {event.meeting_link}
                          <ExternalLink className="h-4 w-4 ml-1" />
                        </a>
                      </div>
                      <p className="text-sm text-muted-foreground mt-2">
                        The meeting link will be available at the time of the event
                      </p>
                    </div>
                  )}
                  
                  {!event.is_virtual && event.address && (
                    <div className="mt-6 p-4 border rounded-lg">
                      <h3 className="font-medium mb-2">Event Location</h3>
                      <p>{event.address}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
              
              {isAttending && !isPastEvent && (
                <Card>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-xl font-semibold">Connect with attendees</h2>
                      <Button variant="ghost" size="sm" className="text-primary" asChild>
                        <Link href={`/events/${eventId}/alignment`}>
                          View all
                          <ChevronRight className="h-4 w-4 ml-1" />
                        </Link>
                      </Button>
                    </div>
                    <p className="text-muted-foreground mb-4">
                      Meet other attendees who match your professional background and interests.
                    </p>
                    
                    {/* Placeholder for aligned members preview */}
                    <div className="bg-muted/30 rounded-lg p-6 text-center">
                      <Users className="h-10 w-10 mx-auto mb-2 text-muted-foreground" />
                      <h3 className="font-medium">Discover aligned attendees</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        Find attendees with similar skills and interests
                      </p>
                      <Button asChild>
                        <Link href={`/events/${eventId}/alignment`}>
                          View Aligned Attendees
                        </Link>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
            
            <TabsContent value="attendees">
              <Card>
                <CardContent className="pt-6">
                  <h2 className="text-xl font-semibold mb-6">People attending</h2>
                  
                  {attendees.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {attendees.map((attendee) => (
                        <div key={attendee.id} className="flex items-center gap-3 p-3 rounded-lg hover:bg-muted/50">
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
                          <div className="min-w-0">
                            <Link
                              href={`/profile/${attendee.id}`}
                              className="font-medium hover:underline truncate block"
                            >
                              {attendee.full_name}
                            </Link>
                            {attendee.title && (
                              <p className="text-xs text-muted-foreground truncate">
                                {attendee.title}
                              </p>
                            )}
                          </div>
                          {attendee.is_organizer && (
                            <Badge variant="outline" className="ml-auto text-xs">
                              Organizer
                            </Badge>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Users className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                      <h3 className="text-lg font-medium mb-2">No attendees yet</h3>
                      <p className="text-muted-foreground mb-4">
                        Be the first to attend this event
                      </p>
                      {!isAttending && !isPastEvent && (
                        <Button onClick={() => handleRsvp('attending')} disabled={isRsvping}>
                          {isRsvping ? 'Processing...' : 'Attend Event'}
                        </Button>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="discussions">
              <Card>
                <CardContent className="pt-6">
                  <h2 className="text-xl font-semibold mb-4">Event Discussion</h2>
                  
                  <div className="bg-muted/30 rounded-lg p-6 text-center">
                    <div className="text-center py-4">
                      <p className="text-sm text-muted-foreground">No discussions yet</p>
                    </div>
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
              {event.organizer && (
                <div className="flex items-center gap-3">
                  <Avatar className="h-10 w-10 border">
                    {event.organizer.avatar_url ? (
                      <Image
                        src={event.organizer.avatar_url}
                        alt={event.organizer.full_name}
                        fill
                        className="object-cover"
                      />
                    ) : (
                      <div className="h-full w-full bg-muted flex items-center justify-center">
                        <span className="text-sm font-medium">
                          {event.organizer.full_name.charAt(0)}
                        </span>
                      </div>
                    )}
                  </Avatar>
                  <div>
                    <Link
                      href={`/profile/${event.organizer.id}`}
                      className="font-medium hover:underline"
                    >
                      {event.organizer.full_name}
                    </Link>
                    <p className="text-xs text-muted-foreground">Organizer</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* RSVP */}
          {!isPastEvent && (
            <Card>
              <CardContent className="pt-6">
                <h3 className="font-medium mb-4">Your RSVP</h3>
                
                {isAttending ? (
                  <>
                    <div className="bg-green-50 text-green-700 px-4 py-3 rounded-md flex items-center mb-4">
                      <Check className="h-5 w-5 mr-2" />
                      <span>You're attending this event</span>
                    </div>
                    <Button 
                      variant="outline" 
                      className="w-full" 
                      onClick={() => handleRsvp('not_attending')}
                      disabled={isRsvping}
                    >
                      {isRsvping ? 'Updating...' : 'Cancel RSVP'}
                    </Button>
                  </>
                ) : (
                  <Button 
                    className="w-full" 
                    onClick={() => handleRsvp('attending')}
                    disabled={isRsvping}
                  >
                    {isRsvping ? 'Processing...' : 'Attend This Event'}
                  </Button>
                )}
              </CardContent>
            </Card>
          )}
          
          {/* Attendees Preview */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium">Attendees</h3>
                <span className="text-sm text-muted-foreground">
                  {event.attendee_count || attendees.length} attending
                </span>
              </div>
              
              {attendees.length > 0 ? (
                <>
                  <div className="flex -space-x-2 overflow-hidden mb-4">
                    {attendees.slice(0, 5).map((attendee) => (
                      <Avatar key={attendee.id} className="h-8 w-8 border border-background">
                        {attendee.avatar_url ? (
                          <Image
                            src={attendee.avatar_url}
                            alt={attendee.full_name}
                            fill
                            className="object-cover"
                          />
                        ) : (
                          <div className="h-full w-full bg-muted flex items-center justify-center">
                            <span className="text-xs font-medium">
                              {attendee.full_name.charAt(0)}
                            </span>
                          </div>
                        )}
                      </Avatar>
                    ))}
                    {attendees.length > 5 && (
                      <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center border border-background">
                        <span className="text-xs">+{attendees.length - 5}</span>
                      </div>
                    )}
                  </div>
                  
                  <Button variant="ghost" size="sm" className="w-full" asChild>
                    <Link href={`/events/${eventId}?tab=attendees`}>
                      View all attendees
                    </Link>
                  </Button>
                </>
              ) : (
                <div className="text-center py-4">
                  <p className="text-sm text-muted-foreground">No attendees yet</p>
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Related Events */}
          <Card>
            <CardContent className="pt-6">
              <h3 className="font-medium mb-4">Similar Events</h3>
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <p className="text-sm text-muted-foreground">
                  Explore more events like this one
                </p>
                <Button variant="outline" size="sm" className="mt-2" asChild>
                  <Link href="/events">Browse Events</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 