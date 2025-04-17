'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { 
  Search, 
  Users, 
  Calendar, 
  MapPin, 
  Clock,
  Filter
} from 'lucide-react';
import { useSearchParams } from 'next/navigation';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';

interface Group {
  id: string;
  name: string;
  description: string;
  category: string;
  location?: string;
  member_count?: number;
  image_url?: string;
  created_at: string;
  is_private?: boolean;
}

interface Event {
  id: string;
  title: string;
  description: string;
  start_time: string;
  end_time?: string;
  location: string;
  format: string;
  category: string;
  organizer_id: string;
  organizer_name?: string;
  image_url?: string;
  attendee_count?: number;
}

export default function DiscoverPage() {
  const searchParams = useSearchParams();
  const tabParam = searchParams?.get('tab');
  const [activeTab, setActiveTab] = useState(tabParam === 'events' ? 'events' : 'groups');
  const [groups, setGroups] = useState<Group[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [loadingGroups, setLoadingGroups] = useState(true);
  const [loadingEvents, setLoadingEvents] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  
  const supabase = createClientComponentClient();
  
  // Fetch groups
  useEffect(() => {
    const fetchGroups = async () => {
      try {
        setLoadingGroups(true);
        
        // Fetch groups from the database
        const { data, error } = await supabase
          .from('groups')
          .select(`
            *,
            member_count:group_members(count)
          `)
          .order('created_at', { ascending: false })
          .limit(12);
        
        if (error) {
          console.error('Error fetching groups:', error);
          // Use placeholder data
          setGroups(getDemoGroups());
        } else if (!data || data.length === 0) {
          // Use placeholder data if no groups exist
          setGroups(getDemoGroups());
        } else {
          // Format the data
          const formattedGroups = data.map(group => ({
            ...group,
            member_count: group.member_count?.[0]?.count || 0
          }));
          setGroups(formattedGroups);
        }
      } catch (error) {
        console.error('Error:', error);
        setGroups(getDemoGroups());
      } finally {
        setLoadingGroups(false);
      }
    };
    
    fetchGroups();
  }, [supabase]);
  
  // Fetch events
  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setLoadingEvents(true);
        
        // Fetch events from the database
        const { data, error } = await supabase
          .from('events')
          .select(`
            *,
            attendee_count:event_attendance(count)
          `)
          .order('start_time', { ascending: true })
          .gte('start_time', new Date().toISOString())
          .limit(12);
        
        if (error) {
          console.error('Error fetching events:', error);
          // Use placeholder data
          setEvents(getDemoEvents());
        } else if (!data || data.length === 0) {
          // Use placeholder data if no events exist
          setEvents(getDemoEvents());
        } else {
          // Format the data
          const formattedEvents = data.map(event => ({
            ...event,
            attendee_count: event.attendee_count?.[0]?.count || 0
          }));
          setEvents(formattedEvents);
        }
      } catch (error) {
        console.error('Error:', error);
        setEvents(getDemoEvents());
      } finally {
        setLoadingEvents(false);
      }
    };
    
    fetchEvents();
  }, [supabase]);

  // Format date for display
  const formatDate = (dateString: string) => {
    const options: Intl.DateTimeFormatOptions = {
      month: 'short',
      day: 'numeric'
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };
  
  // Format time for display
  const formatTime = (dateString: string) => {
    const options: Intl.DateTimeFormatOptions = {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    };
    return new Date(dateString).toLocaleTimeString(undefined, options);
  };
  
  // Filter groups by search query
  const filteredGroups = searchQuery 
    ? groups.filter(group => 
        group.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        group.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        group.category.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : groups;
  
  // Filter events by search query
  const filteredEvents = searchQuery 
    ? events.filter(event => 
        event.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.location.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : events;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8 text-center max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">Discover Communities & Events</h1>
        <p className="text-muted-foreground mb-6">
          Find and connect with professional communities that match your interests, or discover events to expand your network
        </p>
        
        <div className="relative mb-8">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            type="text"
            placeholder="Search groups and events..."
            className="pl-10 pr-4 py-2 w-full"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>
      
      <Tabs 
        defaultValue={activeTab} 
        className="mb-8"
        onValueChange={setActiveTab}
      >
        <div className="flex justify-center mb-6">
          <TabsList>
            <TabsTrigger value="groups" className="px-6">
              <Users className="h-4 w-4 mr-2" />
              Groups
            </TabsTrigger>
            <TabsTrigger value="events" className="px-6">
              <Calendar className="h-4 w-4 mr-2" />
              Events
            </TabsTrigger>
          </TabsList>
        </div>
        
        <TabsContent value="groups">
          {loadingGroups ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <Card key={i}>
                  <Skeleton className="h-48 w-full" />
                  <CardContent className="p-5">
                    <Skeleton className="h-6 w-3/4 mb-4" />
                    <Skeleton className="h-4 w-1/2 mb-2" />
                    <Skeleton className="h-4 w-1/3 mb-4" />
                    <Skeleton className="h-20 w-full mb-4" />
                    <Skeleton className="h-10 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : filteredGroups.length === 0 ? (
            <div className="text-center py-10">
              <Users className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-2">No matching groups found</h3>
              <p className="text-muted-foreground mb-6">
                Try adjusting your search or filters
              </p>
              <Button onClick={() => setSearchQuery('')}>Clear Search</Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredGroups.map((group) => (
                <Link href={`/groups/${group.id}`} key={group.id}>
                  <Card className="h-full overflow-hidden transition-all hover:shadow-md">
                    <div className="aspect-video w-full overflow-hidden bg-muted">
                      {group.image_url ? (
                        <Image
                          src={group.image_url}
                          alt={group.name}
                          width={400}
                          height={225}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="h-full w-full bg-gradient-to-r from-blue-100 to-indigo-100 flex items-center justify-center">
                          <Users className="h-10 w-10 text-blue-300" />
                        </div>
                      )}
                    </div>
                    
                    <CardContent className="p-5">
                      <div className="flex items-start justify-between mb-2">
                        <h3 className="text-xl font-semibold line-clamp-1">{group.name}</h3>
                        {group.is_private && (
                          <Badge variant="outline" className="ml-2">Private</Badge>
                        )}
                      </div>
                      
                      <div className="flex flex-wrap gap-2 mb-3">
                        <Badge variant="secondary">{group.category}</Badge>
                        <div className="flex items-center text-xs text-muted-foreground">
                          <Users className="h-3 w-3 mr-1" />
                          {group.member_count || 0} members
                        </div>
                      </div>
                      
                      {group.location && (
                        <div className="flex items-center text-sm text-muted-foreground mb-3">
                          <MapPin className="h-3 w-3 mr-1" />
                          {group.location}
                        </div>
                      )}
                      
                      <p className="text-muted-foreground text-sm line-clamp-3 mb-4">
                        {group.description}
                      </p>
                      
                      <Button className="w-full" size="sm">View Group</Button>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          )}
          
          {!loadingGroups && filteredGroups.length > 0 && (
            <div className="mt-8 text-center">
              <Link href="/groups">
                <Button variant="outline">View All Groups</Button>
              </Link>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="events">
          {loadingEvents ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <Card key={i}>
                  <Skeleton className="h-48 w-full" />
                  <CardContent className="p-5">
                    <Skeleton className="h-6 w-3/4 mb-4" />
                    <Skeleton className="h-4 w-1/2 mb-2" />
                    <Skeleton className="h-4 w-2/3 mb-2" />
                    <Skeleton className="h-4 w-1/3 mb-4" />
                    <Skeleton className="h-20 w-full mb-4" />
                    <Skeleton className="h-10 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : filteredEvents.length === 0 ? (
            <div className="text-center py-10">
              <Calendar className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-2">No matching events found</h3>
              <p className="text-muted-foreground mb-6">
                Try adjusting your search or filters
              </p>
              <Button onClick={() => setSearchQuery('')}>Clear Search</Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredEvents.map((event) => (
                <Link href={`/events/${event.id}`} key={event.id}>
                  <Card className="h-full overflow-hidden transition-all hover:shadow-md">
                    <div className="aspect-video w-full overflow-hidden bg-muted">
                      {event.image_url ? (
                        <Image
                          src={event.image_url}
                          alt={event.title}
                          width={400}
                          height={225}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="h-full w-full bg-gradient-to-r from-blue-100 to-indigo-100 flex items-center justify-center">
                          <Calendar className="h-10 w-10 text-blue-300" />
                        </div>
                      )}
                    </div>
                    
                    <CardContent className="p-5">
                      <h3 className="text-xl font-semibold line-clamp-1 mb-2">
                        {event.title}
                      </h3>
                      
                      <div className="flex flex-wrap gap-2 mb-3">
                        <Badge variant="secondary">{event.category}</Badge>
                        <Badge variant="outline" className="capitalize">{event.format || 'In Person'}</Badge>
                      </div>
                      
                      <div className="space-y-2 mb-3">
                        <div className="flex items-center text-sm">
                          <Calendar className="h-3 w-3 mr-2 text-muted-foreground" />
                          <span>{formatDate(event.start_time)}</span>
                        </div>
                        
                        <div className="flex items-center text-sm">
                          <Clock className="h-3 w-3 mr-2 text-muted-foreground" />
                          <span>{formatTime(event.start_time)}</span>
                        </div>
                        
                        <div className="flex items-center text-sm">
                          <MapPin className="h-3 w-3 mr-2 text-muted-foreground" />
                          <span className="line-clamp-1">{event.location}</span>
                        </div>
                      </div>
                      
                      <p className="text-muted-foreground text-sm line-clamp-2 mb-4">
                        {event.description}
                      </p>
                      
                      <Button className="w-full" size="sm">View Event</Button>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          )}
          
          {!loadingEvents && filteredEvents.length > 0 && (
            <div className="mt-8 text-center">
              <Link href="/events">
                <Button variant="outline">View All Events</Button>
              </Link>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Demo data for groups
function getDemoGroups(): Group[] {
  return [
    {
      id: '1',
      name: 'Software Engineers Network',
      description: 'A community of software engineers sharing knowledge, job opportunities, and supporting each other in career development.',
      category: 'Technology',
      location: 'San Francisco, CA',
      member_count: 1250,
      image_url: 'https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3',
      created_at: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString(),
      is_private: false,
    },
    {
      id: '2',
      name: 'Women in Tech',
      description: 'Supporting and promoting women in technical roles through mentorship, resources, and community events.',
      category: 'Professional',
      location: 'New York, NY',
      member_count: 875,
      image_url: 'https://images.unsplash.com/photo-1573164574001-518958d9baa2?ixlib=rb-4.0.3',
      created_at: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString(),
      is_private: false,
    },
    {
      id: '3',
      name: 'Startup Founders Club',
      description: 'Exclusive group for founders to connect, share experiences, and help each other solve challenges in building successful startups.',
      category: 'Entrepreneurship',
      location: 'Boston, MA',
      member_count: 456,
      image_url: 'https://images.unsplash.com/photo-1519389950473-47ba0277781c?ixlib=rb-4.0.3',
      created_at: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(),
      is_private: true,
    },
    {
      id: '4',
      name: 'AI Research Community',
      description: 'Discussing the latest advancements in artificial intelligence and machine learning research.',
      category: 'Technology',
      location: 'Seattle, WA',
      member_count: 620,
      image_url: 'https://images.unsplash.com/photo-1531297484001-80022131f5a1?ixlib=rb-4.0.3',
      created_at: new Date(Date.now() - 120 * 24 * 60 * 60 * 1000).toISOString(),
      is_private: false,
    },
    {
      id: '5',
      name: 'Product Managers Collective',
      description: 'A community for product managers to share insights, discuss challenges, and learn from one another.',
      category: 'Product Management',
      location: 'Chicago, IL',
      member_count: 782,
      image_url: 'https://images.unsplash.com/photo-1553877522-43269d4ea984?ixlib=rb-4.0.3',
      created_at: new Date(Date.now() - 150 * 24 * 60 * 60 * 1000).toISOString(),
      is_private: false,
    },
    {
      id: '6',
      name: 'Data Science Professionals',
      description: 'Connect with data scientists, analysts, and professionals working with big data and analytics.',
      category: 'Technology',
      location: 'Austin, TX',
      member_count: 934,
      image_url: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3',
      created_at: new Date(Date.now() - 200 * 24 * 60 * 60 * 1000).toISOString(),
      is_private: false,
    }
  ];
}

// Demo data for events
function getDemoEvents(): Event[] {
  return [
    {
      id: '1',
      title: 'Tech Networking Mixer',
      description: 'Join us for an evening of networking with tech professionals from across the industry.',
      start_time: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      location: 'San Francisco, CA',
      format: 'in-person',
      category: 'Networking',
      organizer_id: '123',
      organizer_name: 'SF Tech Group',
      image_url: 'https://images.unsplash.com/photo-1511578314322-379afb476865?ixlib=rb-4.0.3',
      attendee_count: 87
    },
    {
      id: '2',
      title: 'Women in Engineering Workshop',
      description: 'A workshop focused on supporting women in engineering roles with mentorship and resources.',
      start_time: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(),
      location: 'New York, NY',
      format: 'hybrid',
      category: 'Workshop',
      organizer_id: '456',
      organizer_name: 'Women Engineers Alliance',
      image_url: 'https://images.unsplash.com/photo-1573164574572-cb89e39749b4?ixlib=rb-4.0.3',
      attendee_count: 152
    },
    {
      id: '3',
      title: 'AI and Machine Learning Conference',
      description: 'Explore the latest advancements in AI and machine learning with industry experts.',
      start_time: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
      end_time: new Date(Date.now() + 32 * 24 * 60 * 60 * 1000).toISOString(),
      location: 'Austin, TX',
      format: 'in-person',
      category: 'Conference',
      organizer_id: '789',
      organizer_name: 'AI Research Collective',
      image_url: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?ixlib=rb-4.0.3',
      attendee_count: 423
    },
    {
      id: '4',
      title: 'Product Management Virtual Meetup',
      description: 'Connect with fellow product managers and learn from experienced speakers in our monthly virtual meetup.',
      start_time: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(),
      location: 'Online',
      format: 'virtual',
      category: 'Meetup',
      organizer_id: '101',
      organizer_name: 'Product Managers Association',
      image_url: 'https://images.unsplash.com/photo-1591115765373-5207764f72e4?ixlib=rb-4.0.3',
      attendee_count: 64
    },
    {
      id: '5',
      title: 'Data Science Summit',
      description: 'A two-day summit focused on big data, analytics, and the future of data science.',
      start_time: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000).toISOString(),
      end_time: new Date(Date.now() + 46 * 24 * 60 * 60 * 1000).toISOString(),
      location: 'Chicago, IL',
      format: 'in-person',
      category: 'Summit',
      organizer_id: '202',
      organizer_name: 'Data Science Institute',
      image_url: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3',
      attendee_count: 278
    },
    {
      id: '6',
      title: 'Startup Pitch Night',
      description: 'Watch innovative startups pitch their ideas to investors and network with entrepreneurs.',
      start_time: new Date(Date.now() + 10 * 24 * 60 * 60 * 1000).toISOString(),
      location: 'Boston, MA',
      format: 'in-person',
      category: 'Networking',
      organizer_id: '303',
      organizer_name: 'Founders Club',
      image_url: 'https://images.unsplash.com/photo-1519389950473-47ba0277781c?ixlib=rb-4.0.3',
      attendee_count: 112
    }
  ];
} 