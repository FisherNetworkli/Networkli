'use client';

import { useEffect, useState } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import Link from 'next/link';
import { Button } from '@/components/ui/button';

interface Event {
  id: string;
  title: string;
  description: string;
  date: string;
  location: string;
  organizer_id: string;
  organizer_name?: string;
  image_url?: string;
}

export default function EventsPage() {
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const supabase = createClientComponentClient();
  const [role, setRole] = useState<string | null>(null);

  // Fetch current user's role for permission
  useEffect(() => {
    const getRole = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;
      const { data: profile } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', session.user.id)
        .single();
      setRole(profile?.role || null);
    };
    getRole();
  }, [supabase]);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        // Fetch events from the database
        const { data, error } = await supabase
          .from('events')
          .select('*')
          .order('date', { ascending: true });

        if (error) {
          console.error('Error fetching events:', error);
          setError(`Error loading events: ${error.message}`);
          
          // Use placeholder data if there's an error
          const placeholderEvents: Event[] = [
            {
              id: '1',
              title: 'Tech Networking Mixer',
              description: 'Join us for an evening of networking with tech professionals from across the industry.',
              date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
              location: 'San Francisco, CA',
              organizer_id: '123',
              organizer_name: 'SF Tech Group',
              image_url: 'https://images.unsplash.com/photo-1511578314322-379afb476865?ixlib=rb-4.0.3',
            },
            {
              id: '2',
              title: 'Women in Engineering Workshop',
              description: 'A workshop focused on supporting women in engineering roles with mentorship and resources.',
              date: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(),
              location: 'New York, NY',
              organizer_id: '456',
              organizer_name: 'Women Engineers Alliance',
              image_url: 'https://images.unsplash.com/photo-1573164574572-cb89e39749b4?ixlib=rb-4.0.3',
            },
            {
              id: '3',
              title: 'AI and Machine Learning Conference',
              description: 'Explore the latest advancements in AI and machine learning with industry experts.',
              date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
              location: 'Austin, TX',
              organizer_id: '789',
              organizer_name: 'AI Research Collective',
              image_url: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?ixlib=rb-4.0.3',
            },
          ];
          setEvents(placeholderEvents);
        } else {
          // For demo purposes, if no events in DB yet, use placeholder data
          if (!data || data.length === 0) {
            const placeholderEvents: Event[] = [
              {
                id: '1',
                title: 'Tech Networking Mixer',
                description: 'Join us for an evening of networking with tech professionals from across the industry.',
                date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
                location: 'San Francisco, CA',
                organizer_id: '123',
                organizer_name: 'SF Tech Group',
                image_url: 'https://images.unsplash.com/photo-1511578314322-379afb476865?ixlib=rb-4.0.3',
              },
              {
                id: '2',
                title: 'Women in Engineering Workshop',
                description: 'A workshop focused on supporting women in engineering roles with mentorship and resources.',
                date: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(),
                location: 'New York, NY',
                organizer_id: '456',
                organizer_name: 'Women Engineers Alliance',
                image_url: 'https://images.unsplash.com/photo-1573164574572-cb89e39749b4?ixlib=rb-4.0.3',
              },
              {
                id: '3',
                title: 'AI and Machine Learning Conference',
                description: 'Explore the latest advancements in AI and machine learning with industry experts.',
                date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
                location: 'Austin, TX',
                organizer_id: '789',
                organizer_name: 'AI Research Collective',
                image_url: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?ixlib=rb-4.0.3',
              },
            ];
            setEvents(placeholderEvents);
          } else {
            setEvents(data);
          }
        }
      } catch (err) {
        console.error('Error:', err);
        setError('An unexpected error occurred while loading events.');
      } finally {
        setLoading(false);
      }
    };

    fetchEvents();
  }, [supabase]);

  // Format date for display
  const formatDate = (dateString: string) => {
    const options: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold">Upcoming Events</h1>
        <div className="flex space-x-2">
          <Link
            href="/discover?tab=events"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded"
          >
            Find Events
          </Link>
          {(role === 'organizer' || role === 'admin') && (
            <Button asChild>
              <Link href="/events/create">Create Event</Link>
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-800">
          <p>{error}</p>
          <p className="text-sm mt-1">Showing sample events instead.</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {events.map((event) => (
          <div
            key={event.id}
            className="bg-white rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow duration-300"
          >
            {event.image_url && (
              <div className="h-48 overflow-hidden">
                <img
                  src={event.image_url}
                  alt={event.title}
                  className="w-full h-full object-cover"
                />
              </div>
            )}
            <div className="p-6">
              <h3 className="text-xl font-bold mb-2 text-gray-800">{event.title}</h3>
              <p className="text-sm text-gray-600 mb-4">
                <span className="font-medium">When:</span> {formatDate(event.date)}
              </p>
              <p className="text-sm text-gray-600 mb-4">
                <span className="font-medium">Where:</span> {event.location}
              </p>
              <p className="text-sm text-gray-600 mb-4">
                <span className="font-medium">Organizer:</span> {event.organizer_name}
              </p>
              <p className="text-gray-700 mb-6 line-clamp-3">{event.description}</p>
              <Link
                href={`/events/${event.id}`}
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded text-sm w-full block text-center"
              >
                View Details
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 