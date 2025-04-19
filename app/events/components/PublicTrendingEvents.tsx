'use client';

import { useRef } from 'react';
import Link from 'next/link';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface EventItem {
  id: string;
  title: string;
  start_time: string;
  location: string;
  image_url?: string;
}

interface PublicTrendingEventsProps {
  events: EventItem[];
  isAuthenticated: boolean;
}

export default function PublicTrendingEvents({ events, isAuthenticated }: PublicTrendingEventsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollLeft = () => containerRef.current?.scrollBy({ left: -300, behavior: 'smooth' });
  const scrollRight = () => containerRef.current?.scrollBy({ left: 300, behavior: 'smooth' });

  if (!events || events.length === 0) return null;

  return (
    <section className="section container mx-auto">
      <h2 className="text-3xl font-extrabold mb-4 text-[rgb(var(--networkli-orange))]">Trending Events</h2>

      <div className="relative">
        <button onClick={scrollLeft} className="absolute left-0 top-1/2 transform -translate-y-1/2 p-2 bg-white/50 backdrop-blur rounded-full shadow-lg z-10 hover:bg-white transition-colors">
          <ChevronLeft className="h-6 w-6 text-gray-600" />
        </button>

        <div ref={containerRef} className="flex space-x-6 overflow-x-auto px-6 py-4 scrollbar-thin scrollbar-thumb-gray-300 snap-x snap-mandatory">
          {events.map(evt => (
            <div key={evt.id} className="snap-start flex-shrink-0 w-64 bg-white/30 backdrop-blur-sm p-4 rounded-2xl shadow-lg transform hover:scale-105 transition-transform duration-200">
              {evt.image_url && (
                <div className="h-32 w-full overflow-hidden rounded-lg">
                  <img src={evt.image_url} alt={evt.title} className="w-full h-full object-cover" />
                </div>
              )}

              <h3 className="text-lg font-semibold mt-2 text-gray-800">{evt.title}</h3>
              <p className="text-sm text-gray-600 mt-1">
                {new Date(evt.start_time).toLocaleDateString()} &middot; {evt.location}
              </p>

              <Link
                href={
                  isAuthenticated
                    ? `/events/${evt.id}`
                    : `/signup?redirect=${encodeURIComponent(`/events/${evt.id}`)}`
                }
                className="mt-4 inline-block button-primary"
              >
                {isAuthenticated ? 'View Event' : 'Register'}
              </Link>
            </div>
          ))}
        </div>

        <button onClick={scrollRight} className="absolute right-0 top-1/2 transform -translate-y-1/2 p-2 bg-white/50 backdrop-blur rounded-full shadow-lg z-10 hover:bg-white transition-colors">
          <ChevronRight className="h-6 w-6 text-gray-600" />
        </button>
      </div>
    </section>
  );
} 