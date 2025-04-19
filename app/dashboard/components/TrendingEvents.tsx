'use client';

import { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import Link from 'next/link';

interface TrendingEvent {
  id: string;
  name: string;
  date: string;
  location: string;
}

interface TrendingEventsProps {
  events: TrendingEvent[];
}

export default function TrendingEvents({ events }: TrendingEventsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollLeft = () => containerRef.current?.scrollBy({ left: -300, behavior: 'smooth' });
  const scrollRight = () => containerRef.current?.scrollBy({ left: 300, behavior: 'smooth' });

  if (!events || events.length === 0) return null;

  return (
    <div className="relative mt-8">
      <h2 className="text-3xl font-extrabold mb-4 text-[rgb(var(--networkli-orange))]">Trending Events</h2>
      <div className="relative">
        <button
          onClick={scrollLeft}
          className="absolute left-0 top-1/2 transform -translate-y-1/2 p-2 bg-white/50 backdrop-blur rounded-full shadow-lg z-10 hover:bg-white transition-colors"
        >
          <ChevronLeft className="h-6 w-6 text-gray-600" />
        </button>

        <div
          ref={containerRef}
          className="flex space-x-6 overflow-x-auto px-6 py-4 scrollbar-thin scrollbar-thumb-gray-300 snap-x snap-mandatory"
        >
          {events.map(evt => (
            <div
              key={evt.id}
              className="snap-start flex-shrink-0 w-64 bg-white/30 backdrop-blur-sm p-4 rounded-2xl shadow-lg transform hover:scale-105 transition-transform duration-200"
            >
              <h3 className="text-lg font-semibold text-gray-800">{evt.name}</h3>
              <p className="text-sm text-gray-600 mt-1">
                {new Date(evt.date).toLocaleDateString()} &middot; {evt.location}
              </p>
              <Link
                href={`/dashboard/events/${evt.id}`}
                className="mt-4 inline-block bg-[rgb(var(--networkli-orange))] text-white px-4 py-2 rounded-lg hover:bg-[rgb(var(--networkli-orange-70))] transition-colors"
              >
                View Event
              </Link>
            </div>
          ))}
        </div>

        <button
          onClick={scrollRight}
          className="absolute right-0 top-1/2 transform -translate-y-1/2 p-2 bg-white/50 backdrop-blur rounded-full shadow-lg z-10 hover:bg-white transition-colors"
        >
          <ChevronRight className="h-6 w-6 text-gray-600" />
        </button>
      </div>
    </div>
  );
} 