'use client';

import { useRef } from 'react';
import { ChevronLeft, ChevronRight, UserCheck } from 'lucide-react';
import Link from 'next/link';

interface UserRec {
  id: string;
  first_name?: string;
  last_name?: string;
  avatar_url?: string;
  title?: string;
}

interface PeopleCarouselProps {
  connections: UserRec[];
}

export default function PeopleCarousel({ connections }: PeopleCarouselProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const scrollLeft = () => containerRef.current?.scrollBy({ left: -300, behavior: 'smooth' });
  const scrollRight = () => containerRef.current?.scrollBy({ left: 300, behavior: 'smooth' });

  if (!connections || connections.length === 0) return null;

  return (
    <div className="relative mt-8">
      <h2 className="text-3xl font-extrabold mb-4 text-[rgb(var(--connection-blue))]">People You May Know</h2>
      <div className="relative">
        <button
          onClick={scrollLeft}
          className="absolute left-0 top-1/2 transform -translate-y-1/2 p-2 bg-white/50 backdrop-blur rounded-full shadow-lg z-10 hover:bg-white transition"
        >
          <ChevronLeft className="h-6 w-6 text-gray-600" />
        </button>
        <div
          ref={containerRef}
          className="flex space-x-6 overflow-x-auto px-8 py-4 scrollbar-thin scrollbar-thumb-gray-300 snap-x snap-mandatory"
        >
          {connections.map((u) => (
            <div key={u.id} className="snap-start flex-shrink-0 w-48 transform transition hover:scale-105">
              <div className="relative">
                {u.avatar_url ? (
                  <img
                    src={u.avatar_url}
                    alt={`${u.first_name} ${u.last_name}`}
                    className="h-32 w-32 rounded-full object-cover mx-auto"
                  />
                ) : (
                  <div className="h-32 w-32 rounded-full bg-gray-200 flex items-center justify-center mx-auto">
                    <UserCheck className="h-8 w-8 text-gray-500" />
                  </div>
                )}
                <div className="absolute -bottom-2 -right-2 bg-[rgb(var(--networkli-orange))] text-white rounded-full p-1 shadow-lg">
                  <UserCheck className="h-4 w-4" />
                </div>
              </div>
              <h3 className="mt-3 text-lg font-medium text-gray-800 text-center">
                {u.first_name} {u.last_name}
              </h3>
              {u.title && (
                <p className="text-sm text-gray-600 text-center">{u.title}</p>
              )}
              <Link
                href={`/dashboard/profile/${u.id}?from=carousel`}
                className="mt-4 block text-center bg-[rgb(var(--networkli-orange))] text-white py-1 rounded-lg hover:bg-[rgb(var(--networkli-orange-70))] transition"
              >
                Connect
              </Link>
            </div>
          ))}
        </div>
        <button
          onClick={scrollRight}
          className="absolute right-0 top-1/2 transform -translate-y-1/2 p-2 bg-white/50 backdrop-blur rounded-full shadow-lg z-10 hover:bg-white transition"
        >
          <ChevronRight className="h-6 w-6 text-gray-600" />
        </button>
      </div>
    </div>
  );
} 