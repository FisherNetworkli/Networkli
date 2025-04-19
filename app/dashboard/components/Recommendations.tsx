'use client';

import { Loader2, Users, Lightbulb, Sparkles, Calendar, Tv } from 'lucide-react';
import Link from 'next/link';
import { Card, CardHeader, CardContent, CardTitle } from '@/components/ui/card';

interface ApiRecommendation {
  id: string;
  name?: string;
  title?: string;
  avatar_url?: string;
  reason?: string;
}

interface RecommendationsProps {
  connections: ApiRecommendation[];
  groups: ApiRecommendation[];
  events: ApiRecommendation[];
  loading: boolean;
}

export default function Recommendations({ connections, groups, events, loading }: RecommendationsProps) {
  if (loading) {
    return (
      <div className="flex justify-center items-center py-6">
        <Loader2 className="h-6 w-6 animate-spin text-[rgb(var(--networkli-orange))]" />
        <span className="ml-2 text-[rgb(var(--networkli-orange))]">Just for you...</span>
      </div>
    );
  }

  return (
    <section className="section container mx-auto space-y-6">
      <div className="card-frosted">
        <h2 className="text-2xl font-semibold mb-4 text-[rgb(var(--connection-blue))]">Just For You</h2>
        <div className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {/* Connections */}
          {connections.length > 0 && (
            <Card className="border-l-4 border-[rgb(var(--connection-blue))] bg-white/40 backdrop-blur-sm hover:scale-105 transform transition duration-200 ease-out">
              <CardHeader>
                <CardTitle className="text-lg flex items-center">
                  <Users className="h-5 w-5 mr-2 text-[rgb(var(--connection-blue))]" /> Connections
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {connections.map(rec => (
                    <li key={rec.id} className="flex items-center space-x-3 text-sm hover:scale-105 transform transition duration-150 ease-out">
                      <img
                        src={rec.avatar_url ?? '/placeholder-avatar.png'}
                        alt={rec.name}
                        className="h-8 w-8 rounded-full object-cover"
                      />
                      <Link href="/dashboard/network/swipe" className="font-medium text-gray-800 hover:text-[rgb(var(--connection-blue))]">
                        {rec.name || rec.title || 'User'}
                      </Link>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          {/* Groups */}
          {groups.length > 0 && (
            <Card className="border-l-4 border-[rgb(var(--networkli-orange))] bg-white/40 backdrop-blur-sm hover:scale-105 transform transition duration-200 ease-out">
              <CardHeader>
                <CardTitle className="text-lg flex items-center">
                  <Lightbulb className="h-5 w-5 mr-2 text-[rgb(var(--networkli-orange))]" /> Groups
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {groups.map(rec => (
                    <li key={rec.id} className="flex items-center space-x-3 text-sm hover:scale-105 transform transition duration-150 ease-out">
                      <Sparkles className="h-5 w-5 text-purple-500" />
                      <Link href="/dashboard/network/swipe" className="font-medium text-gray-800 hover:text-[rgb(var(--networkli-orange))]">
                        {rec.name}
                      </Link>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          {/* Events */}
          {events.length > 0 && (
            <Card className="border-l-4 border-[rgb(var(--networkli-orange-70))] bg-white/40 backdrop-blur-sm hover:scale-105 transform transition duration-200 ease-out">
              <CardHeader>
                <CardTitle className="text-lg flex items-center">
                  <Calendar className="h-5 w-5 mr-2 text-[rgb(var(--networkli-orange-70))]" /> Events
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {events.map(rec => (
                    <li key={rec.id} className="flex items-center space-x-3 text-sm hover:scale-105 transform transition duration-150 ease-out">
                      <Tv className="h-5 w-5 text-[rgb(var(--networkli-orange-70))]" />
                      <Link href="/dashboard/network/swipe" className="font-medium text-gray-800 hover:text-[rgb(var(--networkli-orange-70))]">
                        {rec.title || rec.name || 'Event'}
                      </Link>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </section>
  );
} 