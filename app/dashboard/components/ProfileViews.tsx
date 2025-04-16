'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Eye, Clock, Activity, Users, ArrowUpRight } from 'lucide-react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';
import Link from 'next/link';

interface ProfileViewEntry {
  viewer_id: string;
  viewed_at: string;
  viewer_full_name: string | null;
  viewer_avatar_url: string | null;
  source: string | null;
}

interface ProfileViewsProps {
  user: User;
  isPremium?: boolean;
}

export function ProfileViews({ user, isPremium = false }: ProfileViewsProps) {
  const [profileViews, setProfileViews] = useState<ProfileViewEntry[]>([]);
  const [stats, setStats] = useState({
    totalViews: 0,
    uniqueViewers: 0,
    recentViews: 0,
    sources: {} as Record<string, number>
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const supabase = createClientComponentClient();

  useEffect(() => {
    const fetchProfileViews = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data: sessionData } = await supabase.auth.getSession();
        if (!sessionData.session) {
          throw new Error('No authenticated session');
        }

        // Fetch profile views from the API
        const response = await fetch('/api/dashboard', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${sessionData.session.access_token}`
          }
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch profile views: ${response.statusText}`);
        }

        const data = await response.json();
        
        setProfileViews(data.profile_views || []);
        setStats({
          totalViews: data.profile_view_count || 0,
          uniqueViewers: data.unique_viewers_count || 0,
          recentViews: data.recent_views_count || 0,
          sources: data.view_sources || {}
        });
      } catch (err) {
        console.error('Error fetching profile views:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        
        // Set mock data for development
        const mockViews = [
          {
            viewer_id: 'viewer1',
            viewed_at: new Date().toISOString(),
            viewer_full_name: 'Alice Smith',
            viewer_avatar_url: 'https://example.com/avatars/alice.jpg',
            source: 'search'
          },
          {
            viewer_id: 'viewer2',
            viewed_at: new Date().toISOString(),
            viewer_full_name: 'Bob Jones',
            viewer_avatar_url: 'https://example.com/avatars/bob.jpg',
            source: 'recommendation'
          },
          {
            viewer_id: 'viewer3',
            viewed_at: new Date().toISOString(),
            viewer_full_name: 'Carol White',
            viewer_avatar_url: 'https://example.com/avatars/carol.jpg',
            source: 'direct'
          }
        ];
        setProfileViews(isPremium ? mockViews : [mockViews[0]]);
        setStats({
          totalViews: 3,
          uniqueViewers: 3,
          recentViews: 3,
          sources: { search: 1, recommendation: 1, direct: 1 }
        });
      } finally {
        setLoading(false);
      }
    };

    fetchProfileViews();
  }, [supabase, user, isPremium]);

  // Helper function to format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric'
    }).format(date);
  };

  // Helper function to get initials from name
  const getInitials = (name: string | null) => {
    if (!name) return '?';
    return name.split(' ')
      .map(part => part.charAt(0).toUpperCase())
      .slice(0, 2)
      .join('');
  };

  // Get source badge color
  const getSourceColor = (source: string | null) => {
    switch (source?.toLowerCase()) {
      case 'search': return 'bg-blue-100 text-blue-800';
      case 'recommendation': return 'bg-green-100 text-green-800';
      case 'direct': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Profile Views</h2>
          <p className="text-muted-foreground">
            See who's been viewing your profile
          </p>
        </div>
        {!isPremium && (
          <Link href="/premium">
            <Button className="mt-2 sm:mt-0" variant="outline">
              Upgrade to Premium <ArrowUpRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        )}
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Views</CardTitle>
            <Eye className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loading ? '...' : stats.totalViews}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unique Viewers</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loading ? '...' : stats.uniqueViewers}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Recent Views</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loading ? '...' : stats.recentViews}</div>
            <p className="text-xs text-muted-foreground">Last 7 days</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Top Source</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">
              {loading ? '...' : 
                Object.entries(stats.sources).sort((a, b) => b[1] - a[1])[0]?.[0] || 'None'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Profile view list */}
      <Card>
        <CardHeader>
          <CardTitle>View History</CardTitle>
          {!isPremium && profileViews.length > 0 && (
            <p className="text-sm text-muted-foreground">
              Upgrade to premium to see all {stats.totalViews} profile views
            </p>
          )}
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center h-40">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-900"></div>
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <p className="text-red-500">Error loading profile views</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
          ) : profileViews.length === 0 ? (
            <div className="text-center py-8">
              <p>No profile views yet</p>
              <p className="text-sm text-muted-foreground">
                When someone views your profile, they'll appear here
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {profileViews.map((view, index) => (
                <div key={`${view.viewer_id}-${index}`} className="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50">
                  <Avatar>
                    <AvatarImage src={view.viewer_avatar_url || undefined} />
                    <AvatarFallback>{getInitials(view.viewer_full_name)}</AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{view.viewer_full_name || 'Anonymous User'}</p>
                    <p className="text-sm text-muted-foreground">
                      Viewed {formatDate(view.viewed_at)}
                    </p>
                  </div>
                  {view.source && (
                    <Badge className={`${getSourceColor(view.source)} capitalize`}>
                      {view.source}
                    </Badge>
                  )}
                </div>
              ))}
              
              {!isPremium && stats.totalViews > 1 && (
                <div className="text-center pt-4 pb-2">
                  <Link href="/premium">
                    <Button variant="outline">
                      See {stats.totalViews - 1} more profile views
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 