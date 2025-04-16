'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Eye, Clock, Activity, Users, UserPlus, LineChart, BarChart, PieChart,
  Calendar, TrendingUp, Filter
} from 'lucide-react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ProfileViewEntry {
  viewer_id: string;
  viewed_at: string;
  viewer_full_name: string | null;
  viewer_avatar_url: string | null;
  source: string | null;
}

interface OrganizerProfileViewsProps {
  user: User;
}

export function OrganizerProfileViews({ user }: OrganizerProfileViewsProps) {
  const [profileViews, setProfileViews] = useState<ProfileViewEntry[]>([]);
  const [memberStats, setMemberStats] = useState({
    totalMembers: 0,
    activeMembers: 0,
    newMembers: 0,
    memberEngagement: { high: 0, medium: 0, low: 0 }
  });
  const [stats, setStats] = useState({
    totalViews: 0,
    uniqueViewers: 0,
    recentViews: 0,
    sources: {} as Record<string, number>
  });
  const [timeFilter, setTimeFilter] = useState('7days');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const supabase = createClientComponentClient();

  useEffect(() => {
    const fetchOrganizerData = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data: sessionData } = await supabase.auth.getSession();
        if (!sessionData.session) {
          throw new Error('No authenticated session');
        }

        // Fetch organizer dashboard data from the API
        const response = await fetch('/api/organizer/dashboard', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${sessionData.session.access_token}`
          }
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch organizer data: ${response.statusText}`);
        }

        const data = await response.json();
        
        // Set profile views data
        setProfileViews(data.profile_view_data?.profile_views || []);
        setStats({
          totalViews: data.profile_view_data?.profile_view_count || 0,
          uniqueViewers: data.profile_view_data?.unique_viewers_count || 0,
          recentViews: data.profile_view_data?.recent_views_count || 0,
          sources: data.profile_view_data?.view_sources || {}
        });

        // Set member statistics
        setMemberStats({
          totalMembers: data.member_stats?.total_members || 0,
          activeMembers: data.member_stats?.active_members || 0,
          newMembers: data.member_stats?.new_members_last_month || 0,
          memberEngagement: data.member_stats?.member_engagement || { high: 0, medium: 0, low: 0 }
        });
      } catch (err) {
        console.error('Error fetching organizer data:', err);
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
        setProfileViews(mockViews);
        setStats({
          totalViews: 3,
          uniqueViewers: 3,
          recentViews: 3,
          sources: { search: 1, recommendation: 1, direct: 1 }
        });
        setMemberStats({
          totalMembers: 120,
          activeMembers: 75,
          newMembers: 12,
          memberEngagement: { high: 25, medium: 45, low: 50 }
        });
      } finally {
        setLoading(false);
      }
    };

    fetchOrganizerData();
  }, [supabase, user]);

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
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Organizer Dashboard</h2>
          <p className="text-muted-foreground">
            Enhanced analytics and member management
          </p>
        </div>
        <div className="flex items-center gap-2 mt-2 sm:mt-0">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <Select
            value={timeFilter}
            onValueChange={setTimeFilter}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Time period" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7days">Last 7 days</SelectItem>
              <SelectItem value="30days">Last 30 days</SelectItem>
              <SelectItem value="90days">Last 90 days</SelectItem>
              <SelectItem value="all">All time</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <Tabs defaultValue="member-stats" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="member-stats">Member Statistics</TabsTrigger>
          <TabsTrigger value="profile-views">Profile Views</TabsTrigger>
        </TabsList>
        
        {/* Member Statistics Tab */}
        <TabsContent value="member-stats" className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Members</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {loading ? '...' : memberStats.totalMembers}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Members</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {loading ? '...' : memberStats.activeMembers}
                </div>
                <p className="text-xs text-muted-foreground">
                  {loading ? '' : `${Math.round((memberStats.activeMembers / memberStats.totalMembers) * 100)}% of total`}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">New Members</CardTitle>
                <UserPlus className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {loading ? '...' : memberStats.newMembers}
                </div>
                <p className="text-xs text-muted-foreground">Last month</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Growth</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">
                  {loading ? '...' : `+${Math.round((memberStats.newMembers / memberStats.totalMembers) * 100)}%`}
                </div>
                <p className="text-xs text-muted-foreground">Month over month</p>
              </CardContent>
            </Card>
          </div>

          {/* Member engagement breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Member Engagement</CardTitle>
              <CardDescription>
                Breakdown of member activity levels
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div className="flex flex-col items-center p-4 rounded-lg bg-green-50">
                  <h3 className="font-medium text-green-800">High Engagement</h3>
                  <p className="text-3xl font-bold text-green-600">
                    {loading ? '...' : memberStats.memberEngagement.high}
                  </p>
                  <p className="text-sm text-green-700">
                    {loading ? '' : `${Math.round((memberStats.memberEngagement.high / memberStats.totalMembers) * 100)}%`}
                  </p>
                </div>
                <div className="flex flex-col items-center p-4 rounded-lg bg-blue-50">
                  <h3 className="font-medium text-blue-800">Medium Engagement</h3>
                  <p className="text-3xl font-bold text-blue-600">
                    {loading ? '...' : memberStats.memberEngagement.medium}
                  </p>
                  <p className="text-sm text-blue-700">
                    {loading ? '' : `${Math.round((memberStats.memberEngagement.medium / memberStats.totalMembers) * 100)}%`}
                  </p>
                </div>
                <div className="flex flex-col items-center p-4 rounded-lg bg-orange-50">
                  <h3 className="font-medium text-orange-800">Low Engagement</h3>
                  <p className="text-3xl font-bold text-orange-600">
                    {loading ? '...' : memberStats.memberEngagement.low}
                  </p>
                  <p className="text-sm text-orange-700">
                    {loading ? '' : `${Math.round((memberStats.memberEngagement.low / memberStats.totalMembers) * 100)}%`}
                  </p>
                </div>
              </div>
              
              <div className="flex justify-center mt-8">
                <Button>
                  <LineChart className="h-4 w-4 mr-2" />
                  View Detailed Analytics
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Profile Views Tab */}
        <TabsContent value="profile-views" className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
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
                <p className="text-xs text-muted-foreground">
                  {loading ? '' : `${Math.round((stats.uniqueViewers / stats.totalViews) * 100)}% of total views`}
                </p>
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

          {/* Sources Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Traffic Sources</CardTitle>
              <CardDescription>
                How members are finding your organization
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="font-medium mb-4">Source Breakdown</h3>
                  <div className="space-y-2">
                    {Object.entries(stats.sources).map(([source, count]) => (
                      <div key={source} className="flex items-center">
                        <div className="w-32 font-medium capitalize">{source}</div>
                        <div className="flex-1">
                          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${source === 'search' ? 'bg-blue-500' : 
                                source === 'recommendation' ? 'bg-green-500' : 
                                source === 'direct' ? 'bg-purple-500' : 'bg-gray-500'}`}
                              style={{ width: `${Math.round((count / stats.totalViews) * 100)}%` }}
                            ></div>
                          </div>
                        </div>
                        <div className="w-16 text-right text-muted-foreground">
                          {Math.round((count / stats.totalViews) * 100)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="flex items-center justify-center">
                  <div className="h-48 w-48 rounded-full border-8 border-gray-100 flex items-center justify-center">
                    <PieChart className="h-24 w-24 text-muted-foreground" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Profile view list */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Profile Views</CardTitle>
              <CardDescription>
                {stats.totalViews} total views from {stats.uniqueViewers} unique visitors
              </CardDescription>
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
                    When someone views your organization, they'll appear here
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
                  
                  <div className="text-center pt-4 pb-2">
                    <Button variant="outline">
                      <BarChart className="h-4 w-4 mr-2" />
                      View All Profile Views
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 