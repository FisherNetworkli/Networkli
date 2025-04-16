'use client';

import { useState, useEffect } from 'react';
import { User } from '@supabase/supabase-js';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  TooltipProps,
  PieLabelRenderProps
} from 'recharts';
import { Loader2 } from 'lucide-react';

interface OrganizerDashboardProps {
  user: User;
}

// New types for the enhanced dashboard data
interface GroupMemberStats {
  name?: string;
  total_members: number;
  new_members_30d: number;
  active_members: number;
  location_distribution: Record<string, number>;
  industry_distribution: Record<string, number>;
  skill_distribution: Record<string, number>;
  engagement_levels: {
    high: number;
    medium: number;
    low: number;
  };
}

interface EventAttendeeStats {
  title?: string;
  total_registered: number;
  total_attended: number;
  new_attendees: number;
  returning_attendees: number;
  location_distribution: Record<string, number>;
  industry_distribution: Record<string, number>;
}

interface DemographicSummary {
  top_locations: Record<string, number>;
  top_industries: Record<string, number>;
  top_skills: string[];
  age_groups?: Record<string, number>;
}

interface OrganizerDashboardData {
  group_stats: Record<string, GroupMemberStats>;
  event_stats: Record<string, EventAttendeeStats>;
  total_reach: number;
  member_growth_rate: number;
  engagement_rate: number;
  demographic_summary: DemographicSummary;
  retention_rate: number;
}

// Helper function to transform object data into array format for charts
const objectToChartData = (obj: Record<string, number>) => {
  return Object.entries(obj).map(([name, value]) => ({ name, value }));
};

// Colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1'];

export function OrganizerDashboard({ user }: OrganizerDashboardProps) {
  const [dashboardData, setDashboardData] = useState<OrganizerDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimePeriod, setSelectedTimePeriod] = useState('30d');
  const supabase = createClientComponentClient();

  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const { data: sessionData } = await supabase.auth.getSession();
        if (!sessionData.session) {
          throw new Error('No authenticated session');
        }

        // Fetch data from our new endpoint
        const response = await fetch(`/api/organizer/dashboard?time_period=${selectedTimePeriod}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${sessionData.session.access_token}`
          }
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch dashboard data: ${response.statusText}`);
        }

        const data = await response.json();
        setDashboardData(data);
      } catch (err) {
        console.error('Error fetching organizer dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [supabase, user.id, selectedTimePeriod]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-muted-foreground">Loading dashboard data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <p className="text-red-500">{error}</p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <p className="text-muted-foreground">No dashboard data available.</p>
      </div>
    );
  }

  // Transform demographic data for charts
  const topLocationsData = objectToChartData(dashboardData.demographic_summary.top_locations || {});
  const topIndustriesData = objectToChartData(dashboardData.demographic_summary.top_industries || {});
  
  // Engagement data for doughnut chart
  const engagementData = [];
  let totalEngagement = 0;
  
  // Calculate total engagement across all groups
  Object.values(dashboardData.group_stats).forEach(group => {
    totalEngagement += group.engagement_levels.high + group.engagement_levels.medium + group.engagement_levels.low;
  });
  
  if (totalEngagement > 0) {
    let highTotal = 0;
    let mediumTotal = 0;
    let lowTotal = 0;
    
    Object.values(dashboardData.group_stats).forEach(group => {
      highTotal += group.engagement_levels.high;
      mediumTotal += group.engagement_levels.medium;
      lowTotal += group.engagement_levels.low;
    });
    
    engagementData.push(
      { name: 'High Engagement', value: highTotal },
      { name: 'Medium Engagement', value: mediumTotal },
      { name: 'Low Engagement', value: lowTotal },
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold tracking-tight">Organizer Dashboard</h1>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-muted-foreground">Time Period:</span>
          <select 
            className="border rounded p-1 text-sm" 
            value={selectedTimePeriod}
            onChange={(e) => setSelectedTimePeriod(e.target.value)}
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
            <option value="365d">Last year</option>
          </select>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Reach</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardData.total_reach.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Total people engaged with your content</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Member Growth</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardData.member_growth_rate}%</div>
            <p className="text-xs text-muted-foreground">Increase in membership this period</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Engagement Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardData.engagement_rate}%</div>
            <p className="text-xs text-muted-foreground">Members actively participating</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Retention Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardData.retention_rate}%</div>
            <p className="text-xs text-muted-foreground">Members who continue to engage</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="demographics">
        <TabsList className="mb-4">
          <TabsTrigger value="demographics">Demographics</TabsTrigger>
          <TabsTrigger value="groups">Groups</TabsTrigger>
          <TabsTrigger value="events">Events</TabsTrigger>
          <TabsTrigger value="engagement">Engagement</TabsTrigger>
        </TabsList>
        
        {/* Demographics Tab */}
        <TabsContent value="demographics" className="space-y-4">
          <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Top Locations</CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={topLocationsData}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis 
                      dataKey="name"
                      type="category"
                      width={80}
                      tick={{ fontSize: 12 }}
                    />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" fill="#8884d8" name="Number of Members" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Industry Distribution</CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={topIndustriesData}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      label={({ name, percent }: PieLabelRenderProps) => {
                        // Check if percent is defined before using it
                        const percentage = percent !== undefined ? (percent * 100).toFixed(0) : 'N/A';
                        return `${name}: ${percentage}%`;
                      }}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {topIndustriesData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => [`${value} Members`, 'Count']} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>Top Skills</CardTitle>
            </CardHeader>
            <CardContent className="h-80">
              {dashboardData.demographic_summary.top_skills && dashboardData.demographic_summary.top_skills.length > 0 ? (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {dashboardData.demographic_summary.top_skills.map((skill, index) => (
                    <div 
                      key={index} 
                      className="bg-muted p-3 rounded-md text-center flex items-center justify-center"
                      style={{
                        backgroundColor: `${COLORS[index % COLORS.length]}20`,
                        borderLeft: `4px solid ${COLORS[index % COLORS.length]}`
                      }}
                    >
                      {skill}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-center text-muted-foreground">No skill data available</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Groups Tab */}
        <TabsContent value="groups" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Your Groups</h2>
            <Button>Create New Group</Button>
          </div>

          {Object.entries(dashboardData.group_stats).length > 0 ? (
            <div className="grid gap-6 md:grid-cols-2">
              {Object.entries(dashboardData.group_stats).map(([groupId, group]) => (
                <Card key={groupId} className="overflow-hidden">
                  <CardHeader className="bg-muted">
                    <CardTitle>{group.name || 'Unnamed Group'}</CardTitle>
                  </CardHeader>
                  <CardContent className="p-6 space-y-4">
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="bg-muted p-3 rounded-md">
                        <div className="text-2xl font-bold">{group.total_members}</div>
                        <div className="text-xs text-muted-foreground">Total Members</div>
                      </div>
                      <div className="bg-muted p-3 rounded-md">
                        <div className="text-2xl font-bold">{group.new_members_30d}</div>
                        <div className="text-xs text-muted-foreground">New This Month</div>
                      </div>
                      <div className="bg-muted p-3 rounded-md">
                        <div className="text-2xl font-bold">{group.active_members}</div>
                        <div className="text-xs text-muted-foreground">Active Members</div>
                      </div>
                    </div>

                    <div className="pt-4">
                      <h3 className="text-sm font-medium mb-2">Top Member Locations</h3>
                      <div className="space-y-2">
                        {Object.entries(group.location_distribution)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 3)
                          .map(([location, count], index) => (
                            <div key={index} className="flex justify-between items-center">
                              <span className="text-sm">{location}</span>
                              <span className="text-sm font-semibold">{count}</span>
                            </div>
                          ))}
                      </div>
                    </div>

                    <div className="pt-2">
                      <h3 className="text-sm font-medium mb-2">Member Engagement</h3>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div className="bg-green-500 h-2.5 rounded-full" 
                          style={{ width: `${(group.engagement_levels.high / group.total_members) * 100}%` }}>
                        </div>
                      </div>
                      <div className="flex justify-between text-xs mt-1">
                        <span>{group.engagement_levels.high} High</span>
                        <span>{group.engagement_levels.medium} Medium</span>
                        <span>{group.engagement_levels.low} Low</span>
                      </div>
                    </div>
                    
                    <Button variant="outline" className="w-full mt-4">View Detailed Analytics</Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center p-8">
              <p className="text-muted-foreground mb-4">You haven't created any groups yet.</p>
              <Button>Create Your First Group</Button>
            </div>
          )}
        </TabsContent>

        {/* Events Tab */}
        <TabsContent value="events" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Your Events</h2>
            <Button>Create New Event</Button>
          </div>

          {Object.entries(dashboardData.event_stats).length > 0 ? (
            <div className="grid gap-6 md:grid-cols-2">
              {Object.entries(dashboardData.event_stats).map(([eventId, event]) => (
                <Card key={eventId} className="overflow-hidden">
                  <CardHeader className="bg-muted">
                    <CardTitle>{event.title || 'Unnamed Event'}</CardTitle>
                  </CardHeader>
                  <CardContent className="p-6 space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-center">
                      <div className="bg-muted p-3 rounded-md">
                        <div className="text-2xl font-bold">{event.total_registered}</div>
                        <div className="text-xs text-muted-foreground">Total Registered</div>
                      </div>
                      <div className="bg-muted p-3 rounded-md">
                        <div className="text-2xl font-bold">{event.total_attended}</div>
                        <div className="text-xs text-muted-foreground">Total Attended</div>
                      </div>
                    </div>
                    
                    <div className="pt-2">
                      <h3 className="text-sm font-medium mb-2">Attendance Rate</h3>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div className="bg-blue-500 h-2.5 rounded-full" 
                          style={{ width: `${(event.total_attended / Math.max(event.total_registered, 1)) * 100}%` }}>
                        </div>
                      </div>
                      <div className="flex justify-between text-xs mt-1">
                        <span>{((event.total_attended / Math.max(event.total_registered, 1)) * 100).toFixed(0)}% attended</span>
                        <span>{event.total_attended}/{event.total_registered}</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-center mt-4">
                      <div className="p-3 rounded-md border">
                        <div className="text-lg font-bold">{event.new_attendees}</div>
                        <div className="text-xs text-muted-foreground">New Attendees</div>
                      </div>
                      <div className="p-3 rounded-md border">
                        <div className="text-lg font-bold">{event.returning_attendees}</div>
                        <div className="text-xs text-muted-foreground">Returning</div>
                      </div>
                    </div>

                    <div className="pt-4">
                      <h3 className="text-sm font-medium mb-2">Industry Breakdown</h3>
                      <div className="h-40">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={objectToChartData(event.industry_distribution)}
                              cx="50%"
                              cy="50%"
                              innerRadius={30}
                              outerRadius={50}
                              fill="#8884d8"
                              dataKey="value"
                            >
                              {objectToChartData(event.industry_distribution).map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <Tooltip />
                            <Legend />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    
                    <Button variant="outline" className="w-full mt-4">View Attendee Details</Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center p-8">
              <p className="text-muted-foreground mb-4">You haven't created any events yet.</p>
              <Button>Create Your First Event</Button>
            </div>
          )}
        </TabsContent>
        
        {/* Engagement Tab */}
        <TabsContent value="engagement" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Member Engagement Levels</CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={engagementData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      fill="#8884d8"
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, percent }: PieLabelRenderProps) => {
                        const percentage = percent !== undefined ? (percent * 100).toFixed(0) : 'N/A';
                        return `${name}: ${percentage}%`;
                      }}
                    >
                      {engagementData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={
                          index === 0 ? '#4ade80' : 
                          index === 1 ? '#facc15' : 
                          '#f87171'
                        } />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Growth Metrics</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="bg-muted rounded-md p-4">
                  <h3 className="text-sm font-medium mb-2">New Members This Period</h3>
                  <div className="text-2xl font-bold">
                    {Object.values(dashboardData.group_stats).reduce((sum, group) => sum + group.new_members_30d, 0)}
                  </div>
                  <div className="flex items-center mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div className="bg-green-500 h-1.5 rounded-full" 
                        style={{ width: `${dashboardData.member_growth_rate}%` }}>
                      </div>
                    </div>
                    <span className="text-xs ml-2 text-green-500">{dashboardData.member_growth_rate}%</span>
                  </div>
                </div>
                
                <div className="bg-muted rounded-md p-4">
                  <h3 className="text-sm font-medium mb-2">Engagement Rate</h3>
                  <div className="text-2xl font-bold">{dashboardData.engagement_rate}%</div>
                  <p className="text-xs text-muted-foreground">
                    Of your total membership, {dashboardData.engagement_rate}% are actively engaging with your content.
                  </p>
                </div>
                
                <div className="bg-muted rounded-md p-4">
                  <h3 className="text-sm font-medium mb-2">Retention Rate</h3>
                  <div className="text-2xl font-bold">{dashboardData.retention_rate}%</div>
                  <p className="text-xs text-muted-foreground">
                    {dashboardData.retention_rate}% of members continue to engage with your content over time.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>Recommendations to Improve Engagement</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border rounded-md">
                  <h3 className="font-medium">Create targeted content for your top industry segments</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Your members are primarily from: {Object.keys(dashboardData.demographic_summary.top_industries || {}).slice(0, 3).join(', ')}
                  </p>
                </div>
                
                <div className="p-4 border rounded-md">
                  <h3 className="font-medium">Consider hosting virtual events</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    You have members from multiple locations, including remote participants
                  </p>
                </div>
                
                <div className="p-4 border rounded-md">
                  <h3 className="font-medium">Encourage skill sharing within your groups</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Your members have expertise in: {dashboardData.demographic_summary.top_skills?.slice(0, 3).join(', ')}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 