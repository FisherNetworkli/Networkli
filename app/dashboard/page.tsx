'use client';

import { useState, useEffect } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { redirect } from 'next/navigation'
import { DashboardNav } from '@/app/dashboard/components/DashboardNav'
import { DashboardStats } from '@/app/dashboard/components/DashboardStats'
import { ProfileCompletion } from '@/app/dashboard/components/ProfileCompletion'
import { AdminDashboard } from '@/app/dashboard/components/AdminDashboard'
import { OrganizerDashboard } from '@/app/dashboard/components/OrganizerDashboard'
import { ProfileViews } from '@/app/dashboard/components/ProfileViews'
import { OrganizerProfileViews } from '@/app/dashboard/components/OrganizerProfileViews'
import { User } from '@supabase/supabase-js'
import Link from 'next/link';

// Define types for recommendations
interface Connection {
  id: number;
  name: string;
  title: string;
  avatar: string | null;
}

interface Group {
  id: number;
  name: string;
  members: number;
  category: string;
}

interface Event {
  id: number;
  name: string;
  date: string;
  location: string;
}

interface Recommendations {
  connections: Connection[];
  groups: Group[];
  events: Event[];
  skills: string[];
}

export default function DashboardPage() {
  const [helpQuestion, setHelpQuestion] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendations>({
    connections: [],
    groups: [],
    events: [],
    skills: []
  });
  const supabase = createClientComponentClient();

  useEffect(() => {
    const getSessionAndProfile = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        redirect('/signin');
        return;
      }

      setUser(session.user);

      // Fetch user profile to get role
      const { data: profile } = await supabase
        .from('profiles')
        .select('id, role')
        .eq('id', session.user.id)
        .single();

      if (profile) {
        setUserRole(profile.role);
      }
    };

    getSessionAndProfile();
  }, [supabase]);

  useEffect(() => {
    const fetchHelpQuestion = async () => {
      const { data } = await supabase
        .from('system_settings')
        .select('value')
        .eq('key', 'signup_help_question')
        .single();
      
      if (data) {
        setHelpQuestion(data.value);
      }
    };

    fetchHelpQuestion();
  }, [supabase]);

  // Fetch recommendations
  useEffect(() => {
    const fetchRecommendations = async () => {
      if (!user) return;
      
      try {
        // Fetch connections recommendations (people not yet connected)
        const { data: profiles, error: profilesError } = await supabase
          .from('profiles')
          .select('id, first_name, last_name, title, avatar_url')
          .neq('id', user.id)
          .limit(3);
          
        if (profilesError) throw profilesError;
        
        // Format connection recommendations
        const connectionRecs = profiles?.map(profile => ({
          id: profile.id,
          name: `${profile.first_name || ''} ${profile.last_name || ''}`.trim() || 'User',
          title: profile.title || 'Professional',
          avatar: profile.avatar_url
        })) || [];
        
        // Fetch group recommendations
        const { data: groups, error: groupsError } = await supabase
          .from('groups')
          .select('id, name, category, members:group_members(count)')
          .limit(3);
          
        if (groupsError) throw groupsError;
        
        // Format group recommendations
        const groupRecs = groups?.map(group => ({
          id: group.id,
          name: group.name || 'Group',
          members: Array.isArray(group.members) ? group.members.length : 0,
          category: group.category || 'General'
        })) || [];
        
        // Fetch upcoming events
        const today = new Date();
        const { data: events, error: eventsError } = await supabase
          .from('events')
          .select('id, title, date, location')
          .gte('date', today.toISOString())
          .order('date', { ascending: true })
          .limit(3);
          
        if (eventsError) throw eventsError;
        
        // Format event recommendations
        const eventRecs = events?.map(event => ({
          id: event.id,
          name: event.title || 'Event',
          date: event.date,
          location: event.location || 'TBD'
        })) || [];
        
        // Fetch skills recommendations from user_skills
        const { data: skills, error: skillsError } = await supabase
          .from('skills')
          .select('name')
          .limit(5);
          
        if (skillsError) throw skillsError;
        
        // Format skill recommendations
        const skillRecs = skills?.map(skill => skill.name) || [];
        
        // Use real data if available, fallback to mock data for any missing categories
        setRecommendations({
          connections: connectionRecs.length > 0 ? connectionRecs : [
            { id: 1, name: 'Sarah Chen', title: 'UX Designer', avatar: null },
            { id: 2, name: 'Michael Park', title: 'Product Manager', avatar: null },
            { id: 3, name: 'Jessica Wong', title: 'Marketing Director', avatar: null }
          ],
          groups: groupRecs.length > 0 ? groupRecs : [
            { id: 1, name: 'Tech Professionals', members: 1250, category: 'Technology' },
            { id: 2, name: 'UX/UI Designers', members: 850, category: 'Design' },
            { id: 3, name: 'Startup Founders', members: 620, category: 'Entrepreneurship' }
          ],
          events: eventRecs.length > 0 ? eventRecs : [
            { id: 1, name: 'Networking Mixer', date: '2024-07-15', location: 'San Francisco' },
            { id: 2, name: 'Tech Conference', date: '2024-07-22', location: 'Online' },
            { id: 3, name: 'Career Fair', date: '2024-07-30', location: 'New York' }
          ],
          skills: skillRecs.length > 0 ? skillRecs : [
            'Project Management',
            'Data Analysis',
            'Public Speaking',
            'UX Research',
            'Content Strategy'
          ]
        });
      } catch (error) {
        console.error('Error fetching recommendations:', error);
        // Fallback to mock data in case of error
        setRecommendations({
          connections: [
            { id: 1, name: 'Sarah Chen', title: 'UX Designer', avatar: null },
            { id: 2, name: 'Michael Park', title: 'Product Manager', avatar: null },
            { id: 3, name: 'Jessica Wong', title: 'Marketing Director', avatar: null }
          ],
          groups: [
            { id: 1, name: 'Tech Professionals', members: 1250, category: 'Technology' },
            { id: 2, name: 'UX/UI Designers', members: 850, category: 'Design' },
            { id: 3, name: 'Startup Founders', members: 620, category: 'Entrepreneurship' }
          ],
          events: [
            { id: 1, name: 'Networking Mixer', date: '2024-07-15', location: 'San Francisco' },
            { id: 2, name: 'Tech Conference', date: '2024-07-22', location: 'Online' },
            { id: 3, name: 'Career Fair', date: '2024-07-30', location: 'New York' }
          ],
          skills: [
            'Project Management',
            'Data Analysis',
            'Public Speaking',
            'UX Research',
            'Content Strategy'
          ]
        });
      }
    };

    fetchRecommendations();
  }, [user, supabase]);

  const handleUpdateProfile = async () => {
    setIsUpdating(true);
    try {
      const { error } = await supabase
        .from('system_settings')
        .upsert({
          key: 'signup_help_question',
          value: helpQuestion
        });

      if (error) throw error;
      // Could add a toast notification here
    } catch (error) {
      console.error('Error updating profile:', error);
      // Could add error toast here
    } finally {
      setIsUpdating(false);
    }
  };

  if (!user) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  // Render different dashboard based on user role
  const renderDashboardContent = () => {
    const isPremiumUser = userRole === 'premium';
    
    switch (userRole) {
      case 'organizer':
        return (
          <>
            <div className="space-y-8">
              {/* Welcome Section */}
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  Welcome back, {user.user_metadata?.full_name || user.email}
                </h1>
                <p className="text-muted-foreground mt-2">
                  Here's what's happening with your groups and events.
                </p>
              </div>

              {/* Stats Grid */}
              <DashboardStats />

              {/* Organizer Profile Views with enhanced analytics */}
              <OrganizerProfileViews user={user} />

              {/* Organizer-specific content */}
              <OrganizerDashboard user={user} />

              {/* Profile Completion */}
              <div className="grid gap-4 md:grid-cols-2">
                <ProfileCompletion user={user} />
              </div>
            </div>
          </>
        )
      case 'admin':
        return <AdminDashboard user={user} />
      case 'premium':
        return (
          <div className="space-y-8">
            {/* Welcome Section */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  Welcome back, {user.user_metadata?.full_name || user.email}
                </h1>
                <p className="text-muted-foreground mt-2">
                  Here's your personalized premium dashboard.
                </p>
              </div>
              <div className="flex items-center bg-yellow-100 text-yellow-800 px-4 py-2 rounded-lg">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
                Premium Member
              </div>
            </div>

            {/* Stats Grid */}
            <DashboardStats />
            
            {/* Profile Views with complete history for premium users */}
            <ProfileViews user={user} isPremium={true} />

            {/* Profile Completion */}
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <ProfileCompletion user={user} />
              
              {/* Quick Actions */}
              <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
                <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
                <div className="grid gap-3">
                  <Link href="/dashboard/network" className="flex items-center justify-between p-3 rounded-lg border hover:bg-gray-50 transition-colors">
                    <div>
                      <h4 className="font-medium">Find New Connections</h4>
                      <p className="text-xs text-muted-foreground">Discover people with similar interests</p>
                    </div>
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
                    </svg>
                  </Link>
                  <Link href="/dashboard/groups" className="flex items-center justify-between p-3 rounded-lg border hover:bg-gray-50 transition-colors">
                    <div>
                      <h4 className="font-medium">Browse Groups</h4>
                      <p className="text-xs text-muted-foreground">Find professional communities</p>
                    </div>
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
                    </svg>
                  </Link>
                  <Link href="/dashboard/events" className="flex items-center justify-between p-3 rounded-lg border hover:bg-gray-50 transition-colors">
                    <div>
                      <h4 className="font-medium">Upcoming Events</h4>
                      <p className="text-xs text-muted-foreground">View your calendar and networking events</p>
                    </div>
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
                    </svg>
                  </Link>
                </div>
              </div>
              
              {/* Premium Benefits */}
              <div className="rounded-lg border bg-gradient-to-br from-blue-50 to-indigo-50 text-card-foreground shadow-sm p-6">
                <h3 className="text-lg font-semibold mb-4">Premium Benefits</h3>
                <ul className="space-y-2">
                  <li className="flex items-start">
                    <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                    </svg>
                    <span className="text-sm">Advanced networking tools</span>
                  </li>
                  <li className="flex items-start">
                    <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                    </svg>
                    <span className="text-sm">Exclusive events and groups</span>
                  </li>
                  <li className="flex items-start">
                    <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                    </svg>
                    <span className="text-sm">Priority support</span>
                  </li>
                  <li className="flex items-start">
                    <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                    </svg>
                    <span className="text-sm">Personalized recommendations</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Recommendation Tabs - 3 column grid for premium users */}
            <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
              {/* People Recommendations */}
              <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">Recommended People</h3>
                  <Link href="/dashboard/recommended" className="text-sm text-blue-600 hover:text-blue-800">View All</Link>
                </div>
                <div className="space-y-4">
                  {recommendations.connections.map(person => (
                    <div key={person.id} className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center text-gray-500">
                          {person.avatar ? (
                            <img src={person.avatar} alt={person.name} className="w-10 h-10 rounded-full" />
                          ) : (
                            person.name.charAt(0)
                          )}
                        </div>
                        <div>
                          <p className="font-medium text-sm">{person.name}</p>
                          <p className="text-xs text-muted-foreground">{person.title}</p>
                        </div>
                      </div>
                      <button className="text-sm text-blue-600 hover:text-blue-800 font-medium">Connect</button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Groups Recommendations */}
              <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">Recommended Groups</h3>
                  <Link href="/dashboard/groups" className="text-sm text-blue-600 hover:text-blue-800">View All</Link>
                </div>
                <div className="space-y-4">
                  {recommendations.groups.map(group => (
                    <div key={group.id} className="p-3 border rounded-lg hover:bg-gray-50">
                      <h4 className="font-medium text-sm">{group.name}</h4>
                      <p className="text-xs text-muted-foreground mb-2">{group.category} • {group.members} members</p>
                      <button className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium">Join Group</button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Events Recommendations */}
              <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">Upcoming Events</h3>
                  <Link href="/dashboard/events" className="text-sm text-blue-600 hover:text-blue-800">View All</Link>
                </div>
                <div className="space-y-4">
                  {recommendations.events.map(event => (
                    <div key={event.id} className="p-3 border rounded-lg hover:bg-gray-50">
                      <h4 className="font-medium text-sm">{event.name}</h4>
                      <p className="text-xs text-muted-foreground mb-2">
                        {new Date(event.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} • {event.location}
                      </p>
                      <button className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium">RSVP</button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )
      default:
        return (
          <div className="space-y-8">
            {/* Welcome Section */}
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Welcome back, {user.user_metadata?.full_name || user.email}
              </h1>
              <p className="text-muted-foreground mt-2">
                Here's what's happening in your network.
              </p>
            </div>

            {/* Stats Grid */}
            <DashboardStats />

            {/* Profile Views with feature gating for non-premium users */}
            <ProfileViews user={user} isPremium={false} />

            {/* Help Question - kept for both versions */}
            <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
              <h3 className="text-lg font-semibold mb-4">Customize Your Profile</h3>
              <div className="space-y-4">
                <div>
                  <label htmlFor="helpQuestion" className="block text-sm font-medium text-gray-700">
                    What do you need help with?
                  </label>
                  <div className="mt-1">
                    <textarea
                      id="helpQuestion"
                      name="helpQuestion"
                      rows={3}
                      className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md"
                      placeholder="Describe what kind of help or connections you're looking for..."
                      value={helpQuestion}
                      onChange={(e) => setHelpQuestion(e.target.value)}
                    ></textarea>
                  </div>
                  <p className="mt-2 text-sm text-gray-500">
                    This helps us match you with the right connections and opportunities.
                  </p>
                </div>
                <button 
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  onClick={handleUpdateProfile}
                  disabled={isUpdating}
                >
                  {isUpdating ? 'Updating...' : 'Update Profile'}
                </button>
              </div>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="container mx-auto">
      {renderDashboardContent()}
    </div>
  );
} 