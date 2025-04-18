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
import { Loader2, Users, UserCheck, Lightbulb, Sparkles, Calendar, Tv } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import toast from 'react-hot-toast';

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

// --- Define types for recommendations from API --- 
// Copied from profile page - adjust if API response differs
type ApiRecommendation = {
  id: string;
  name?: string; // For profiles/groups
  title?: string; // For events/profiles
  avatar_url?: string; // For profiles
  reason?: string; // Optional: Why was this recommended?
  score?: number; // Optional: Recommendation score
  // Add other relevant fields from your API response
  // Event-specific fields if needed from API
  date?: string;
  location?: string;
  // Group-specific fields if needed from API
  members?: number;
  category?: string;
  // Added fields for better recommendations display
  first_name?: string;
  last_name?: string;
  headline?: string;
};

// Keep old structure for UI compatibility for now
interface OldConnection {
  id: string; // Use string ID from API
  name: string;
  title: string;
  avatar: string | null;
}

interface OldGroup {
  id: string; // Use string ID from API
  name: string;
  members: number;
  category: string;
}

interface OldEvent {
  id: string; // Use string ID from API
  name: string;
  date: string;
  location: string;
}

interface OldRecommendations {
  connections: OldConnection[];
  groups: OldGroup[];
  events: OldEvent[];
  skills: string[]; // Keep skills as is for now
}
// --- End Types --- 

export default function DashboardPage() {
  const [helpQuestion, setHelpQuestion] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);
  
  // --- State for API Recommendations ---
  const [profileRecommendations, setProfileRecommendations] = useState<ApiRecommendation[]>([]);
  const [groupRecommendations, setGroupRecommendations] = useState<ApiRecommendation[]>([]);
  const [eventRecommendations, setEventRecommendations] = useState<ApiRecommendation[]>([]);
  const [recommendationLoading, setRecommendationLoading] = useState(true); // Start loading true
  // --- End State for API Recommendations ---
  
  // Existing state for UI - will be populated from new state
  const [uiRecommendations, setUiRecommendations] = useState<OldRecommendations>({
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
      // TODO: Reinstate when system_settings table exists and is populated
      // const { data } = await supabase
      //   .from('system_settings')
      //   .select('value')
      //   .eq('key', 'signup_help_question')
      //   .single();
      
      // if (data) {
      //   setHelpQuestion(data.value);
      // }
      setHelpQuestion('What are you hoping to achieve on Networkli?'); // Default question
    };

    fetchHelpQuestion();
  }, [supabase]);

  // Fetch recommendations
  useEffect(() => {
    const fetchApiRecommendations = async () => {
      if (!user) return;
      
      setRecommendationLoading(true);
      console.log(`[Dashboard] Fetching recommendations for logged-in user: ${user.id}`);

      let accessToken: string | null = null;
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.access_token) {
          accessToken = session.access_token;
        } else {
          throw new Error("No access token found");
        }
      } catch (error) {
        console.error("[Dashboard Recs] Error getting session token:", error);
        toast.error("Authentication error loading recommendations.");
        setRecommendationLoading(false);
        return;
      }
      
      const headers: HeadersInit = {
        'Authorization': `Bearer ${accessToken}`
      };

      try {
        const recTypes: ('profile' | 'group' | 'event')[] = ['profile', 'group', 'event'];
        const promises = recTypes.map(type => 
          fetch(`/api/recommendations?profile_id=${user.id}&type=${type}&limit=3`, { headers })
            .then(async res => {
               if (!res.ok) {
                   const errorData = await res.json().catch(() => ({})); // Try to parse error
                   console.error(`[Dashboard Recs] API Error (${res.status}) fetching ${type}:`, errorData.error || res.statusText);
                   return { type, data: { recommendations: [] } }; // Return empty on error
               }
               return res.json().then(data => ({ type, data }));
            })
        );
        
        const results = await Promise.allSettled(promises);
        
        const fetchedRecommendations: { [key: string]: ApiRecommendation[] } = {
          profile: [],
          group: [],
          event: []
        };

        results.forEach(result => {
          if (result.status === 'fulfilled' && result.value) {
            const { type, data } = result.value;
            // Basic validation of received data structure
            if (data && Array.isArray(data.recommendations)) {
                fetchedRecommendations[type] = data.recommendations;
            } else {
                 console.warn(`[Dashboard Recs] Unexpected data structure for type ${type}:`, data);
            }
          }
        });

        console.log("[Dashboard Recs] Fetched API recommendations:", fetchedRecommendations);
        
        // Set the new state
        setProfileRecommendations(fetchedRecommendations.profile);
        setGroupRecommendations(fetchedRecommendations.group);
        setEventRecommendations(fetchedRecommendations.event);

        // --- Adapt API data to old UI state structure ---
        const adaptedConnections: OldConnection[] = fetchedRecommendations.profile.map(p => ({
          id: p.id,
          name: p.name || p.title || 'User',
          title: p.title || 'Professional',
          avatar: p.avatar_url || null
        }));

        const adaptedGroups: OldGroup[] = fetchedRecommendations.group.map(g => ({
          id: g.id,
          name: g.name || 'Group',
          members: g.members || 0, // Assuming API provides members count
          category: g.category || 'General' // Assuming API provides category
        }));

        const adaptedEvents: OldEvent[] = fetchedRecommendations.event.map(e => ({
          id: e.id,
          name: e.title || 'Event',
          date: e.date || new Date().toISOString(), // Assuming API provides date
          location: e.location || 'TBD' // Assuming API provides location
        }));
        // --- End Adaptation ---

        // Fetch skills separately for now (or integrate into API later)
        let skillRecs: string[] = [];
        try {
          const { data: skills, error: skillsError } = await supabase
            .from('skills') // Or perhaps 'user_skills' linked to user.id?
            .select('name')
            .limit(5);
          if (skillsError) throw skillsError;
          skillRecs = skills?.map(skill => skill.name) || [];
        } catch(skillErr) {
            console.error("[Dashboard Recs] Error fetching skills:", skillErr);
        }

        // Set the UI-compatible state
        setUiRecommendations({
          connections: adaptedConnections,
          groups: adaptedGroups,
          events: adaptedEvents,
          skills: skillRecs // Keep using fetched skills
        });
        
      } catch (err) {
        console.error('[Dashboard Recs] Error fetching API recommendations:', err);
        toast.error('Could not load recommendations');
        // Optionally set fallback data for uiRecommendations here
      } finally {
        setRecommendationLoading(false);
      }
    };

    fetchApiRecommendations();
  }, [user, supabase]);

  const handleUpdateProfile = async () => {
    setIsUpdating(true);
    try {
      // TODO: Reinstate when system_settings table exists and is populated
      // const { error } = await supabase
      //   .from('system_settings')
      //   .upsert({
      //     key: 'signup_help_question',
      //     value: helpQuestion
      //   });

      // if (error) throw error;
      console.log("Simulating update for help question:", helpQuestion); // Simulate update
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate delay
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
    
    // --- Define Recommendation Section UI --- 
    const RecommendationSection = () => (
       !recommendationLoading && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Suggestions For You</h2>
          <div className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {/* People Recommendations */}
            {profileRecommendations.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center">
                    <Users className="h-5 w-5 mr-2 text-blue-600"/> Suggested Connections
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3">
                    {profileRecommendations.map(rec => (
                      <li key={rec.id} className="flex items-center space-x-3 text-sm">
                         <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                           {rec.avatar_url ? (
                              <img src={rec.avatar_url} alt={`${rec.first_name || ''} ${rec.last_name || ''}`} className="h-full w-full rounded-full object-cover"/>
                           ) : <UserCheck className="h-4 w-4 text-gray-500"/>}
                         </div>
                        <div>
                          <Link href={`/dashboard/profile/${rec.id}?from=recommendation`} className="font-medium text-gray-800 hover:text-blue-600">
                             {rec.first_name || ''} {rec.last_name || ''}{rec.headline ? ` - ${rec.headline}` : ''}
                          </Link>
                          {rec.reason && <p className="text-xs text-gray-500 italic">({rec.reason})</p>}
                        </div>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}
            {/* Groups Recommendations */}
            {groupRecommendations.length > 0 && (
              <Card>
                 <CardHeader>
                    <CardTitle className="text-lg flex items-center">
                      <Lightbulb className="h-5 w-5 mr-2 text-purple-600"/> Recommended Groups
                    </CardTitle>
                 </CardHeader>
                 <CardContent>
                   <ul className="space-y-3">
                    {groupRecommendations.map(rec => (
                      <li key={rec.id} className="flex items-center space-x-3 text-sm">
                         <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                           <Sparkles className="h-4 w-4 text-gray-500"/>
                         </div>
                        <div>
                          {/* TODO: Update href to actual group page */}
                          <Link href={`#group-${rec.id}`} className="font-medium text-gray-800 hover:text-purple-600">
                             {rec.name || 'View Group'}
                          </Link>
                          {rec.reason && <p className="text-xs text-gray-500 italic">({rec.reason})</p>}
                        </div>
                      </li>
                    ))}
                  </ul>
                 </CardContent>
              </Card>
            )}
            {/* Events Recommendations */}
            {eventRecommendations.length > 0 && (
              <Card>
                 <CardHeader>
                    <CardTitle className="text-lg flex items-center">
                      <Calendar className="h-5 w-5 mr-2 text-pink-600"/> Recommended Events
                    </CardTitle>
                 </CardHeader>
                 <CardContent>
                    <ul className="space-y-3">
                    {eventRecommendations.map(rec => (
                      <li key={rec.id} className="flex items-center space-x-3 text-sm">
                         <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">
                           <Tv className="h-4 w-4 text-gray-500"/>
                         </div>
                        <div>
                           {/* TODO: Update href to actual event page */}
                           <Link href={`#event-${rec.id}`} className="font-medium text-gray-800 hover:text-pink-600">
                             {rec.title || 'View Event'}
                           </Link>
                           {rec.reason && <p className="text-xs text-gray-500 italic">({rec.reason})</p>}
                           {rec.date && <p className="text-xs text-gray-500">{new Date(rec.date).toLocaleDateString()}</p>} 
                        </div>
                      </li>
                    ))}
                  </ul>
                 </CardContent>
              </Card>
            )}
          </div>
        </div>
       )
    );
    // --- End Recommendation Section UI ---

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
            <RecommendationSection />
          </>
        )
      case 'admin':
        return (
          <>
            <AdminDashboard user={user} />
            <RecommendationSection />
          </>
        )
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
              {recommendationLoading ? (
                 <div className="flex justify-center items-center p-6 col-span-1 lg:col-span-3">
                   <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
                   <span className="ml-2">Loading suggestions...</span>
                 </div>
             ) : (
                <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
                   {/* People Recommendations */} 
                  {profileRecommendations.length > 0 && (
                    <Card>
                       <CardHeader> <CardTitle className="text-lg flex items-center"><Users className="h-5 w-5 mr-2 text-blue-600"/> Suggested Connections</CardTitle></CardHeader>
                       <CardContent> <ul className="space-y-3"> {profileRecommendations.map(rec => (<li key={rec.id} className="flex items-center space-x-3 text-sm"> <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center"> {rec.avatar_url ? (<img src={rec.avatar_url} alt={`${rec.first_name || ''} ${rec.last_name || ''}`} className="h-full w-full rounded-full object-cover"/>) : <UserCheck className="h-4 w-4 text-gray-500"/>} </div> <div> <Link href={`/dashboard/profile/${rec.id}?from=recommendation`} className="font-medium text-gray-800 hover:text-blue-600"> {rec.first_name || ''} {rec.last_name || ''} {rec.headline ? ` - ${rec.headline}` : ''} </Link> {rec.reason && <p className="text-xs text-gray-500 italic">({rec.reason})</p>} </div> </li>))} </ul> </CardContent>
                    </Card>
                  )}
                   {/* Groups Recommendations */} 
                   {groupRecommendations.length > 0 && (
                     <Card>
                       <CardHeader><CardTitle className="text-lg flex items-center"><Lightbulb className="h-5 w-5 mr-2 text-purple-600"/> Recommended Groups</CardTitle></CardHeader>
                       <CardContent><ul className="space-y-3">{groupRecommendations.map(rec => (<li key={rec.id} className="flex items-center space-x-3 text-sm"><div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center"><Sparkles className="h-4 w-4 text-gray-500"/></div><div><Link href={`#group-${rec.id}`} className="font-medium text-gray-800 hover:text-purple-600">{rec.name || 'View Group'}</Link>{rec.reason && <p className="text-xs text-gray-500 italic">({rec.reason})</p>}</div></li>))}</ul></CardContent>
                     </Card>
                   )}
                    {/* Events Recommendations */} 
                    {eventRecommendations.length > 0 && (
                       <Card>
                          <CardHeader><CardTitle className="text-lg flex items-center"><Calendar className="h-5 w-5 mr-2 text-pink-600"/> Recommended Events</CardTitle></CardHeader>
                          <CardContent><ul className="space-y-3">{eventRecommendations.map(rec => (<li key={rec.id} className="flex items-center space-x-3 text-sm"><div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center"><Tv className="h-4 w-4 text-gray-500"/></div><div><Link href={`#event-${rec.id}`} className="font-medium text-gray-800 hover:text-pink-600">{rec.title || 'View Event'}</Link>{rec.reason && <p className="text-xs text-gray-500 italic">({rec.reason})</p>}{rec.date && <p className="text-xs text-gray-500">{new Date(rec.date).toLocaleDateString()}</p>}</div></li>))}</ul></CardContent>
                       </Card>
                    )}
                 </div>
             )}
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
            <RecommendationSection />
          </div>
        )
    }
  }

  return (
    <div className="container mx-auto">
      <div className="mt-6">
         {renderDashboardContent()}
      </div>
    </div>
  );
} 