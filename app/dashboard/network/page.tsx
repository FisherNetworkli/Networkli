'use client';

import { useState, useEffect } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';
import Link from 'next/link';
import { Button } from '@/components/ui/button';

interface Connection {
  id: number;
  name: string;
  title: string;
  company: string;
  avatar: string | null;
  mutualConnections: number;
  isConnected: boolean;
}

export default function NetworkPage() {
  const [user, setUser] = useState<User | null>(null);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [pendingConnections, setPendingConnections] = useState<Connection[]>([]);
  const [suggestedConnections, setSuggestedConnections] = useState<Connection[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const supabase = createClientComponentClient();

  useEffect(() => {
    const getUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setUser(session.user);
      }
    };

    getUser();
  }, [supabase.auth]);

  // Fetch mock connections data
  useEffect(() => {
    const fetchConnections = async () => {
      setLoading(true);
      
      // In a real app, this would be an API call to your database
      // For now, using mock data
      const mockConnections: Connection[] = [
        {
          id: 1,
          name: 'Sarah Chen',
          title: 'UX Designer',
          company: 'Design Co.',
          avatar: null,
          mutualConnections: 12,
          isConnected: true
        },
        {
          id: 2,
          name: 'Michael Park',
          title: 'Product Manager',
          company: 'Tech Solutions Inc.',
          avatar: null,
          mutualConnections: 8,
          isConnected: true
        },
        {
          id: 3,
          name: 'Jessica Wong',
          title: 'Marketing Director',
          company: 'Growth Marketing',
          avatar: null,
          mutualConnections: 5,
          isConnected: true
        },
        {
          id: 4,
          name: 'David Rodriguez',
          title: 'Software Engineer',
          company: 'Code Ventures',
          avatar: null,
          mutualConnections: 7,
          isConnected: true
        }
      ];

      const mockPendingConnections: Connection[] = [
        {
          id: 5,
          name: 'Emily Johnson',
          title: 'Content Strategist',
          company: 'Media Group',
          avatar: null,
          mutualConnections: 3,
          isConnected: false
        },
        {
          id: 6,
          name: 'Alex Thompson',
          title: 'Data Scientist',
          company: 'Analytics Pro',
          avatar: null,
          mutualConnections: 2,
          isConnected: false
        }
      ];

      const mockSuggestedConnections: Connection[] = [
        {
          id: 7,
          name: 'Ryan Garcia',
          title: 'Frontend Developer',
          company: 'Web Solutions',
          avatar: null,
          mutualConnections: 10,
          isConnected: false
        },
        {
          id: 8,
          name: 'Laura Kim',
          title: 'Product Designer',
          company: 'Design Studio',
          avatar: null,
          mutualConnections: 6,
          isConnected: false
        },
        {
          id: 9,
          name: 'James Wilson',
          title: 'Sales Director',
          company: 'Revenue Inc.',
          avatar: null,
          mutualConnections: 4,
          isConnected: false
        },
        {
          id: 10,
          name: 'Sophia Martinez',
          title: 'HR Manager',
          company: 'People First',
          avatar: null,
          mutualConnections: 7,
          isConnected: false
        }
      ];
      
      setConnections(mockConnections);
      setPendingConnections(mockPendingConnections);
      setSuggestedConnections(mockSuggestedConnections);
      
      setLoading(false);
    };

    fetchConnections();
  }, []);

  // Filter connections based on search
  const filteredConnections = connections.filter(connection =>
    connection.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    connection.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    connection.company.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleConnect = (connectionId: number) => {
    // Update suggested connections
    setSuggestedConnections(prev => 
      prev.map(connection => 
        connection.id === connectionId 
          ? { ...connection, isConnected: true }
          : connection
      )
    );
    
    // In a real app, you would make an API call to update the connection status
  };

  const handleAccept = (connectionId: number) => {
    // Move from pending to connections
    const connectionToAccept = pendingConnections.find(c => c.id === connectionId);
    if (connectionToAccept) {
      setConnections(prev => [...prev, { ...connectionToAccept, isConnected: true }]);
      setPendingConnections(prev => prev.filter(c => c.id !== connectionId));
    }
    
    // In a real app, you would make an API call to update the connection status
  };

  const handleIgnore = (connectionId: number) => {
    // Remove from pending
    setPendingConnections(prev => prev.filter(c => c.id !== connectionId));
    
    // In a real app, you would make an API call to update the connection status
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">My Network</h1>
        <p className="text-muted-foreground mt-2">
          Manage your professional connections and discover new opportunities.
        </p>
      </div>

      {/* New Swipe Feature Promotion */}
      <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg shadow-md overflow-hidden">
        <div className="p-6 md:p-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="text-white">
            <div className="inline-block px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs font-semibold mb-3">
              NEW FEATURE
            </div>
            <h2 className="text-xl md:text-2xl font-bold mb-2">Swipe Match</h2>
            <p className="mb-4 max-w-md">
              Discover potential connections through our new swipe interface. 
              See compatibility scores and swipe right to connect with professionals who share your interests.
            </p>
            <Link href="/dashboard/network/swipe">
              <Button variant="secondary" className="font-medium">
                Try Swipe Match
                <svg className="ml-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
                </svg>
              </Button>
            </Link>
          </div>
          <div className="shrink-0 w-36 h-36 md:w-44 md:h-44 bg-white bg-opacity-10 rounded-full flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
            </svg>
          </div>
        </div>
      </div>

      {/* Network Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border shadow-sm text-center">
          <p className="text-3xl font-bold text-gray-800">{connections.length}</p>
          <p className="text-sm text-gray-500">Connections</p>
        </div>
        <div className="bg-white p-4 rounded-lg border shadow-sm text-center">
          <p className="text-3xl font-bold text-gray-800">{pendingConnections.length}</p>
          <p className="text-sm text-gray-500">Pending</p>
        </div>
        <div className="bg-white p-4 rounded-lg border shadow-sm text-center">
          <p className="text-3xl font-bold text-gray-800">42</p>
          <p className="text-sm text-gray-500">Profile Views</p>
        </div>
        <div className="bg-white p-4 rounded-lg border shadow-sm text-center">
          <p className="text-3xl font-bold text-gray-800">18</p>
          <p className="text-sm text-gray-500">Search Appearances</p>
        </div>
      </div>

      {/* Pending Connections */}
      {pendingConnections.length > 0 && (
        <div className="bg-white p-6 rounded-lg border shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Pending Invitations</h2>
          <div className="space-y-4">
            {pendingConnections.map(connection => (
              <div key={connection.id} className="flex flex-col sm:flex-row sm:items-center justify-between p-4 border rounded-lg bg-gray-50">
                <div className="flex items-center mb-4 sm:mb-0">
                  <div className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center text-gray-500 mr-4">
                    {connection.avatar ? (
                      <img src={connection.avatar} alt={connection.name} className="w-12 h-12 rounded-full" />
                    ) : (
                      connection.name.charAt(0)
                    )}
                  </div>
                  <div>
                    <p className="font-medium">{connection.name}</p>
                    <p className="text-sm text-gray-500">{connection.title} at {connection.company}</p>
                    <p className="text-xs text-gray-400 mt-1">{connection.mutualConnections} mutual connections</p>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button 
                    onClick={() => handleIgnore(connection.id)}
                    className="px-4 py-2 border rounded-md text-sm font-medium text-gray-700 hover:bg-gray-100"
                  >
                    Ignore
                  </button>
                  <button 
                    onClick={() => handleAccept(connection.id)}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
                  >
                    Accept
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* My Connections */}
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
          <h2 className="text-lg font-semibold mb-2 sm:mb-0">My Connections</h2>
          <div className="w-full sm:w-64">
            <input
              type="text"
              placeholder="Search connections..."
              className="w-full px-3 py-2 border rounded-md"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>
        
        {filteredConnections.length === 0 ? (
          <p className="text-center py-8 text-gray-500">No connections found matching your search.</p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {filteredConnections.map(connection => (
              <div key={connection.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center">
                  <div className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center text-gray-500 mr-4">
                    {connection.avatar ? (
                      <img src={connection.avatar} alt={connection.name} className="w-12 h-12 rounded-full" />
                    ) : (
                      connection.name.charAt(0)
                    )}
                  </div>
                  <div>
                    <p className="font-medium">{connection.name}</p>
                    <p className="text-sm text-gray-500">{connection.title}</p>
                    <p className="text-xs text-gray-400">{connection.company}</p>
                  </div>
                </div>
                <div className="mt-4 flex justify-between">
                  <button className="text-sm text-blue-600 hover:text-blue-800">View Profile</button>
                  <button className="text-sm text-gray-600 hover:text-gray-800">Message</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* People You May Know */}
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <h2 className="text-lg font-semibold mb-4">People You May Know</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {suggestedConnections.map(connection => (
            <div key={connection.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="flex items-center">
                <div className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center text-gray-500 mr-4">
                  {connection.avatar ? (
                    <img src={connection.avatar} alt={connection.name} className="w-12 h-12 rounded-full" />
                  ) : (
                    connection.name.charAt(0)
                  )}
                </div>
                <div>
                  <p className="font-medium">{connection.name}</p>
                  <p className="text-sm text-gray-500">{connection.title}</p>
                  <p className="text-xs text-gray-400">{connection.company}</p>
                </div>
              </div>
              <div className="mt-2">
                <p className="text-xs text-gray-500">{connection.mutualConnections} mutual connections</p>
              </div>
              <div className="mt-4">
                <button 
                  onClick={() => handleConnect(connection.id)}
                  className={`w-full py-2 text-sm font-medium rounded-md ${
                    connection.isConnected
                      ? 'bg-gray-100 text-gray-800 cursor-default'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                  disabled={connection.isConnected}
                >
                  {connection.isConnected ? 'Invitation Sent' : 'Connect'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 