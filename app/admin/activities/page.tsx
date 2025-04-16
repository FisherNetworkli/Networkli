'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  HomeIcon, 
  MagnifyingGlassIcon,
  ArrowPathIcon, 
  UserIcon,
  CalendarIcon,
  DocumentTextIcon,
  EnvelopeIcon,
  QuestionMarkCircleIcon,
  PencilSquareIcon,
  UsersIcon,
  CogIcon,
  ChatBubbleLeftRightIcon
} from '@heroicons/react/24/outline';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

interface Activity {
  id: string;
  activity_type: 'login' | 'signup' | 'profile_update' | 'question_posted' | 'message_sent' | 'event_created' | 'event_joined' | 'settings_updated' | 'connection_made' | 'admin_action';
  user_id: string;
  user_name: string;
  user_email: string;
  details: string;
  target_id?: string;
  target_type?: string;
  created_at: string;
}

export default function ActivitiesPage() {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [filteredActivities, setFilteredActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [activityTypeFilter, setActivityTypeFilter] = useState<string[]>([]);
  const [timeframeFilter, setTimeframeFilter] = useState<'all' | 'today' | 'week' | 'month'>('all');
  const [pagination, setPagination] = useState({
    page: 1,
    pageSize: 15,
    total: 0
  });

  const supabase = createClientComponentClient();

  // Generate mock activity data
  const generateMockActivities = (): Activity[] => {
    const activityTypes: Activity['activity_type'][] = [
      'login', 'signup', 'profile_update', 'question_posted', 'message_sent',
      'event_created', 'event_joined', 'settings_updated', 'connection_made', 'admin_action'
    ];
    
    const userNames = [
      'John Doe',
      'Jane Smith',
      'Robert Johnson',
      'Emily Williams',
      'Michael Brown',
      'Admin User'
    ];
    
    const userEmails = [
      'john.doe@example.com',
      'jane.smith@example.com',
      'robert.johnson@example.com',
      'emily.williams@example.com',
      'michael.brown@example.com',
      'test.admin@networkli.com',
      'test.organizer@networkli.com',
      'test.user@networkli.com'
    ];

    const getActivityDetails = (type: Activity['activity_type'], userName: string): string => {
      switch (type) {
        case 'login':
          return `${userName} logged in`;
        case 'signup':
          return `${userName} created a new account`;
        case 'profile_update':
          return `${userName} updated their profile`;
        case 'question_posted':
          return `${userName} posted a new help question`;
        case 'message_sent':
          return `${userName} sent a message`;
        case 'event_created':
          return `${userName} created a new event`;
        case 'event_joined':
          return `${userName} joined an event`;
        case 'settings_updated':
          return `${userName} updated system settings`;
        case 'connection_made':
          return `${userName} connected with another user`;
        case 'admin_action':
          return `${userName} performed an administrative action`;
        default:
          return `${userName} performed an action`;
      }
    };

    const getTargetInfo = (type: Activity['activity_type']): { id?: string; type?: string } => {
      switch (type) {
        case 'question_posted':
          return { id: `question-${Math.floor(Math.random() * 1000) + 1}`, type: 'question' };
        case 'message_sent':
          return { id: `message-${Math.floor(Math.random() * 1000) + 1}`, type: 'message' };
        case 'event_created':
        case 'event_joined':
          return { id: `event-${Math.floor(Math.random() * 100) + 1}`, type: 'event' };
        case 'connection_made':
          return { id: `user-${Math.floor(Math.random() * 1000) + 1}`, type: 'user' };
        case 'admin_action':
          const targets = ['user', 'event', 'question', 'system'];
          const target = targets[Math.floor(Math.random() * targets.length)];
          return { id: `${target}-${Math.floor(Math.random() * 1000) + 1}`, type: target };
        default:
          return {};
      }
    };

    // Generate 200 mock activities
    const mockActivities: Activity[] = [];
    
    for (let i = 0; i < 200; i++) {
      const activityType = activityTypes[Math.floor(Math.random() * activityTypes.length)];
      const userIndex = Math.floor(Math.random() * userNames.length);
      const userName = userNames[userIndex];
      const userEmail = userEmails[Math.min(userIndex, userEmails.length - 1)];
      
      // Generate a date within the last 30 days, more recent activities more likely
      const daysAgo = Math.floor(Math.random() * Math.random() * 30);
      const date = new Date();
      date.setDate(date.getDate() - daysAgo);
      date.setHours(
        Math.floor(Math.random() * 24),
        Math.floor(Math.random() * 60),
        Math.floor(Math.random() * 60)
      );
      
      const details = getActivityDetails(activityType, userName);
      const { id: targetId, type: targetType } = getTargetInfo(activityType);
      
      mockActivities.push({
        id: `activity-${i + 1}`,
        activity_type: activityType,
        user_id: `user-${userIndex + 1}`,
        user_name: userName,
        user_email: userEmail,
        details,
        target_id: targetId,
        target_type: targetType,
        created_at: date.toISOString()
      });
    }
    
    // Sort by date (newest first)
    return mockActivities.sort((a, b) => 
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  };

  useEffect(() => {
    // In a real app, fetch activities from database
    const mockActivities = generateMockActivities();
    setActivities(mockActivities);
    setFilteredActivities(mockActivities);
    setPagination(prev => ({ ...prev, total: mockActivities.length }));
    setLoading(false);
  }, []);

  // Apply filters whenever they change
  useEffect(() => {
    let result = [...activities];
    
    // Apply search filter
    if (searchTerm.trim()) {
      const search = searchTerm.toLowerCase();
      result = result.filter(activity => 
        activity.user_name.toLowerCase().includes(search) ||
        activity.user_email.toLowerCase().includes(search) ||
        activity.details.toLowerCase().includes(search)
      );
    }
    
    // Apply activity type filter
    if (activityTypeFilter.length > 0) {
      result = result.filter(activity => 
        activityTypeFilter.includes(activity.activity_type)
      );
    }
    
    // Apply timeframe filter
    if (timeframeFilter !== 'all') {
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      
      result = result.filter(activity => {
        const activityDate = new Date(activity.created_at);
        
        switch (timeframeFilter) {
          case 'today':
            return activityDate >= today;
          case 'week': {
            const weekAgo = new Date(now);
            weekAgo.setDate(now.getDate() - 7);
            return activityDate >= weekAgo;
          }
          case 'month': {
            const monthAgo = new Date(now);
            monthAgo.setMonth(now.getMonth() - 1);
            return activityDate >= monthAgo;
          }
          default:
            return true;
        }
      });
    }
    
    setFilteredActivities(result);
    setPagination(prev => ({ ...prev, total: result.length, page: 1 }));
  }, [activities, searchTerm, activityTypeFilter, timeframeFilter]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  const toggleActivityTypeFilter = (type: Activity['activity_type']) => {
    setActivityTypeFilter(prev => 
      prev.includes(type)
        ? prev.filter(t => t !== type)
        : [...prev, type]
    );
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      
      // Check if it's today
      const today = new Date();
      const isToday = date.getDate() === today.getDate() &&
                      date.getMonth() === today.getMonth() &&
                      date.getFullYear() === today.getFullYear();
      
      if (isToday) {
        return new Intl.DateTimeFormat('en-US', {
          hour: '2-digit',
          minute: '2-digit'
        }).format(date);
      }
      
      // Check if it's this year
      const isThisYear = date.getFullYear() === today.getFullYear();
      
      if (isThisYear) {
        return new Intl.DateTimeFormat('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        }).format(date);
      }
      
      // It's a previous year
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch (error) {
      return 'Invalid date';
    }
  };

  const getActivityIcon = (type: Activity['activity_type']) => {
    switch (type) {
      case 'login':
        return <UserIcon className="h-5 w-5 text-blue-500" />;
      case 'signup':
        return <UserIcon className="h-5 w-5 text-green-500" />;
      case 'profile_update':
        return <PencilSquareIcon className="h-5 w-5 text-indigo-500" />;
      case 'question_posted':
        return <QuestionMarkCircleIcon className="h-5 w-5 text-yellow-500" />;
      case 'message_sent':
        return <EnvelopeIcon className="h-5 w-5 text-purple-500" />;
      case 'event_created':
        return <CalendarIcon className="h-5 w-5 text-red-500" />;
      case 'event_joined':
        return <CalendarIcon className="h-5 w-5 text-orange-500" />;
      case 'settings_updated':
        return <CogIcon className="h-5 w-5 text-gray-500" />;
      case 'connection_made':
        return <UsersIcon className="h-5 w-5 text-teal-500" />;
      case 'admin_action':
        return <DocumentTextIcon className="h-5 w-5 text-red-600" />;
      default:
        return <ChatBubbleLeftRightIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getActivityTypeLabel = (type: Activity['activity_type']) => {
    switch (type) {
      case 'login': return 'Login';
      case 'signup': return 'Sign Up';
      case 'profile_update': return 'Profile Update';
      case 'question_posted': return 'Question Posted';
      case 'message_sent': return 'Message Sent';
      case 'event_created': return 'Event Created';
      case 'event_joined': return 'Event Joined';
      case 'settings_updated': return 'Settings Updated';
      case 'connection_made': return 'Connection Made';
      case 'admin_action': return 'Admin Action';
      default: return 'Unknown';
    }
  };

  const handleRefresh = () => {
    setLoading(true);
    setTimeout(() => {
      const mockActivities = generateMockActivities();
      setActivities(mockActivities);
      setFilteredActivities(mockActivities);
      setPagination(prev => ({ ...prev, total: mockActivities.length }));
      setLoading(false);
    }, 1000);
  };

  const paginatedActivities = filteredActivities.slice(
    (pagination.page - 1) * pagination.pageSize,
    pagination.page * pagination.pageSize
  );

  const totalPages = Math.ceil(pagination.total / pagination.pageSize);

  const handlePageChange = (newPage: number) => {
    setPagination(prev => ({
      ...prev,
      page: newPage
    }));
  };

  return (
    <div className="py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-4 md:mb-0">Recent Activities</h1>
          <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
            <Link
              href="/admin"
              className="inline-flex items-center justify-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue"
            >
              <HomeIcon className="h-4 w-4 mr-1" />
              Back to Dashboard
            </Link>
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue"
            >
              <ArrowPathIcon className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white shadow rounded-lg mb-6 p-4">
          <div className="flex flex-col space-y-4 md:flex-row md:space-y-0 md:space-x-4">
            {/* Search */}
            <div className="flex-1 relative rounded-md shadow-sm">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={searchTerm}
                onChange={handleSearchChange}
                className="focus:ring-connection-blue focus:border-connection-blue block w-full pl-10 sm:text-sm border-gray-300 rounded-md"
                placeholder="Search by user or activity details"
              />
            </div>
            
            {/* Timeframe Filter */}
            <div className="flex-shrink-0">
              <select
                value={timeframeFilter}
                onChange={(e) => setTimeframeFilter(e.target.value as any)}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-connection-blue focus:border-connection-blue sm:text-sm rounded-md"
              >
                <option value="all">All Time</option>
                <option value="today">Today</option>
                <option value="week">Last 7 Days</option>
                <option value="month">Last 30 Days</option>
              </select>
            </div>
          </div>
          
          {/* Activity Type Filter */}
          <div className="mt-4">
            <div className="text-sm font-medium text-gray-700 mb-2">Filter by activity type:</div>
            <div className="flex flex-wrap gap-2">
              {['login', 'signup', 'profile_update', 'question_posted', 'message_sent', 'event_created', 'event_joined', 'settings_updated', 'connection_made', 'admin_action'].map((type) => (
                <button
                  key={type}
                  onClick={() => toggleActivityTypeFilter(type as Activity['activity_type'])}
                  className={`inline-flex items-center px-2.5 py-1.5 rounded-full text-xs font-medium ${
                    activityTypeFilter.includes(type)
                      ? 'bg-connection-blue text-white'
                      : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                  }`}
                >
                  {getActivityTypeLabel(type as Activity['activity_type'])}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Activity List */}
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          {loading ? (
            <div className="py-16 flex justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-connection-blue"></div>
            </div>
          ) : paginatedActivities.length === 0 ? (
            <div className="py-16 text-center">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900">No activities found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Try adjusting your search or filter to find what you're looking for.
              </p>
              <div className="mt-6">
                <button
                  onClick={() => {
                    setSearchTerm('');
                    setActivityTypeFilter([]);
                    setTimeframeFilter('all');
                  }}
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-connection-blue hover:bg-connection-blue-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-connection-blue"
                >
                  Clear all filters
                </button>
              </div>
            </div>
          ) : (
            <div>
              <ul className="divide-y divide-gray-200">
                {paginatedActivities.map((activity) => (
                  <li key={activity.id} className="px-4 py-4 sm:px-6 hover:bg-gray-50">
                    <div className="flex items-center">
                      <div className="flex-shrink-0">
                        {getActivityIcon(activity.activity_type)}
                      </div>
                      <div className="min-w-0 flex-1 px-4">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-connection-blue truncate">
                            {activity.user_name}
                          </p>
                          <div className="ml-2 flex-shrink-0 flex">
                            <p className="text-sm text-gray-500">
                              {formatDate(activity.created_at)}
                            </p>
                          </div>
                        </div>
                        <div className="mt-1">
                          <p className="text-sm text-gray-900">
                            {activity.details}
                            {activity.target_id && (
                              <span className="ml-1 text-xs text-gray-500">
                                ({activity.target_type}: {activity.target_id})
                              </span>
                            )}
                          </p>
                        </div>
                        <div className="mt-1">
                          <p className="text-xs text-gray-500 truncate">
                            {activity.user_email}
                          </p>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
              
              {/* Pagination */}
              <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
                <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm text-gray-700">
                      Showing <span className="font-medium">{((pagination.page - 1) * pagination.pageSize) + 1}</span> to{' '}
                      <span className="font-medium">
                        {Math.min(pagination.page * pagination.pageSize, pagination.total)}
                      </span>{' '}
                      of <span className="font-medium">{pagination.total}</span> results
                    </p>
                  </div>
                  <div>
                    <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                      <button
                        onClick={() => handlePageChange(Math.max(1, pagination.page - 1))}
                        disabled={pagination.page === 1}
                        className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <span className="sr-only">Previous</span>
                        <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                          <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </button>
                      
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        // For showing numbered pages around the current page
                        let pageNum;
                        if (totalPages <= 5) {
                          pageNum = i + 1;
                        } else if (pagination.page <= 3) {
                          pageNum = i + 1;
                        } else if (pagination.page >= totalPages - 2) {
                          pageNum = totalPages - 4 + i;
                        } else {
                          pageNum = pagination.page - 2 + i;
                        }
                        
                        return (
                          <button
                            key={pageNum}
                            onClick={() => handlePageChange(pageNum)}
                            className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                              pageNum === pagination.page
                                ? 'z-10 bg-connection-blue text-white border-connection-blue'
                                : 'bg-white text-gray-500 hover:bg-gray-50 border-gray-300'
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                      
                      <button
                        onClick={() => handlePageChange(Math.min(totalPages, pagination.page + 1))}
                        disabled={pagination.page === totalPages}
                        className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <span className="sr-only">Next</span>
                        <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </nav>
                  </div>
                </div>
                
                <div className="flex sm:hidden">
                  <div className="flex-1 flex justify-between">
                    <button
                      onClick={() => handlePageChange(Math.max(1, pagination.page - 1))}
                      disabled={pagination.page === 1}
                      className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <button
                      onClick={() => handlePageChange(Math.min(totalPages, pagination.page + 1))}
                      disabled={pagination.page === totalPages}
                      className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 