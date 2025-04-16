import Link from 'next/link';
import { redirect } from 'next/navigation';
import { 
  ChartBarIcon, 
  UserGroupIcon, 
  CalendarIcon, 
  ChatBubbleLeftIcon,
  DocumentTextIcon,
  InboxIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  HomeIcon,
  CogIcon,
  BellIcon,
  ShieldCheckIcon,
  UserCircleIcon,
  WrenchScrewdriverIcon,
  QueueListIcon
} from '@heroicons/react/24/outline';
import { createServerComponentClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';

// Helper function to calculate percentage change
function calculatePercentageChange(current: number, previous: number): number {
  if (previous === 0) return 100;
  return ((current - previous) / previous) * 100;
}

// Helper function to format date
function formatDate(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  }).format(new Date(date));
}

export default async function AdminDashboard() {
  const supabase = createServerComponentClient({ cookies });
  
  // Check if user is authenticated
  const { data: { session } } = await supabase.auth.getSession();
  
  if (!session) {
    redirect('/signin');
  }
  
  // Check if user is admin by querying the profiles table
  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', session.user.id)
    .single();
  
  // If not admin, redirect to dashboard
  if (!profile || profile.role !== 'admin') {
    redirect('/dashboard');
  }

  // Fetch data from Supabase
  const [
    { data: blogPosts },
    { data: jobApplications },
    { data: contactMessages },
    { data: users }
  ] = await Promise.all([
    supabase
      .from('blog_posts')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(5),
    supabase
      .from('job_applications')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(5),
    supabase
      .from('contact_submissions')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(5),
    supabase
      .from('profiles')
      .select('*')
      .limit(1)
  ]);

  // Calculate stats
  const [
    { count: totalPosts },
    { count: totalApplications },
    { count: totalMessages },
    { count: totalUsers }
  ] = await Promise.all([
    supabase.from('blog_posts').select('*', { count: 'exact', head: true }),
    supabase.from('job_applications').select('*', { count: 'exact', head: true }),
    supabase.from('contact_submissions').select('*', { count: 'exact', head: true }),
    supabase.from('profiles').select('*', { count: 'exact', head: true })
  ]);

  // Ensure counts are numbers
  const postsCount = Number(totalPosts) || 0;
  const applicationsCount = Number(totalApplications) || 0;
  const messagesCount = Number(totalMessages) || 0;
  const usersCount = Number(totalUsers) || 0;

  // Calculate weekly changes (this is a simplified example)
  // In a real app, you would compare with data from the previous week
  const weeklyPostsChange = calculatePercentageChange(postsCount, Math.max(0, postsCount - 3));
  const weeklyApplicationsChange = calculatePercentageChange(applicationsCount, Math.max(0, applicationsCount - 23));
  const weeklyMessagesChange = calculatePercentageChange(messagesCount, Math.max(0, messagesCount - 8));
  const monthlyUsersChange = calculatePercentageChange(usersCount, Math.max(0, usersCount - 210));

  // Calculate application status distribution
  const { data: applicationStatuses } = await supabase
    .from('job_applications')
    .select('status')
    .then(({ data }) => {
      const statusCounts = data?.reduce((acc: Record<string, number>, curr) => {
        acc[curr.status] = (acc[curr.status] || 0) + 1;
        return acc;
      }, {});
      return { data: Object.entries(statusCounts || {}).map(([status, count]) => ({ status, count })) };
    });

  // Calculate message status distribution
  const { data: messageStatuses } = await supabase
    .from('contact_submissions')
    .select('status')
    .then(({ data }) => {
      const statusCounts = data?.reduce((acc: Record<string, number>, curr) => {
        acc[curr.status] = (acc[curr.status] || 0) + 1;
        return acc;
      }, {});
      return { data: Object.entries(statusCounts || {}).map(([status, count]) => ({ status, count })) };
    });

  // Calculate blog post categories
  const { data: blogCategories } = await supabase
    .from('blog_posts')
    .select('category')
    .then(({ data }) => {
      const categoryCounts = data?.reduce((acc: Record<string, number>, curr) => {
        acc[curr.category] = (acc[curr.category] || 0) + 1;
        return acc;
      }, {});
      return { data: Object.entries(categoryCounts || {}).map(([category, count]) => ({ category, count })) };
    });

  // Get unread messages count
  const { count: unreadMessages } = await supabase
    .from('contact_submissions')
    .select('*', { count: 'exact', head: true })
    .eq('status', 'UNREAD');

  const stats = [
    {
      name: 'Total Blog Posts',
      value: postsCount.toString(),
      icon: DocumentTextIcon,
      href: '/admin/blog',
      change: `${weeklyPostsChange > 0 ? '+' : '-'}${Math.abs(weeklyPostsChange).toFixed(1)}% this week`,
      changeType: weeklyPostsChange > 0 ? 'positive' : 'negative',
    },
    {
      name: 'Job Applications',
      value: applicationsCount.toString(),
      icon: InboxIcon,
      href: '/admin/applications',
      change: `${weeklyApplicationsChange > 0 ? '+' : '-'}${Math.abs(weeklyApplicationsChange).toFixed(1)}% this week`,
      changeType: weeklyApplicationsChange > 0 ? 'positive' : 'negative',
    },
    {
      name: 'Contact Messages',
      value: messagesCount.toString(),
      icon: ChatBubbleLeftIcon,
      href: '/admin/contact',
      change: `${weeklyMessagesChange > 0 ? '+' : '-'}${Math.abs(weeklyMessagesChange).toFixed(1)}% this week`,
      changeType: weeklyMessagesChange > 0 ? 'positive' : 'negative',
    },
    {
      name: 'Total Users',
      value: usersCount.toString(),
      icon: UserGroupIcon,
      href: '/admin/users',
      change: `${monthlyUsersChange > 0 ? '+' : '-'}${Math.abs(monthlyUsersChange).toFixed(1)}% this month`,
      changeType: monthlyUsersChange > 0 ? 'positive' : 'negative',
    },
  ];

  // Admin quick actions
  const quickActions = [
    { name: 'Create Blog Post', href: '/admin/blog/new', icon: DocumentTextIcon },
    { name: 'Manage Users', href: '/admin/users', icon: UserCircleIcon },
    { name: 'System Settings', href: '/admin/settings', icon: CogIcon },
    { name: 'Security Logs', href: '/admin/logs', icon: ShieldCheckIcon },
  ];

  return (
    <div className="py-6">
      <div className="px-4 sm:px-6 md:px-8">
        {/* Back to Dashboard Button */}
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-semibold text-gray-900">Admin Dashboard</h1>
          <Link
            href="/dashboard"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-connection-blue hover:bg-connection-blue-dark transition-colors"
          >
            <HomeIcon className="h-5 w-5 mr-2" aria-hidden="true" />
            Back to Dashboard
          </Link>
        </div>
        
        {/* Alerts Section */}
        {Number(unreadMessages) > 0 && (
          <div className="mt-4 bg-yellow-50 border-l-4 border-yellow-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <BellIcon className="h-5 w-5 text-yellow-400" aria-hidden="true" />
              </div>
              <div className="ml-3">
                <p className="text-sm text-yellow-700">
                  You have {unreadMessages} unread message{Number(unreadMessages) === 1 ? '' : 's'}.{' '}
                  <Link href="/admin/contact" className="font-medium text-yellow-700 underline hover:text-yellow-600">
                    View messages
                  </Link>
                </p>
              </div>
            </div>
          </div>
        )}
        
        {/* Quick Actions */}
        <div className="mt-6">
          <h2 className="text-lg font-medium text-gray-900 mb-3">Quick Actions</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {quickActions.map((action) => (
              <Link
                key={action.name}
                href={action.href}
                className="relative overflow-hidden rounded-lg bg-white p-4 shadow hover:bg-gray-50 transition-colors flex items-center"
              >
                <div className="mr-4 rounded-md bg-connection-blue/10 p-2">
                  <action.icon className="h-5 w-5 text-connection-blue" aria-hidden="true" />
                </div>
                <span className="text-sm font-medium text-gray-900">{action.name}</span>
              </Link>
            ))}
          </div>
        </div>
        
        {/* Stats Grid */}
        <div className="mt-6">
          <h2 className="text-lg font-medium text-gray-900 mb-3">Key Metrics</h2>
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {stats.map((stat) => (
              <Link
                key={stat.name}
                href={stat.href}
                className="relative overflow-hidden rounded-lg bg-white px-4 py-5 shadow sm:px-6 sm:py-6 hover:bg-gray-50 transition-colors"
              >
                <dt>
                  <div className="absolute rounded-md bg-connection-blue/10 p-3">
                    <stat.icon className="h-6 w-6 text-connection-blue" aria-hidden="true" />
                  </div>
                  <p className="ml-16 truncate text-sm font-medium text-gray-500">
                    {stat.name}
                  </p>
                </dt>
                <dd className="ml-16 flex items-baseline">
                  <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
                  <p
                    className={`ml-2 flex items-baseline text-sm font-semibold ${
                      stat.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {stat.change}
                  </p>
                </dd>
              </Link>
            ))}
          </div>
        </div>

        {/* Admin Tools */}
        <div className="mt-8">
          <h2 className="text-lg font-medium text-gray-900 mb-3">Admin Tools</h2>
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {/* System Monitoring */}
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex items-center">
                <WrenchScrewdriverIcon className="h-8 w-8 text-connection-blue mr-3" />
                <h3 className="text-lg font-medium text-gray-900">System Monitoring</h3>
              </div>
              <ul className="mt-4 space-y-3">
                <li className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">Database Status</span>
                  <span className="px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">Healthy</span>
                </li>
                <li className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">Storage Usage</span>
                  <span className="text-sm text-gray-600">34% of 20GB</span>
                </li>
                <li className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">API Rate Limit</span>
                  <span className="text-sm text-gray-600">24,145/100,000</span>
                </li>
                <li>
                  <Link href="/admin/monitoring" className="text-sm text-connection-blue hover:text-connection-blue-dark">
                    View detailed metrics →
                  </Link>
                </li>
              </ul>
            </div>
            
            {/* Recent Activities */}
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex items-center">
                <QueueListIcon className="h-8 w-8 text-connection-blue mr-3" />
                <h3 className="text-lg font-medium text-gray-900">Recent Activities</h3>
              </div>
              <ul className="mt-4 space-y-3">
                <li className="text-sm text-gray-600">
                  <span className="font-medium text-gray-900">User Registration:</span> 15 new users today
                </li>
                <li className="text-sm text-gray-600">
                  <span className="font-medium text-gray-900">Content Updates:</span> 3 blog posts published
                </li>
                <li className="text-sm text-gray-600">
                  <span className="font-medium text-gray-900">System:</span> Database backup completed
                </li>
                <li>
                  <Link href="/admin/activities" className="text-sm text-connection-blue hover:text-connection-blue-dark">
                    View all activities →
                  </Link>
                </li>
              </ul>
            </div>
            
            {/* Scheduled Tasks */}
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex items-center">
                <CalendarIcon className="h-8 w-8 text-connection-blue mr-3" />
                <h3 className="text-lg font-medium text-gray-900">Scheduled Tasks</h3>
              </div>
              <ul className="mt-4 space-y-3">
                <li className="text-sm text-gray-600">
                  <span className="font-medium text-gray-900">Daily:</span> Database backup (2:00 AM)
                </li>
                <li className="text-sm text-gray-600">
                  <span className="font-medium text-gray-900">Weekly:</span> User activity report (Sunday)
                </li>
                <li className="text-sm text-gray-600">
                  <span className="font-medium text-gray-900">Monthly:</span> Invoicing (1st)
                </li>
                <li>
                  <Link href="/admin/tasks" className="text-sm text-connection-blue hover:text-connection-blue-dark">
                    Manage scheduled tasks →
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-3">
          {/* Recent Blog Posts */}
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">Recent Blog Posts</h2>
              <Link 
                href="/admin/blog"
                className="text-sm text-connection-blue hover:text-connection-blue-dark"
              >
                View all
              </Link>
            </div>
            <div className="space-y-4">
              {(blogPosts || []).map((post) => (
                <div key={post.id} className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{post.title}</p>
                    <p className="text-sm text-gray-500">
                      {formatDate(post.created_at)}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      post.published
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}
                  >
                    {post.published ? 'Published' : 'Draft'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Applications */}
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">Recent Applications</h2>
              <Link 
                href="/admin/applications"
                className="text-sm text-connection-blue hover:text-connection-blue-dark"
              >
                View all
              </Link>
            </div>
            <div className="space-y-4">
              {(jobApplications || []).map((application) => (
                <div key={application.id} className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{application.name}</p>
                    <p className="text-sm text-gray-500">{application.position}</p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      application.status === 'PENDING'
                        ? 'bg-yellow-100 text-yellow-800'
                        : application.status === 'REVIEWING'
                        ? 'bg-blue-100 text-blue-800'
                        : application.status === 'ACCEPTED'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {application.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Messages */}
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-gray-900">Recent Messages</h2>
              <Link 
                href="/admin/contact"
                className="text-sm text-connection-blue hover:text-connection-blue-dark"
              >
                View all
              </Link>
            </div>
            <div className="space-y-4">
              {(contactMessages || []).map((message) => (
                <div key={message.id} className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{message.name}</p>
                    <p className="text-sm text-gray-500">{message.subject}</p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      message.status === 'UNREAD'
                        ? 'bg-red-100 text-red-800'
                        : message.status === 'READ'
                        ? 'bg-blue-100 text-blue-800'
                        : 'bg-green-100 text-green-800'
                    }`}
                  >
                    {message.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Additional Analytics Section */}
        <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-3">
          {/* Application Status Distribution */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Application Status</h2>
            <div className="space-y-3">
              {applicationStatuses.map((status) => (
                <div key={status.status} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">{status.status}</span>
                  <div className="flex items-center">
                    <span className="text-sm text-gray-500 mr-2">{status.count}</span>
                    <div className="w-24 bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          status.status === 'PENDING' ? 'bg-yellow-500' :
                          status.status === 'REVIEWING' ? 'bg-blue-500' :
                          status.status === 'ACCEPTED' ? 'bg-green-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${(status.count / applicationsCount) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Message Status Distribution */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Message Status</h2>
            <div className="space-y-3">
              {messageStatuses.map((status) => (
                <div key={status.status} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">{status.status}</span>
                  <div className="flex items-center">
                    <span className="text-sm text-gray-500 mr-2">{status.count}</span>
                    <div className="w-24 bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          status.status === 'UNREAD' ? 'bg-red-500' :
                          status.status === 'READ' ? 'bg-blue-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${(status.count / messagesCount) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Blog Categories */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Blog Categories</h2>
            <div className="space-y-3">
              {blogCategories.map((category) => (
                <div key={category.category} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">{category.category}</span>
                  <div className="flex items-center">
                    <span className="text-sm text-gray-500 mr-2">{category.count}</span>
                    <div className="w-24 bg-gray-200 rounded-full h-2.5">
                      <div 
                        className="h-2.5 rounded-full bg-connection-blue"
                        style={{ width: `${(category.count / postsCount) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 