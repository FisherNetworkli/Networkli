import Link from 'next/link';
import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { 
  ChartBarIcon, 
  UserGroupIcon, 
  CalendarIcon, 
  ChatBubbleLeftIcon,
  DocumentTextIcon,
  InboxIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/outline';
import { prisma } from '@/lib/prisma';
import { authOptions } from '@/lib/auth';

// Helper function to calculate percentage change
function calculatePercentageChange(current: number, previous: number): { value: number; type: 'positive' | 'negative' } {
  if (previous === 0) return { value: 0, type: 'positive' };
  const change = ((current - previous) / previous) * 100;
  return { 
    value: Math.abs(Math.round(change * 10) / 10), 
    type: change >= 0 ? 'positive' : 'negative' 
  };
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
  const session = await getServerSession(authOptions);

  if (!session) {
    redirect('/login');
  }

  // Fetch data from the database
  const [
    blogPosts,
    jobApplications,
    contactMessages,
    users
  ] = await Promise.all([
    prisma.blogPost.findMany({
      orderBy: { createdAt: 'desc' },
      take: 5,
    }),
    prisma.jobApplication.findMany({
      orderBy: { createdAt: 'desc' },
      take: 5,
    }),
    prisma.contactSubmission.findMany({
      orderBy: { createdAt: 'desc' },
      take: 5,
    }),
    prisma.user.findMany({
      take: 1,
    }),
  ]);

  // Calculate stats
  const totalPosts = await prisma.blogPost.count();
  const totalApplications = await prisma.jobApplication.count();
  const totalMessages = await prisma.contactSubmission.count();
  const totalUsers = await prisma.user.count();

  // Calculate weekly changes (this is a simplified example)
  // In a real app, you would compare with data from the previous week
  const weeklyPostsChange = calculatePercentageChange(totalPosts, Math.max(0, totalPosts - 3));
  const weeklyApplicationsChange = calculatePercentageChange(totalApplications, Math.max(0, totalApplications - 23));
  const weeklyMessagesChange = calculatePercentageChange(totalMessages, Math.max(0, totalMessages - 8));
  const monthlyUsersChange = calculatePercentageChange(totalUsers, Math.max(0, totalUsers - 210));

  // Calculate application status distribution
  const applicationStatuses = await prisma.jobApplication.groupBy({
    by: ['status'],
    _count: true,
  });

  // Calculate message status distribution
  const messageStatuses = await prisma.contactSubmission.groupBy({
    by: ['status'],
    _count: true,
  });

  // Calculate blog post categories
  const blogCategories = await prisma.blogPost.groupBy({
    by: ['category'],
    _count: true,
  });

  const stats = [
    {
      name: 'Total Blog Posts',
      value: totalPosts.toString(),
      icon: DocumentTextIcon,
      href: '/admin/blog',
      change: `${weeklyPostsChange.type === 'positive' ? '+' : '-'}${weeklyPostsChange.value}% this week`,
      changeType: weeklyPostsChange.type,
    },
    {
      name: 'Job Applications',
      value: totalApplications.toString(),
      icon: InboxIcon,
      href: '/admin/applications',
      change: `${weeklyApplicationsChange.type === 'positive' ? '+' : '-'}${weeklyApplicationsChange.value}% this week`,
      changeType: weeklyApplicationsChange.type,
    },
    {
      name: 'Contact Messages',
      value: totalMessages.toString(),
      icon: ChatBubbleLeftIcon,
      href: '/admin/contact',
      change: `${weeklyMessagesChange.type === 'positive' ? '+' : '-'}${weeklyMessagesChange.value}% this week`,
      changeType: weeklyMessagesChange.type,
    },
    {
      name: 'Total Users',
      value: totalUsers.toString(),
      icon: ChartBarIcon,
      href: '/admin/users',
      change: `${monthlyUsersChange.type === 'positive' ? '+' : '-'}${monthlyUsersChange.value}% this month`,
      changeType: monthlyUsersChange.type,
    },
  ];

  return (
    <div className="py-6">
      <div className="px-4 sm:px-6 md:px-8">
        <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
        
        {/* Stats Grid */}
        <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
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
              {blogPosts.length > 0 ? (
                blogPosts.map((post) => (
                  <div key={post.id} className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{post.title}</p>
                      <p className="text-sm text-gray-500">
                        {formatDate(post.createdAt)}
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
                ))
              ) : (
                <p className="text-gray-500 text-sm">No blog posts found</p>
              )}
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
              {jobApplications.length > 0 ? (
                jobApplications.map((application) => (
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
                ))
              ) : (
                <p className="text-gray-500 text-sm">No applications found</p>
              )}
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
              {contactMessages.length > 0 ? (
                contactMessages.map((message) => (
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
                ))
              ) : (
                <p className="text-gray-500 text-sm">No messages found</p>
              )}
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
                    <span className="text-sm text-gray-500 mr-2">{status._count}</span>
                    <div className="w-24 bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          status.status === 'PENDING' ? 'bg-yellow-500' :
                          status.status === 'REVIEWING' ? 'bg-blue-500' :
                          status.status === 'ACCEPTED' ? 'bg-green-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${(status._count / totalApplications) * 100}%` }}
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
                    <span className="text-sm text-gray-500 mr-2">{status._count}</span>
                    <div className="w-24 bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          status.status === 'UNREAD' ? 'bg-red-500' :
                          status.status === 'READ' ? 'bg-blue-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${(status._count / totalMessages) * 100}%` }}
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
                    <span className="text-sm text-gray-500 mr-2">{category._count}</span>
                    <div className="w-24 bg-gray-200 rounded-full h-2.5">
                      <div 
                        className="h-2.5 rounded-full bg-connection-blue"
                        style={{ width: `${(category._count / totalPosts) * 100}%` }}
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