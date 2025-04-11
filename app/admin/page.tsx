import Link from 'next/link';
import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { 
  UsersIcon, 
  DocumentTextIcon, 
  InboxIcon, 
  ChartBarIcon,
  ChatBubbleLeftIcon
} from '@heroicons/react/24/outline';

const stats = [
  {
    name: 'Total Blog Posts',
    value: '24',
    icon: DocumentTextIcon,
    href: '/admin/blog',
    change: '+3 this week',
    changeType: 'positive',
  },
  {
    name: 'Job Applications',
    value: '156',
    icon: InboxIcon,
    href: '/admin/applications',
    change: '+23 this week',
    changeType: 'positive',
  },
  {
    name: 'Contact Messages',
    value: '42',
    icon: ChatBubbleLeftIcon,
    href: '/admin/contact',
    change: '+8 this week',
    changeType: 'positive',
  },
  {
    name: 'Total Users',
    value: '2,103',
    icon: ChartBarIcon,
    href: '/admin/users',
    change: '+12% this month',
    changeType: 'positive',
  },
];

const recentPosts = [
  {
    id: 1,
    title: 'Getting Started with Networkli',
    status: 'Published',
    date: '2024-03-10',
  },
  {
    id: 2,
    title: 'Networking Tips for Introverts',
    status: 'Draft',
    date: '2024-03-09',
  },
];

const recentApplications = [
  {
    id: 1,
    name: 'John Smith',
    position: 'Senior Developer',
    status: 'Pending',
    date: '2024-03-10',
  },
  {
    id: 2,
    name: 'Sarah Johnson',
    position: 'Product Manager',
    status: 'Reviewed',
    date: '2024-03-09',
  },
];

const recentMessages = [
  {
    id: 1,
    name: 'Michael Brown',
    subject: 'Partnership Inquiry',
    status: 'Unread',
    date: '2024-03-10',
  },
  {
    id: 2,
    name: 'Emily Davis',
    subject: 'Support Request',
    status: 'Replied',
    date: '2024-03-09',
  },
];

export default async function AdminDashboard() {
  const session = await getServerSession();

  if (!session) {
    redirect('/login');
  }

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
              {recentPosts.map((post) => (
                <div key={post.id} className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{post.title}</p>
                    <p className="text-sm text-gray-500">
                      {new Date(post.date).toLocaleDateString()}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      post.status === 'Published'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}
                  >
                    {post.status}
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
              {recentApplications.map((application) => (
                <div key={application.id} className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{application.name}</p>
                    <p className="text-sm text-gray-500">{application.position}</p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      application.status === 'Pending'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-blue-100 text-blue-800'
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
              {recentMessages.map((message) => (
                <div key={message.id} className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{message.name}</p>
                    <p className="text-sm text-gray-500">{message.subject}</p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      message.status === 'Unread'
                        ? 'bg-red-100 text-red-800'
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
      </div>
    </div>
  );
} 