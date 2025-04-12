import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { 
  UserIcon, 
  BriefcaseIcon, 
  ChatBubbleLeftRightIcon, 
  UserGroupIcon 
} from '@heroicons/react/24/outline';

export default async function UserDashboard() {
  const session = await getServerSession();

  if (!session) {
    redirect('/login');
  }

  const stats = [
    {
      name: 'Profile Views',
      value: '245',
      icon: UserIcon,
      change: '+15%',
      changeType: 'positive',
    },
    {
      name: 'Network Size',
      value: '89',
      icon: UserGroupIcon,
      change: '+5',
      changeType: 'positive',
    },
    {
      name: 'Job Matches',
      value: '12',
      icon: BriefcaseIcon,
      change: '+3',
      changeType: 'positive',
    },
    {
      name: 'Messages',
      value: '28',
      icon: ChatBubbleLeftRightIcon,
      change: '+7',
      changeType: 'positive',
    },
  ];

  return (
    <div className="py-6">
      <div className="px-4 sm:px-6 md:px-8">
        <h1 className="text-2xl font-semibold text-gray-900">My Dashboard</h1>
        
        {/* Stats Grid */}
        <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => (
            <div
              key={stat.name}
              className="relative overflow-hidden rounded-lg bg-white px-4 py-5 shadow sm:px-6 sm:py-6"
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
            </div>
          ))}
        </div>

        {/* Recent Activity */}
        <div className="mt-8">
          <h2 className="text-lg font-medium text-gray-900">Recent Activity</h2>
          <div className="mt-4 overflow-hidden rounded-lg bg-white shadow">
            <ul role="list" className="divide-y divide-gray-200">
              <li className="px-6 py-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      New Connection Request
                    </p>
                    <p className="text-sm text-gray-500">
                      Sarah Miller wants to connect with you
                    </p>
                  </div>
                  <div className="text-sm text-gray-500">
                    5 minutes ago
                  </div>
                </div>
              </li>
              <li className="px-6 py-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      Job Match
                    </p>
                    <p className="text-sm text-gray-500">
                      New position matches your profile: Senior Developer at Tech Co.
                    </p>
                  </div>
                  <div className="text-sm text-gray-500">
                    1 hour ago
                  </div>
                </div>
              </li>
              <li className="px-6 py-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      Profile View
                    </p>
                    <p className="text-sm text-gray-500">
                      Your profile was viewed by a recruiter at Innovation Labs
                    </p>
                  </div>
                  <div className="text-sm text-gray-500">
                    2 hours ago
                  </div>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 