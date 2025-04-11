'use client';

import Link from 'next/link';
import { 
  HomeIcon, 
  DocumentTextIcon, 
  InboxIcon, 
  ChatBubbleLeftIcon,
  ArrowLeftOnRectangleIcon 
} from '@heroicons/react/24/outline';
import { signOut } from 'next-auth/react';

const navigation = [
  { name: 'Dashboard', href: '/admin', icon: HomeIcon },
  { name: 'Blog Posts', href: '/admin/blog', icon: DocumentTextIcon },
  { name: 'Applications', href: '/admin/applications', icon: InboxIcon },
  { name: 'Contact Messages', href: '/admin/contact', icon: ChatBubbleLeftIcon },
];

export default function AdminNav() {
  return (
    <div className="hidden md:fixed md:inset-y-0 md:flex md:w-64 md:flex-col">
      <div className="flex min-h-0 flex-1 flex-col bg-connection-blue">
        <div className="flex flex-1 flex-col overflow-y-auto pt-5 pb-4">
          <div className="flex flex-shrink-0 items-center px-4">
            <h1 className="text-xl font-bold text-white">Admin Panel</h1>
          </div>
          <nav className="mt-5 flex-1 space-y-1 px-2">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="group flex items-center px-2 py-2 text-sm font-medium text-white hover:bg-connection-blue-dark rounded-md"
              >
                <item.icon
                  className="mr-3 h-6 w-6 flex-shrink-0 text-white"
                  aria-hidden="true"
                />
                {item.name}
              </Link>
            ))}
            <button
              onClick={() => signOut()}
              className="w-full group flex items-center px-2 py-2 text-sm font-medium text-white hover:bg-connection-blue-dark rounded-md"
            >
              <ArrowLeftOnRectangleIcon
                className="mr-3 h-6 w-6 flex-shrink-0 text-white"
                aria-hidden="true"
              />
              Sign Out
            </button>
          </nav>
        </div>
      </div>
    </div>
  );
} 