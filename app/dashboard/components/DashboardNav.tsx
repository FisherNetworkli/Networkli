'use client';

import Link from 'next/link';
import { 
  HomeIcon, 
  UserIcon, 
  BriefcaseIcon, 
  ChatBubbleLeftRightIcon,
  UserGroupIcon,
  CalendarIcon,
  StarIcon,
  BookmarkIcon,
  HandRaisedIcon,
  ArrowLeftOnRectangleIcon,
  UserPlusIcon
} from '@heroicons/react/24/outline';
import { User } from '@supabase/supabase-js'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Settings, LogOut } from 'lucide-react'
import { useState, useEffect } from 'react'
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { useRouter, usePathname } from 'next/navigation'

// Common navigation items for all users
const commonNavigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'My Profile', href: '/dashboard/profile', icon: UserIcon },
  { name: 'Network', href: '/dashboard/network', icon: UserGroupIcon },
  { name: 'Jobs', href: '/dashboard/jobs', icon: BriefcaseIcon },
  { name: 'Messages', href: '/dashboard/messages', icon: ChatBubbleLeftRightIcon },
];

// Premium-only navigation items
const premiumNavigation = [
  { name: 'Events', href: '/dashboard/events', icon: CalendarIcon },
  { name: 'Groups', href: '/dashboard/groups', icon: UserGroupIcon },
  { name: 'Saved', href: '/dashboard/saved', icon: BookmarkIcon },
  { name: 'Recommended', href: '/dashboard/network/recommended', icon: UserPlusIcon },
  { name: 'Mentorship', href: '/dashboard/mentorship', icon: HandRaisedIcon },
];

interface DashboardNavProps {
  user: User
}

export function DashboardNav({ user }: DashboardNavProps) {
  const [userRole, setUserRole] = useState<string | null>(null);
  const supabase = createClientComponentClient();
  const router = useRouter();
  const pathname = usePathname();
  
  useEffect(() => {
    const fetchUserRole = async () => {
      if (user) {
        const { data: profile } = await supabase
          .from('profiles')
          .select('role')
          .eq('id', user.id)
          .single();
        
        if (profile) {
          setUserRole(profile.role);
        }
      }
    };
    
    fetchUserRole();
  }, [user, supabase]);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.push('/signin');
  };

  const isPremiumUser = userRole === 'premium';

  return (
    <div className="hidden md:fixed md:inset-y-0 md:flex md:w-64 md:flex-col">
      <div className="flex min-h-0 flex-1 flex-col bg-connection-blue">
        <div className="flex flex-1 flex-col overflow-y-auto pt-5 pb-4">
          <div className="flex flex-shrink-0 items-center px-4">
            <h1 className="text-xl font-bold text-white">My Networkli</h1>
          </div>
          <nav className="mt-5 flex-1 space-y-1 px-2">
            {/* Common navigation items for all users */}
            {commonNavigation.map((item) => (
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
            
            {/* Premium-only navigation items */}
            {isPremiumUser && (
              <>
                <div className="pt-4 pb-2">
                  <div className="flex items-center px-2">
                    <StarIcon className="h-5 w-5 text-yellow-400" />
                    <p className="ml-2 text-sm font-medium text-white">Premium Features</p>
                  </div>
                </div>
                {premiumNavigation.map((item) => (
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
              </>
            )}
            
            {/* Admin link if user is admin */}
            {userRole === 'admin' && (
              <Link
                href="/admin"
                className="group flex items-center px-2 py-2 text-sm font-medium text-white hover:bg-connection-blue-dark rounded-md"
              >
                <HomeIcon
                  className="mr-3 h-6 w-6 flex-shrink-0 text-white"
                  aria-hidden="true"
                />
                Admin Dashboard
              </Link>
            )}
            
            {/* Upgrade to premium link if not premium user */}
            {!isPremiumUser && (
              <Link
                href="/pricing"
                className="mt-4 group flex items-center px-2 py-2 text-sm font-medium bg-yellow-500 text-white hover:bg-yellow-600 rounded-md"
              >
                <StarIcon
                  className="mr-3 h-6 w-6 flex-shrink-0 text-white"
                  aria-hidden="true"
                />
                Upgrade to Premium
              </Link>
            )}
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  className="w-full group flex items-center px-2 py-2 text-sm font-medium text-white hover:bg-connection-blue-dark rounded-md"
                >
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={user.user_metadata?.avatar_url} alt={user.email || ''} />
                    <AvatarFallback>{user.email?.charAt(0).toUpperCase()}</AvatarFallback>
                  </Avatar>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end" forceMount>
                <DropdownMenuLabel>
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{user.user_metadata?.full_name || user.email}</p>
                    <p className="text-xs leading-none text-muted-foreground">{user.email}</p>
                    {isPremiumUser && (
                      <p className="text-xs text-yellow-500 flex items-center mt-1">
                        <StarIcon className="h-3 w-3 mr-1" /> Premium Member
                      </p>
                    )}
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link href="/dashboard/settings" className="cursor-pointer">
                    <Settings className="mr-2 h-4 w-4" />
                    Settings
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleSignOut} className="cursor-pointer">
                  <LogOut className="mr-2 h-4 w-4" />
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </nav>
        </div>
      </div>
    </div>
  );
} 