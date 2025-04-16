'use client';

import React, { useState, useEffect } from 'react';
import { SideNav } from './components/SideNav';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname, useRouter } from 'next/navigation';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { Toaster } from 'react-hot-toast';
import Notifications from '@/app/components/Notifications';

// Metadata is handled in a separate metadata.ts file to avoid the "use client" conflict

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState<any>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const supabase = createClientComponentClient();

  useEffect(() => {
    const checkAuth = async () => {
      // Use getUser instead of getSession for better security
      const { data: userData, error: userError } = await supabase.auth.getUser();
      
      if (userError || !userData.user) {
        router.push('/login');
        return;
      }
      
      setUser(userData.user);
      setIsLoading(false);
    };
    
    checkAuth();
  }, [supabase, router]);

  useEffect(() => {
    // Close mobile menu when path changes
    setMobileMenuOpen(false);
    // Also close user menu when path changes
    setUserMenuOpen(false);
  }, [pathname]);

  // Close user menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (userMenuOpen && !target.closest('.user-menu-container')) {
        setUserMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [userMenuOpen]);

  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="animate-spin h-10 w-10 border-4 border-blue-600 rounded-full border-t-transparent"></div>
      </div>
    );
  }

  // Essential navigation items for mobile bottom nav
  const bottomNavItems = [
    {
      name: 'Home',
      href: '/dashboard',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
          <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
          <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </svg>
      )
    },
    {
      name: 'Network',
      href: '/dashboard/network',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
          <circle cx="9" cy="7" r="4"></circle>
          <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
          <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
        </svg>
      )
    },
    {
      name: 'Swipe',
      href: '/dashboard/network/swipe',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
          <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
        </svg>
      )
    },
    {
      name: 'Messages',
      href: '/dashboard/messages',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      )
    },
    {
      name: 'Profile',
      href: '/dashboard/profile',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="12" cy="7" r="4"></circle>
        </svg>
      )
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile drawer navigation */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div 
            className="fixed inset-0 bg-black/30 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
          ></div>
          <div className="fixed inset-y-0 left-0 w-4/5 max-w-sm bg-white shadow-lg overflow-y-auto">
            <div className="p-4 border-b flex items-center justify-between">
              <span className="text-xl font-bold">Menu</span>
              <button 
                onClick={() => setMobileMenuOpen(false)}
                className="p-2 rounded-full hover:bg-gray-100"
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <SideNav />
          </div>
        </div>
      )}
      
      {/* Header - Mobile optimized */}
      <header className="sticky top-0 z-40 bg-white shadow-sm">
        <div className="mx-auto flex h-14 items-center justify-between px-4">
          <div className="flex items-center">
            {/* Mobile menu button */}
            <button 
              type="button" 
              className="mr-2 rounded-md p-2 text-gray-500 md:hidden"
              onClick={() => setMobileMenuOpen(true)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
              </svg>
            </button>
            
            {/* Logo */}
            <Link href="/dashboard" className="flex items-center">
              <Image 
                src="/logos/networkli-logo-blue.png" 
                alt="Networkli Logo" 
                width={150} 
                height={50} 
              />
            </Link>
          </div>
          
          {/* Right side actions */}
          <div className="flex items-center space-x-3">
            {/* Notifications Component */}
            {user && <Notifications userId={user.id} />}
            
            {/* User menu */}
            <div className="relative user-menu-container">
              <button 
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                className="flex items-center focus:outline-none"
                aria-label="User menu"
                title="Click to open user menu"
              >
                <div className="h-8 w-8 overflow-hidden rounded-full bg-gray-200">
                  <Image 
                    src={user?.user_metadata?.avatar_url || '/images/placeholder-avatar.png'} 
                    alt="Profile" 
                    width={32} 
                    height={32}
                    className="h-full w-full object-cover"
                  />
                </div>
              </button>
              
              {/* User dropdown menu */}
              {userMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 rounded-md bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 z-50">
                  <Link 
                    href="/dashboard/profile" 
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => setUserMenuOpen(false)}
                  >
                    Your Profile
                  </Link>
                  <button
                    onClick={async () => {
                      setUserMenuOpen(false);
                      await supabase.auth.signOut();
                      window.location.href = '/';
                    }}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>
      
      <div className="flex">
        {/* Desktop sidebar */}
        <div className="hidden md:flex md:w-64 md:flex-col">
          <div className="fixed inset-y-0 pt-16 w-64 border-r bg-white">
            <SideNav />
          </div>
        </div>
        
        {/* Main content */}
        <main className="flex-1 md:pl-64">
          <div className="mx-auto px-4 pt-6 pb-28 md:pb-10">
            {children}
          </div>
        </main>
      </div>
      
      {/* Mobile bottom navigation bar */}
      <div className="fixed bottom-0 left-0 right-0 z-40 bg-white border-t shadow-lg md:hidden">
        <nav className="flex justify-between">
          {bottomNavItems.map((item) => {
            const isActive = pathname === item.href || 
                           (pathname?.startsWith(item.href) && item.href !== '/dashboard');
            
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex flex-col items-center justify-center py-3 flex-1 ${
                  isActive ? 'text-blue-600' : 'text-gray-500'
                }`}
              >
                <div className={`${isActive ? 'bg-blue-50 p-2 rounded-full' : ''}`}>
                  {item.icon}
                </div>
                <span className="mt-1 text-xs font-medium">{item.name}</span>
              </Link>
            );
          })}
          
          {/* Sign out button for mobile */}
          <button
            onClick={async () => {
              await supabase.auth.signOut();
              window.location.href = '/';
            }}
            className="flex flex-col items-center justify-center py-3 flex-1 text-gray-500"
          >
            <div>
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                <polyline points="16 17 21 12 16 7"></polyline>
                <line x1="21" y1="12" x2="9" y2="12"></line>
              </svg>
            </div>
            <span className="mt-1 text-xs font-medium">Sign Out</span>
          </button>
        </nav>
      </div>
      
      <Toaster position="top-right" />
    </div>
  );
}