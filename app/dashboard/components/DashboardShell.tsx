import React, { useEffect, useState } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { User } from '@supabase/supabase-js';
import { SideNav } from './SideNav';

interface DashboardShellProps {
  children: React.ReactNode;
}

export default function DashboardShell({ children }: DashboardShellProps) {
  const [user, setUser] = useState<User | null>(null);
  const supabase = createClientComponentClient();
  
  useEffect(() => {
    const fetchUser = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setUser(session.user);
      }
    };
    
    fetchUser();
  }, [supabase]);
  
  if (!user) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center">
        <div className="animate-pulse rounded-md bg-muted w-20 h-20 mb-4"></div>
        <p>Loading your dashboard...</p>
      </div>
    );
  }
  
  return (
    <div className="flex min-h-screen flex-col">
      <div className="container flex-1 items-start md:grid md:grid-cols-[220px_minmax(0,1fr)] md:gap-6 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-10">
        <aside className="fixed z-30 -ml-2 hidden h-[calc(100vh)] w-full shrink-0 md:sticky md:block">
          <SideNav />
        </aside>
        <main className="flex w-full flex-col overflow-hidden py-6">
          {children}
        </main>
      </div>
    </div>
  );
} 