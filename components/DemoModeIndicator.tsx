'use client';

import { useState, useEffect } from 'react';
import { BeakerIcon } from '@heroicons/react/24/outline';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import { isDemoModeEnabled } from '@/lib/demo-mode';

export default function DemoModeIndicator() {
  const [demoMode, setDemoMode] = useState(false);
  const supabase = createClientComponentClient();
  
  useEffect(() => {
    // Check if demo mode is enabled
    async function checkDemoMode() {
      const isEnabled = await isDemoModeEnabled(supabase);
      setDemoMode(isEnabled);
    }
    
    checkDemoMode();
    
    // Set up a listener to check for changes every minute
    const interval = setInterval(checkDemoMode, 60000);
    
    return () => clearInterval(interval);
  }, [supabase]);
  
  if (!demoMode) {
    return null;
  }
  
  return (
    <div className="fixed bottom-4 right-4 bg-amber-100 border border-amber-300 text-amber-800 rounded-lg px-4 py-2 flex items-center space-x-2 z-50 shadow-md">
      <BeakerIcon className="h-5 w-5" />
      <span className="font-medium">Demo Mode</span>
    </div>
  );
} 