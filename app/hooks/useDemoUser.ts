'use client';

import { useState, useEffect } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

/**
 * Hook that returns true if the current user is the designated demo/prospect user
 */
export function useDemoUser() {
  const [isDemoUser, setIsDemoUser] = useState(false);

  useEffect(() => {
    const supabase = createClientComponentClient();
    supabase.auth.getSession().then(({ data: { session } }) => {
      const email = session?.user.email;
      // Ensure NEXT_PUBLIC_PROSPECT_EMAIL env var is set
      if (email === process.env.NEXT_PUBLIC_PROSPECT_EMAIL) {
        setIsDemoUser(true);
      }
    });
  }, []);

  return isDemoUser;
} 