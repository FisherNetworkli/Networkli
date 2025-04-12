'use client';

import { useEffect, useState } from 'react';
import { isReadOnlyMode } from '../lib/utils';

export function ReadOnlyBanner() {
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    // Only show the banner in read-only mode
    setShowBanner(isReadOnlyMode());
  }, []);

  if (!showBanner) return null;

  return (
    <div className="bg-amber-100 border-l-4 border-amber-500 text-amber-700 p-4 mb-4" role="alert">
      <p className="font-bold">Read-Only Mode</p>
      <p>The site is currently in read-only mode. Some features may be limited.</p>
    </div>
  );
} 