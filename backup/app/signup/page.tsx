'use client';

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { SignupFlow } from '../components/signup/SignupFlow';

export default function SignupPage() {
  const router = useRouter();

  useEffect(() => {
    const checkAuth = async () => {
      const response = await fetch('/api/auth/check');
      const data = await response.json();
      
      if (data.redirect) {
        router.replace(data.redirect);
      }
    };

    checkAuth();
  }, [router]);

  return (
    <div className="min-h-screen bg-gray-100 py-12">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <SignupFlow />
        </div>
      </div>
    </div>
  );
}