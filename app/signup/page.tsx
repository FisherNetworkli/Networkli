import React from 'react';
import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import SignupForm from './SignupForm';
import { authOptions } from '../api/auth/[...nextauth]/auth';

export default async function SignupPage() {
  const session = await getServerSession(authOptions);

  if (session) {
    // Redirect based on user role
    if (session.user?.role === 'ADMIN') {
      redirect('/admin');
    } else {
      redirect('/dashboard');
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-lg shadow-lg">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Create your account
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Join Networkli and start building meaningful connections
          </p>
        </div>
        <SignupForm />
      </div>
    </div>
  );
} 